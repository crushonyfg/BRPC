# calib/delta_gp.py
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, Literal, Optional, Sequence, Union
import math
import torch

from .kernels import Kernel, make_kernel
from .utils import normal_logpdf

# ------------------------- Online exact GP (with rank-1 append) -------------------------
@dataclass
class OnlineGPState:
    """
    Online exact GP state for discrepancy δ(x).

    update_mode:
      - 'exact_full': rebuild full K and Cholesky each append (O(t^3))
      - 'exact_rank1': rank-1 Cholesky append using previous factor (O(t^2) per new point)
      - 'sparse_inducing': delegates to SVGPState (gpytorch) via an internal adapter

    hyperparam_mode:
      - 'fixed': kernel hyperparameters are held constant
      - 'fit': refit hyperparameters via ML-II when calling refit_hyperparams()
               or when append(..., maybe_refit=True)
    """
    X: torch.Tensor  # [t, dx]
    y: torch.Tensor  # [t]
    kernel: Kernel
    # noise: float
    noise: Union[float, torch.Tensor]
    update_mode: Literal['exact_full', 'exact_rank1', 'sparse_inducing'] = 'exact_rank1'
    hyperparam_mode: Literal['fixed', 'fit'] = 'fixed'
    cache: Dict[str, Any] = field(default_factory=dict)
    Z: Optional[torch.Tensor] = None  # for sparse_inducing adapter (inducing inputs)

    # --- internal helpers ---
    def _kernel_matrix(self, X: torch.Tensor) -> torch.Tensor:
        K = self.kernel.cov(X, X)
        t = X.shape[0]
        if isinstance(self.noise, torch.Tensor):
            jitter = 1e-8
            K = K + torch.diag(self.noise.to(X.device, X.dtype)+jitter)
        else:
            K = K + (self.noise + 1e-8) * torch.eye(t, dtype=X.dtype, device=X.device)
        return K

    def _recompute_cache_full(self) -> None:
        if self.X.numel() == 0:
            self.cache.clear()
            return
        K = self._kernel_matrix(self.X)
        L = torch.linalg.cholesky(K)
        alpha = torch.cholesky_solve(self.y[:, None], L).squeeze(-1)  # [t]
        self.cache["L"] = L
        self.cache["alpha"] = alpha

    def _append_rank1_single(self, x_new: torch.Tensor, y_new: torch.Tensor) -> None:
        """
        Rank-1 Cholesky append for a single new point.
        """
        t = self.X.shape[0]
        if t == 0:
            self.X = x_new[None, :]
            self.y = y_new[None]
            self._recompute_cache_full()
            return

        X_old = self.X
        self.X = torch.cat([self.X, x_new[None, :]], dim=0)
        self.y = torch.cat([self.y, y_new[None]], dim=0)

        if "L" not in self.cache:
            self._recompute_cache_full()
            return
        L = self.cache["L"]  # [t,t]

        k = self.kernel.cov(x_new[None, :], X_old).squeeze(0)  # [t]
        if isinstance(self.noise, torch.Tensor):
            kxx = self.kernel.cov(x_new[None, :], x_new[None, :]).squeeze() + torch.diag(self.noise.to(x_new.device, x_new.dtype)+1e-8)
        else:
            kxx = self.kernel.cov(x_new[None, :], x_new[None, :]).squeeze() + (self.noise + 1e-8)

        v = torch.linalg.solve_triangular(L, k[:, None], upper=False).squeeze(-1)
        residual_var = kxx - (v @ v)
        residual_var = torch.clamp(residual_var, min=1e-14, max=self.kernel.variance)
        diag_new = torch.sqrt(residual_var)

        L_new = torch.zeros(t + 1, t + 1, dtype=L.dtype, device=L.device)
        L_new[:t, :t] = L
        L_new[:t, t] = v
        L_new[t, t] = diag_new

        y_aug = self.y
        alpha_new = torch.cholesky_solve(y_aug[:, None], L_new).squeeze(-1)

        self.cache["L"] = L_new
        self.cache["alpha"] = alpha_new

    def _recompute_cache(self) -> None:
        if self.update_mode in ('exact_full', 'exact_rank1'):
            # For recompute, both modes just rebuild fully
            self._recompute_cache_full()
        elif self.update_mode == 'sparse_inducing':
            raise RuntimeError("Use SVGPState adapter for sparse_inducing mode.")
        else:
            raise ValueError(f"Unknown update_mode: {self.update_mode}")

    # --- public API ---
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.update_mode == 'sparse_inducing':
            raise RuntimeError("This OnlineGPState is not in SVGP mode. Use SVGPState for predictions.")
        if self.X.numel() == 0:
            return torch.zeros(x.shape[0], dtype=x.dtype, device=x.device), self.kernel.diag(x)
        if "L" not in self.cache:
            self._recompute_cache()
        L: torch.Tensor = self.cache["L"]
        alpha: torch.Tensor = self.cache["alpha"]
        kx = self.kernel.cov(x, self.X)  # [b, t]
        mu = kx @ alpha  # [b]
        v = torch.cholesky_solve(kx.transpose(0, 1), L)  # [t,b]
        var = self.kernel.diag(x) - (kx * v.transpose(0, 1)).sum(-1)
        var = var.clamp_min(1e-12)
        return mu, var

    def log_predictive(self, y: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return (mu, var, logpdf) for δ at x for observation y.
        """
        x_ = x if x.dim() == 2 else x[None, :]
        mu, var = self.predict(x_)
        lp = normal_logpdf(y if y.dim() > 0 else y[None], mu[:, None], var[:, None]).squeeze(-1)
        return mu, var, lp

    def append(self, x_new: torch.Tensor, y_new: torch.Tensor, maybe_refit: bool = False) -> "OnlineGPState":
        if self.update_mode == 'sparse_inducing':
            raise RuntimeError("Use SVGPState adapter for sparse_inducing mode.")
        if x_new.dim() == 1:
            if self.update_mode == 'exact_rank1':
                self._append_rank1_single(x_new, y_new)
            elif self.update_mode == 'exact_full':
                if self.X.numel() == 0:
                    self.X = x_new[None, :]
                    self.y = y_new[None]
                else:
                    self.X = torch.cat([self.X, x_new[None, :]], dim=0)
                    self.y = torch.cat([self.y, y_new[None]], dim=0)
                self._recompute_cache_full()
        else:
            # Batch append: loop rank-1 for each new point for simplicity
            assert x_new.dim() == 2 and y_new.dim() == 1 and x_new.shape[0] == y_new.shape[0]
            if self.update_mode == 'exact_full':
                self.X = torch.cat([self.X, x_new], dim=0) if self.X.numel() > 0 else x_new.clone()
                self.y = torch.cat([self.y, y_new], dim=0) if self.y.numel() > 0 else y_new.clone()
                self._recompute_cache_full()
            elif self.update_mode == 'exact_rank1':
                for i in range(x_new.shape[0]):
                    self._append_rank1_single(x_new[i], y_new[i])
            else:
                raise ValueError(f"Unknown update_mode: {self.update_mode}")

        if self.hyperparam_mode == 'fit' and maybe_refit:
            self.refit_hyperparams()
        return self

    def append_batch(self, X_new: torch.Tensor, y_new: torch.Tensor, maybe_refit: bool = False) -> None:
        """批量添加新数据点到GP状态"""
        if X_new.shape[0] == 0:
            return
        
        if self.X.numel() == 0:
            self.X = X_new.clone()
            self.y = y_new.clone()
            self._recompute_cache_full()
            return
        
        # 批量添加到现有数据
        self.X = torch.cat([self.X, X_new], dim=0)
        self.y = torch.cat([self.y, y_new], dim=0)
        
        if self.update_mode == "exact_full":
            self._recompute_cache_full()
        elif self.update_mode == "exact_rank1":
            # 对于批量数据，使用full recompute更稳定
            self._recompute_cache_full()
        
        if maybe_refit and self.hyperparam_mode == "fit":
            self.refit_hyperparams()

        # ---------------- Hyperparameter refit (optional) ----------------
    def refit_hyperparams(self, max_iter: int = 100, lr: float = 0.1, fit_noise: bool = True) -> None:
        """
        Refit kernel hyperparameters by maximizing the GP log marginal likelihood (ML-II).
        Supports vector lengthscales.

        Args:
            max_iter: number of optimization steps
            lr: learning rate
            fit_noise: whether to refit noise as well
        """
        if self.X.numel() == 0:
            return

        device, dtype = self.X.device, self.X.dtype
        # Initialize trainable log-parameters
        lengthscale = getattr(self.kernel, 'lengthscale', torch.tensor([1.0], dtype=dtype, device=device))
        variance = getattr(self.kernel, 'variance', 1.0)
        if not isinstance(lengthscale, torch.Tensor):
            lengthscale = torch.tensor([float(lengthscale)], dtype=dtype, device=device)

        log_ls = torch.nn.Parameter(torch.log(lengthscale.to(device, dtype)))
        log_var = torch.nn.Parameter(torch.log(torch.tensor(float(variance), dtype=dtype, device=device)))
        if fit_noise:
            log_noise = torch.nn.Parameter(torch.log(torch.tensor(self.noise, dtype=dtype, device=device)))
            params = [log_ls, log_var, log_noise]
        else:
            log_noise = None
            params = [log_ls, log_var]

        opt = torch.optim.Adam(params, lr=lr)

        for _ in range(max_iter):
            opt.zero_grad()

            # Use current values without detach (to keep graph)
            ls_val = torch.exp(log_ls)            # [dx]
            var_val = torch.exp(log_var)          # scalar
            noise_val = torch.exp(log_noise) if fit_noise else torch.tensor(self.noise, dtype=dtype, device=device)

            # Temporarily plug values into kernel for K computation
            orig_ls, orig_var = self.kernel.lengthscale, self.kernel.variance
            self.kernel.lengthscale = ls_val
            self.kernel.variance = var_val

            K = self._kernel_matrix(self.X)
            L = torch.linalg.cholesky(K)
            alpha = torch.cholesky_solve(self.y[:, None], L)

            # log marginal likelihood
            mll = -0.5 * (self.y[:, None].T @ alpha).squeeze()
            mll += -torch.log(torch.diagonal(L)).sum()
            mll += -0.5 * self.X.shape[0] * torch.log(torch.tensor(2.0 * math.pi, dtype=dtype, device=device))

            # restore original kernel params (do not detach yet)
            self.kernel.lengthscale, self.kernel.variance = orig_ls, orig_var

            loss = -mll
            loss.backward()
            opt.step()

        # After optimization: write back final detached values
        self.kernel.lengthscale = torch.exp(log_ls).detach()
        self.kernel.variance = float(torch.exp(log_var).detach().item())
        if fit_noise and log_noise is not None:
            self.noise = float(torch.exp(log_noise).detach().item())

        # Recompute cache with updated hyperparameters
        self._recompute_cache_full()


import gpytorch

class DeltaExactGP(gpytorch.models.ExactGP):
    def __init__(self, X: torch.Tensor, y: torch.Tensor, likelihood: gpytorch.likelihoods.Likelihood, kernel: gpytorch.kernels.Kernel): # [t,dx],[t],Likelihood,Kernel
        super().__init__(X, y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = kernel

    def forward(self, x: torch.Tensor):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

import math
from dataclasses import dataclass
@dataclass
class GPyTorchDeltaState:
    """
    Exact GP state for discrepancy δ(x) using gpytorch.
    No online updates, no rank-1 updates.
    """
    X: torch.Tensor  # [t,dx]
    y: torch.Tensor  # [t]
    kernel: gpytorch.kernels.Kernel
    noise: Union[float, torch.Tensor]
    model: DeltaExactGP=None
    likelihood: gpytorch.likelihoods.Likelihood=None

    def __post_init__(self):
        self._build_model()

    def _build_model(self):
        if isinstance(self.noise, torch.Tensor):
            self.likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=self.noise.to(self.X.device, self.X.dtype), learn_additional_noise=True)
        else:
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
            # self.likelihood.noise = torch.tensor(max(float(self.noise), 1e-8), dtype=self.X.dtype, device=self.X.device)

        self.model = DeltaExactGP(self.X, self.y, likelihood=self.likelihood, kernel=self.kernel).to(self.X.device, self.X.dtype)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.to(self.X.device, self.X.dtype)
        self.model.eval()
        self.likelihood.eval()
        with gpytorch.settings.fast_pred_var():
            pred = self.likelihood(self.model(x))
            mu = pred.mean
            var = pred.variance.clamp_min(1e-12)
        return mu, var

    @torch.no_grad()
    def log_predictive(self, y: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.model.eval()
        self.likelihood.eval()
        with gpytorch.settings.fast_pred_var():
            pred = self.likelihood(self.model(x))
            mu = pred.mean
            var = pred.variance.clamp_min(1e-12)
            logpdf = pred.log_prob(y)
        return mu, var, logpdf

def fit_gpytorch_delta(state: GPyTorchDeltaState, max_iter: int = 100, lr: float = 0.1):
    state.model.train()
    state.likelihood.train()
    opt = torch.optim.Adam(state.model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(state.likelihood, state.model)
    for _ in range(max_iter):
        opt.zero_grad()
        out = state.model(state.X)
        loss = -mll(out, state.y)
        loss.backward()
        opt.step()
    return state


@dataclass
class GPyTorchScaleRBFHyperSpec:
    lengthscale: torch.Tensor
    variance: float
    noise: float


def _normalize_scale_rbf_lengthscale(lengthscale, dx: int, *, device, dtype) -> torch.Tensor:
    ls = torch.as_tensor(lengthscale, device=device, dtype=dtype).reshape(-1)
    if ls.numel() == 1:
        ls = ls.repeat(dx)
    if ls.numel() != dx:
        raise ValueError(f"lengthscale dimension mismatch: expected {dx}, got {ls.numel()}")
    return ls.clamp_min(1e-8)


def make_scale_rbf_hyper_spec(
    lengthscale,
    variance: float,
    noise: float,
    *,
    dx: int,
    device,
    dtype,
) -> GPyTorchScaleRBFHyperSpec:
    return GPyTorchScaleRBFHyperSpec(
        lengthscale=_normalize_scale_rbf_lengthscale(lengthscale, dx, device=device, dtype=dtype),
        variance=max(float(variance), 1e-8),
        noise=max(float(noise), 1e-8),
    )


def build_scale_rbf_kernel(spec: GPyTorchScaleRBFHyperSpec, dx: int) -> gpytorch.kernels.Kernel:
    kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=dx))
    kernel.base_kernel.lengthscale = spec.lengthscale.reshape(1, -1)
    kernel.outputscale = float(spec.variance)
    return kernel


def fit_scale_rbf_gpytorch_hyper(
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    noise: float,
    lengthscale_init=1.0,
    variance_init: float = 0.1,
    max_iter: int = 80,
) -> GPyTorchScaleRBFHyperSpec:
    dx = int(X.shape[1])
    spec0 = make_scale_rbf_hyper_spec(
        lengthscale_init,
        variance_init,
        noise,
        dx=dx,
        device=X.device,
        dtype=X.dtype,
    )
    state = GPyTorchDeltaState(
        X=X,
        y=y.reshape(-1),
        kernel=build_scale_rbf_kernel(spec0, dx),
        noise=spec0.noise,
    )
    fitted = fit_gpytorch_delta(state, max_iter=max_iter)
    noise_attr = getattr(fitted.likelihood, 'noise', None)
    if noise_attr is None:
        noise_val = spec0.noise
    else:
        noise_val = float(torch.as_tensor(noise_attr).detach().reshape(-1).mean().item())
    return GPyTorchScaleRBFHyperSpec(
        lengthscale=fitted.model.covar_module.base_kernel.lengthscale.detach().reshape(-1).clone(),
        variance=float(fitted.model.covar_module.outputscale.detach().reshape(()).item()),
        noise=max(noise_val, 1e-8),
    )


class OnlineGPyTorchDeltaState(GPyTorchDeltaState):
    """
    Exact gpytorch discrepancy state with frozen hyperparameters and append-only updates.
    """

    def __init__(self, X: torch.Tensor, y: torch.Tensor, hyper_spec: GPyTorchScaleRBFHyperSpec):
        if X.numel() == 0 or y.numel() == 0:
            raise ValueError('OnlineGPyTorchDeltaState requires non-empty training data')
        self.hyper_spec = GPyTorchScaleRBFHyperSpec(
            lengthscale=hyper_spec.lengthscale.detach().clone(),
            variance=float(hyper_spec.variance),
            noise=max(float(hyper_spec.noise), 1e-8),
        )
        dx = int(X.shape[1])
        kernel = build_scale_rbf_kernel(self.hyper_spec, dx)
        super().__init__(X=X.reshape(-1, dx), y=y.reshape(-1), kernel=kernel, noise=self.hyper_spec.noise)
        self._apply_frozen_hyperparameters()

    def _apply_frozen_hyperparameters(self) -> None:
        self.model.covar_module.base_kernel.lengthscale = self.hyper_spec.lengthscale.reshape(1, -1)
        self.model.covar_module.outputscale = float(self.hyper_spec.variance)
        if hasattr(self.likelihood, 'noise'):
            self.likelihood.noise = float(self.hyper_spec.noise)

    def append(self, X_new: torch.Tensor, y_new: torch.Tensor) -> 'OnlineGPyTorchDeltaState':
        if X_new.numel() == 0 or y_new.numel() == 0:
            return self
        X_new = X_new.to(self.X.device, self.X.dtype).reshape(-1, self.X.shape[1])
        y_new = y_new.to(self.y.device, self.y.dtype).reshape(-1)
        self.X = torch.cat([self.X, X_new], dim=0)
        self.y = torch.cat([self.y, y_new], dim=0)
        self.model.set_train_data(inputs=self.X, targets=self.y, strict=False)
        self._apply_frozen_hyperparameters()
        return self

    def truncate_recent(self, keep_n: int) -> 'OnlineGPyTorchDeltaState':
        keep_n = max(int(keep_n), 0)
        if keep_n <= 0:
            raise ValueError('OnlineGPyTorchDeltaState cannot be truncated to zero observations')
        if keep_n >= int(self.X.shape[0]):
            return self
        self.X = self.X[-keep_n:].clone()
        self.y = self.y[-keep_n:].clone()
        self.model.set_train_data(inputs=self.X, targets=self.y, strict=False)
        self._apply_frozen_hyperparameters()
        return self

    def copy(self) -> 'OnlineGPyTorchDeltaState':
        return OnlineGPyTorchDeltaState(self.X.clone(), self.y.clone(), self.hyper_spec)


def _select_sparse_inducing_points(X: torch.Tensor, num_points: int) -> torch.Tensor:
    n = int(X.shape[0])
    if n <= 0:
        raise ValueError("cannot select inducing points from empty history")
    m = max(1, min(int(num_points), n))
    if m == n:
        idx = torch.arange(n, device=X.device)
    else:
        idx = torch.linspace(0, n - 1, m, device=X.device).round().long().unique(sorted=True)
        if idx.numel() < m:
            idx = torch.arange(m, device=X.device)
    return X[idx].clone()


def _select_dynamic_basis_centers(X: torch.Tensor, num_features: int) -> torch.Tensor:
    n = int(X.shape[0])
    if n <= 0:
        raise ValueError("cannot select dynamic basis centers from empty history")
    m = max(1, min(int(num_features), n))
    if m == n:
        idx = torch.arange(n, device=X.device)
    else:
        idx = torch.linspace(0, n - 1, m, device=X.device).round().long().unique(sorted=True)
        if idx.numel() < m:
            idx = torch.arange(m, device=X.device)
    return X[idx].clone()


def _dynamic_rbf_basis(
    X: torch.Tensor,
    centers: torch.Tensor,
    *,
    lengthscale: torch.Tensor,
) -> torch.Tensor:
    X = X.to(centers.device, centers.dtype)
    ls = _normalize_scale_rbf_lengthscale(lengthscale, centers.shape[1], device=centers.device, dtype=centers.dtype)
    if centers.numel() == 0:
        feats = torch.empty(X.shape[0], 0, device=centers.device, dtype=centers.dtype)
    else:
        scaled = (X[:, None, :] - centers[None, :, :]) / ls.view(1, 1, -1)
        dist2 = scaled.pow(2).sum(dim=-1)
        feats = torch.exp(-0.5 * dist2)
    ones = torch.ones(X.shape[0], 1, device=centers.device, dtype=centers.dtype)
    return torch.cat([ones, feats], dim=1)


class BasisPosteriorDeltaState:
    """
    Full-posterior Bayesian linear discrepancy model on a fixed RBF basis.

    This is the offline analogue of the dynamic basis approximation: we keep the same
    basis family delta(x) = phi(x)^T beta, but recompute the posterior over beta from
    the retained discrepancy history instead of using recursive Kalman-style filtering.
    """

    def __init__(
        self,
        hyper_spec: GPyTorchScaleRBFHyperSpec,
        X: torch.Tensor,
        y: torch.Tensor,
        *,
        num_features: int = 20,
        prior_var_scale: float = 1.0,
    ):
        if X.numel() == 0 or y.numel() == 0:
            raise ValueError("BasisPosteriorDeltaState requires non-empty training data")
        self.hyper_spec = GPyTorchScaleRBFHyperSpec(
            lengthscale=hyper_spec.lengthscale.detach().clone(),
            variance=max(float(hyper_spec.variance), 1e-8),
            noise=max(float(hyper_spec.noise), 1e-8),
        )
        self.device = X.device
        self.dtype = X.dtype
        self.dx = int(X.shape[1])
        self.num_features = max(int(num_features), 1)
        self.prior_var_scale = max(float(prior_var_scale), 1e-6)
        self.X = X.to(self.device, self.dtype).reshape(-1, self.dx).clone()
        self.y = y.to(self.device, self.dtype).reshape(-1).clone()
        self.centers = _select_dynamic_basis_centers(self.X, self.num_features)
        self.state_dim = int(self.centers.shape[0]) + 1
        self._identity = torch.eye(self.state_dim, device=self.device, dtype=self.dtype)
        self.posterior_chol = None
        self.beta_mean = None
        self._build_posterior()

    def _basis(self, X: torch.Tensor) -> torch.Tensor:
        return _dynamic_rbf_basis(X.to(self.device, self.dtype), self.centers, lengthscale=self.hyper_spec.lengthscale)

    def _prior_var(self) -> float:
        return max(self.prior_var_scale * self.hyper_spec.variance, self.hyper_spec.noise)

    def _build_posterior(self) -> None:
        Phi = self._basis(self.X)
        noise = max(self.hyper_spec.noise, 1e-10)
        prior_prec = 1.0 / self._prior_var()
        precision = (Phi.transpose(0, 1) @ Phi) / noise + prior_prec * self._identity
        chol = torch.linalg.cholesky(precision)
        rhs = (Phi.transpose(0, 1) @ self.y) / noise
        beta_mean = torch.cholesky_solve(rhs[:, None], chol).squeeze(-1)
        self.posterior_chol = chol
        self.beta_mean = beta_mean

    def predict(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        Phi = self._basis(X)
        mu = Phi @ self.beta_mean
        solve = torch.cholesky_solve(Phi.transpose(0, 1), self.posterior_chol)
        latent_var = (Phi * solve.transpose(0, 1)).sum(dim=1)
        var = (latent_var + self.hyper_spec.noise).clamp_min(1e-12)
        return mu, var

    def copy(self) -> "BasisPosteriorDeltaState":
        return BasisPosteriorDeltaState(
            hyper_spec=self.hyper_spec,
            X=self.X.clone(),
            y=self.y.clone(),
            num_features=self.num_features,
            prior_var_scale=self.prior_var_scale,
        )


class DynamicBasisDeltaState:
    """
    Small-state dynamic discrepancy filter.

    We approximate the discrepancy by
        delta_t(x) = phi(x)^T beta_t
    and propagate the latent coefficients with covariance inflation / process noise:
        beta_t | D_{1:t-1} ~ N(beta_{t-1}, P_{t-1}/lambda + q I).

    This gives the intended "posterior becomes prior" online semantics while keeping
    the update cost bounded by the basis dimension instead of the history length.
    """

    def __init__(
        self,
        hyper_spec: GPyTorchScaleRBFHyperSpec,
        X_init: torch.Tensor,
        y_init: torch.Tensor,
        *,
        batch_sizes: Optional[Sequence[int]] = None,
        num_features: int = 20,
        forgetting: float = 0.99,
        process_noise_scale: float = 1e-3,
        prior_var_scale: float = 1.0,
    ):
        if X_init.numel() == 0 or y_init.numel() == 0:
            raise ValueError("DynamicBasisDeltaState requires non-empty initialization data")
        self.hyper_spec = GPyTorchScaleRBFHyperSpec(
            lengthscale=hyper_spec.lengthscale.detach().clone(),
            variance=max(float(hyper_spec.variance), 1e-8),
            noise=max(float(hyper_spec.noise), 1e-8),
        )
        self.device = X_init.device
        self.dtype = X_init.dtype
        self.dx = int(X_init.shape[1])
        self.num_features = max(int(num_features), 1)
        self.forgetting = min(max(float(forgetting), 1e-4), 1.0)
        self.process_noise_scale = max(float(process_noise_scale), 0.0)
        self.prior_var_scale = max(float(prior_var_scale), 1e-6)
        self.centers = _select_dynamic_basis_centers(X_init, self.num_features)
        self.state_dim = int(self.centers.shape[0]) + 1
        self.mean = torch.zeros(self.state_dim, device=self.device, dtype=self.dtype)
        self.cov = self._initial_covariance()
        self._identity = torch.eye(self.state_dim, device=self.device, dtype=self.dtype)
        self.X_hist = torch.empty(0, self.dx, device=self.device, dtype=self.dtype)
        self.y_hist = torch.empty(0, device=self.device, dtype=self.dtype)
        self.batch_sizes = []
        self.assimilate_history(X_init, y_init, batch_sizes=batch_sizes)

    def _initial_covariance(self) -> torch.Tensor:
        prior_var = max(self.prior_var_scale * self.hyper_spec.variance, self.hyper_spec.noise)
        return prior_var * torch.eye(self.state_dim, device=self.device, dtype=self.dtype)

    def _basis(self, X: torch.Tensor) -> torch.Tensor:
        return _dynamic_rbf_basis(X.to(self.device, self.dtype), self.centers, lengthscale=self.hyper_spec.lengthscale)

    def _symmetrize_covariance(self) -> None:
        self.cov = 0.5 * (self.cov + self.cov.transpose(0, 1))
        jitter = max(self.hyper_spec.noise, 1e-8)
        self.cov = self.cov + jitter * 1e-8 * self._identity

    def _propagate(self) -> None:
        if self.forgetting < 1.0:
            self.cov = self.cov / self.forgetting
        if self.process_noise_scale > 0.0:
            q = self.process_noise_scale * max(self.hyper_spec.variance, self.hyper_spec.noise)
            self.cov = self.cov + q * self._identity
        self._symmetrize_covariance()

    def _update_single(self, phi: torch.Tensor, y_val: torch.Tensor) -> None:
        phi = phi.to(self.device, self.dtype).reshape(-1)
        y_scalar = y_val.to(self.device, self.dtype).reshape(())
        Pphi = self.cov @ phi
        innovation_var = torch.clamp(phi @ Pphi + self.hyper_spec.noise, min=1e-10)
        gain = Pphi / innovation_var
        innovation = y_scalar - phi.dot(self.mean)
        self.mean = self.mean + gain * innovation
        KH = torch.outer(gain, phi)
        self.cov = (self._identity - KH) @ self.cov @ (self._identity - KH).transpose(0, 1)
        self.cov = self.cov + self.hyper_spec.noise * torch.outer(gain, gain)
        self._symmetrize_covariance()

    def update_batch(self, X_batch: torch.Tensor, y_batch: torch.Tensor, *, propagate_first: bool = True) -> None:
        if X_batch.numel() == 0 or y_batch.numel() == 0:
            return
        X_batch = X_batch.to(self.device, self.dtype).reshape(-1, self.dx)
        y_batch = y_batch.to(self.device, self.dtype).reshape(-1)
        if propagate_first:
            self._propagate()
        Phi = self._basis(X_batch)
        for i in range(Phi.shape[0]):
            self._update_single(Phi[i], y_batch[i])
        if self.X_hist.numel() == 0:
            self.X_hist = X_batch.clone()
            self.y_hist = y_batch.clone()
        else:
            self.X_hist = torch.cat([self.X_hist, X_batch], dim=0)
            self.y_hist = torch.cat([self.y_hist, y_batch], dim=0)
        self.batch_sizes.append(int(X_batch.shape[0]))

    def assimilate_history(
        self,
        X_hist: torch.Tensor,
        y_hist: torch.Tensor,
        *,
        batch_sizes: Optional[Sequence[int]] = None,
    ) -> None:
        X_hist = X_hist.to(self.device, self.dtype).reshape(-1, self.dx)
        y_hist = y_hist.to(self.device, self.dtype).reshape(-1)
        if X_hist.shape[0] != y_hist.shape[0]:
            raise ValueError("history size mismatch for DynamicBasisDeltaState")
        if X_hist.shape[0] == 0:
            return
        if batch_sizes is None or len(batch_sizes) == 0:
            batch_sizes = [int(X_hist.shape[0])]
        offset = 0
        for i, raw_size in enumerate(batch_sizes):
            size = int(raw_size)
            if size <= 0:
                continue
            end = min(offset + size, int(X_hist.shape[0]))
            if end <= offset:
                continue
            self.update_batch(X_hist[offset:end], y_hist[offset:end], propagate_first=(i > 0))
            offset = end
            if offset >= int(X_hist.shape[0]):
                break
        if offset < int(X_hist.shape[0]):
            self.update_batch(X_hist[offset:], y_hist[offset:], propagate_first=(offset > 0))

    def predict(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        Phi = self._basis(X)
        mu = Phi @ self.mean
        cov_proj = Phi @ self.cov
        latent_var = (cov_proj * Phi).sum(dim=1)
        var = (latent_var + self.hyper_spec.noise).clamp_min(1e-12)
        return mu, var

    def copy(self) -> "DynamicBasisDeltaState":
        out = object.__new__(DynamicBasisDeltaState)
        out.hyper_spec = GPyTorchScaleRBFHyperSpec(
            lengthscale=self.hyper_spec.lengthscale.detach().clone(),
            variance=float(self.hyper_spec.variance),
            noise=float(self.hyper_spec.noise),
        )
        out.device = self.device
        out.dtype = self.dtype
        out.dx = self.dx
        out.num_features = self.num_features
        out.forgetting = self.forgetting
        out.process_noise_scale = self.process_noise_scale
        out.prior_var_scale = self.prior_var_scale
        out.centers = self.centers.clone()
        out.state_dim = self.state_dim
        out.mean = self.mean.clone()
        out.cov = self.cov.clone()
        out._identity = self._identity.clone()
        out.X_hist = self.X_hist.clone()
        out.y_hist = self.y_hist.clone()
        out.batch_sizes = list(self.batch_sizes)
        return out


# ------------------------- SVGP (gpytorch) adapter -------------------------
class SVGPState:
    """
    Sparse variational GP state for discrepancy delta(x).

    This wrapper is used for the online-inducing shared discrepancy path. Kernel
    hyperparameters are typically fitted once outside the class and then frozen so
    only the variational posterior is shallow-updated on each new batch.
    """

    def __init__(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        inducing_points: torch.Tensor,
        noise: float = 1e-3,
        lengthscale: Union[float, Sequence[float], torch.Tensor] = 1.0,
        variance: float = 1.0,
        learn_inducing_locations: bool = False,
        freeze_kernel_hyperparams: bool = True,
        freeze_likelihood_noise: bool = True,
        init_steps: int = 40,
        init_lr: float = 0.05,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        try:
            import gpytorch  # noqa
        except Exception as e:
            raise ImportError("gpytorch is required for SVGPState. Please install gpytorch.") from e

        self.device = device or X.device
        self.dtype = dtype or X.dtype
        self.X = X.to(self.device, self.dtype)
        self.y = y.to(self.device, self.dtype)
        self.Z = inducing_points.to(self.device, self.dtype)
        self.noise = float(noise)
        self.lengthscale = lengthscale
        self.variance = float(variance)
        self.learn_inducing_locations = bool(learn_inducing_locations)
        self.freeze_kernel_hyperparams = bool(freeze_kernel_hyperparams)
        self.freeze_likelihood_noise = bool(freeze_likelihood_noise)
        self.num_data_seen = int(self.y.numel())

        self._build_model()
        self._apply_training_constraints()

        if self.X.numel() > 0:
            self.train_steps(
                self.X,
                self.y,
                steps=max(int(init_steps), 0),
                lr=float(init_lr),
                num_data=max(int(self.y.numel()), 1),
            )

    def _build_model(self):
        import gpytorch
        from gpytorch.kernels import ScaleKernel, RBFKernel
        from gpytorch.means import ZeroMean
        from gpytorch.likelihoods import GaussianLikelihood
        from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
        from gpytorch.models import ApproximateGP

        Z = self.Z

        class _SVGPModel(ApproximateGP):
            def __init__(self, Z, init_ls, init_var, learn_inducing_locations):
                variational_distribution = CholeskyVariationalDistribution(Z.shape[0])
                variational_strategy = VariationalStrategy(
                    self,
                    Z,
                    variational_distribution,
                    learn_inducing_locations=learn_inducing_locations,
                )
                super().__init__(variational_strategy)
                self.mean_module = ZeroMean()
                self.base_kernel = RBFKernel(ard_num_dims=Z.shape[1])
                self.covar_module = ScaleKernel(self.base_kernel)
                with torch.no_grad():
                    if isinstance(init_ls, torch.Tensor):
                        self.base_kernel.lengthscale.copy_(init_ls.view(1, -1))
                    else:
                        self.base_kernel.lengthscale.fill_(float(init_ls))
                    self.covar_module.outputscale.fill_(float(init_var))

            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

        init_ls = torch.as_tensor(self.lengthscale, dtype=self.dtype, device=self.device)
        self.model = _SVGPModel(Z, init_ls, self.variance, self.learn_inducing_locations).to(self.device, self.dtype)
        self.likelihood = GaussianLikelihood(noise=self.noise).to(self.device, self.dtype)

    def _apply_training_constraints(self):
        if self.freeze_kernel_hyperparams:
            for name, param in self.model.named_parameters():
                if "variational_strategy.inducing_points" in name and self.learn_inducing_locations:
                    continue
                if "base_kernel" in name or "covar_module.raw_outputscale" in name:
                    param.requires_grad_(False)
        if not self.learn_inducing_locations:
            for name, param in self.model.named_parameters():
                if "variational_strategy.inducing_points" in name:
                    param.requires_grad_(False)
        if self.freeze_likelihood_noise:
            for _, param in self.likelihood.named_parameters():
                param.requires_grad_(False)

    def train_steps(self, X, y, steps=200, lr=0.05, num_data: Optional[int] = None):
        import gpytorch
        steps = max(int(steps), 0)
        if steps == 0 or y.numel() == 0:
            return
        self.model.train()
        self.likelihood.train()
        params = [p for p in self.model.parameters() if p.requires_grad]
        params.extend(p for p in self.likelihood.parameters() if p.requires_grad)
        if len(params) == 0:
            return
        opt = torch.optim.Adam(params, lr=lr)
        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=max(int(num_data or y.numel()), 1))

        for _ in range(steps):
            opt.zero_grad()
            out = self.model(X)
            loss = -mll(out, y)
            loss.backward()
            opt.step()

    def append(self, X_new: torch.Tensor, y_new: torch.Tensor, steps: int = 50, lr: float = 0.05):
        if X_new.dim() == 1:
            X_new = X_new[None, :]
            y_new = y_new[None]
        X_new = X_new.to(self.device, self.dtype)
        y_new = y_new.to(self.device, self.dtype).reshape(-1)
        self.X = torch.cat([self.X, X_new], dim=0) if self.X.numel() > 0 else X_new.clone()
        self.y = torch.cat([self.y, y_new], dim=0) if self.y.numel() > 0 else y_new.clone()
        self.num_data_seen = int(self.y.numel())
        self.train_steps(self.X, self.y, steps=steps, lr=lr, num_data=self.num_data_seen)

    def append_shallow(self, X_new: torch.Tensor, y_new: torch.Tensor, steps: int = 6, lr: float = 0.03):
        if X_new.dim() == 1:
            X_new = X_new[None, :]
            y_new = y_new[None]
        X_new = X_new.to(self.device, self.dtype)
        y_new = y_new.to(self.device, self.dtype).reshape(-1)
        if y_new.numel() == 0:
            return
        self.X = torch.cat([self.X, X_new], dim=0) if self.X.numel() > 0 else X_new.clone()
        self.y = torch.cat([self.y, y_new], dim=0) if self.y.numel() > 0 else y_new.clone()
        self.num_data_seen += int(y_new.numel())
        self.train_steps(X_new, y_new, steps=steps, lr=lr, num_data=max(self.num_data_seen, int(y_new.numel())))

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        import gpytorch
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.likelihood(self.model(x.to(self.device, self.dtype)))
            mu = pred.mean
            var = pred.variance.clamp_min(1e-12)
        return mu, var


class OnlineInducingGPyTorchDeltaState:
    """
    Shared sparse variational discrepancy state with shallow batch updates.

    Hyperparameters are fitted once from an initialization buffer, inducing inputs are
    selected from that same buffer, and each new batch performs only a small number of
    ELBO steps starting from the previous variational posterior.
    """

    def __init__(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        hyper_spec: GPyTorchScaleRBFHyperSpec,
        *,
        num_inducing: int = 20,
        init_steps: int = 40,
        update_steps: int = 6,
        lr: float = 0.03,
        buffer_max_points: int = 256,
        learn_inducing_locations: bool = False,
        inducing_points: Optional[torch.Tensor] = None,
    ):
        if X.numel() == 0 or y.numel() == 0:
            raise ValueError("OnlineInducingGPyTorchDeltaState requires non-empty training data")
        X = X.reshape(-1, X.shape[1])
        y = y.reshape(-1)
        self.hyper_spec = GPyTorchScaleRBFHyperSpec(
            lengthscale=hyper_spec.lengthscale.detach().clone(),
            variance=float(hyper_spec.variance),
            noise=max(float(hyper_spec.noise), 1e-8),
        )
        self.device = X.device
        self.dtype = X.dtype
        self.dx = int(X.shape[1])
        self.num_inducing = max(int(num_inducing), 1)
        self.init_steps = max(int(init_steps), 0)
        self.update_steps = max(int(update_steps), 0)
        self.lr = max(float(lr), 1e-5)
        self.buffer_max_points = max(int(buffer_max_points), 1)
        self.learn_inducing_locations = bool(learn_inducing_locations)
        self.inducing_points = (
            inducing_points.to(self.device, self.dtype).reshape(-1, self.dx).clone()
            if inducing_points is not None
            else _select_sparse_inducing_points(X, self.num_inducing)
        )
        self.svgp = SVGPState(
            X=X.to(self.device, self.dtype),
            y=y.to(self.device, self.dtype),
            inducing_points=self.inducing_points,
            noise=self.hyper_spec.noise,
            lengthscale=self.hyper_spec.lengthscale,
            variance=self.hyper_spec.variance,
            learn_inducing_locations=self.learn_inducing_locations,
            freeze_kernel_hyperparams=True,
            freeze_likelihood_noise=True,
            init_steps=self.init_steps,
            init_lr=self.lr,
            device=self.device,
            dtype=self.dtype,
        )

    def predict(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.svgp.predict(X.to(self.device, self.dtype))

    def append(self, X_new: torch.Tensor, y_new: torch.Tensor) -> 'OnlineInducingGPyTorchDeltaState':
        self.svgp.append_shallow(X_new, y_new, steps=self.update_steps, lr=self.lr)
        return self

    def truncate_recent(self, keep_n: int) -> 'OnlineInducingGPyTorchDeltaState':
        keep_n = max(int(keep_n), 0)
        if keep_n <= 0:
            raise ValueError("OnlineInducingGPyTorchDeltaState cannot be truncated to zero observations")
        X_keep = self.svgp.X[-keep_n:].clone()
        y_keep = self.svgp.y[-keep_n:].clone()
        rebuilt = OnlineInducingGPyTorchDeltaState(
            X=X_keep,
            y=y_keep,
            hyper_spec=self.hyper_spec,
            num_inducing=self.num_inducing,
            init_steps=self.init_steps,
            update_steps=self.update_steps,
            lr=self.lr,
            buffer_max_points=self.buffer_max_points,
            learn_inducing_locations=self.learn_inducing_locations,
        )
        self.__dict__.update(rebuilt.__dict__)
        return self


# ------------------------- Adapter factory -------------------------
def make_gp_state(
    mode: Literal['exact_full', 'exact_rank1', 'sparse_inducing'],
    X: torch.Tensor,
    y: torch.Tensor,
    kernel_or_cfg: Union[Kernel, dict],
    noise: float,
    inducing_points: Optional[torch.Tensor] = None,
    hyperparam_mode: Literal['fixed', 'fit'] = 'fixed',
):
    """
    Helper to construct either OnlineGPState or SVGPState based on mode.

    - For exact_* modes: kernel_or_cfg should be a Kernel instance.
    - For sparse_inducing: kernel_or_cfg can be a dict with keys for 'lengthscale', 'variance'.
    """
    if mode in ('exact_full', 'exact_rank1'):
        if isinstance(kernel_or_cfg, dict):
            from .configs import DeltaKernelConfig
            cfg = DeltaKernelConfig(**kernel_or_cfg)
            k = make_kernel(cfg)  # expects DeltaKernelConfig-like dict
        else:
            k = kernel_or_cfg
        return OnlineGPState(
            X=X, y=y, kernel=k, noise=noise,
            update_mode=mode, hyperparam_mode=hyperparam_mode
        )
    elif mode == 'sparse_inducing':
        if inducing_points is None:
            raise ValueError("inducing_points is required for sparse_inducing mode.")
        if not isinstance(kernel_or_cfg, dict):
            raise ValueError("For sparse_inducing, pass kernel params as dict {'name','lengthscale','variance'}")
        lengthscale = kernel_or_cfg.get('lengthscale', 1.0)
        variance = kernel_or_cfg.get('variance', 1.0)
        return SVGPState(
            X=X, y=y, inducing_points=inducing_points, noise=noise,
            lengthscale=lengthscale, variance=variance,
            device=X.device, dtype=X.dtype
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")


def _scale_rbf_cov_from_spec(
    x1: torch.Tensor,
    x2: torch.Tensor,
    spec: GPyTorchScaleRBFHyperSpec,
) -> torch.Tensor:
    ls = _normalize_scale_rbf_lengthscale(spec.lengthscale, x1.shape[1], device=x1.device, dtype=x1.dtype)
    x1s = x1 / ls.view(1, -1)
    x2s = x2 / ls.view(1, -1)
    dist2 = (x1s[:, None, :] - x2s[None, :, :]).pow(2).sum(dim=-1)
    return float(spec.variance) * torch.exp(-0.5 * dist2)



def _support_union_with_batch(
    existing: torch.Tensor,
    X_batch: torch.Tensor,
    *,
    atol: float = 1e-10,
    rtol: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    X_batch = X_batch.reshape(-1, X_batch.shape[1])
    dx = int(X_batch.shape[1])
    device = X_batch.device
    dtype = X_batch.dtype
    support_rows = []
    if existing is not None and existing.numel() > 0:
        existing = existing.to(device, dtype).reshape(-1, dx)
        support_rows = [existing[i].clone() for i in range(existing.shape[0])]
    batch_indices = []
    new_indices = []
    for i in range(X_batch.shape[0]):
        row = X_batch[i].to(device, dtype)
        found = None
        for j, srow in enumerate(support_rows):
            if bool(torch.isclose(srow, row, atol=atol, rtol=rtol).all().item()):
                found = j
                break
        if found is None:
            found = len(support_rows)
            support_rows.append(row.clone())
            new_indices.append(found)
        batch_indices.append(found)
    support_X = torch.stack(support_rows, dim=0) if support_rows else torch.empty(0, dx, device=device, dtype=dtype)
    return support_X, torch.tensor(batch_indices, device=device, dtype=torch.long), torch.tensor(new_indices, device=device, dtype=torch.long)


class SharedBatchSupportOnlineBPCDeltaState:
    """
    Shared exact online-BPC discrepancy state with one-batch support.

    The state keeps the posterior q_t(f_t) over discrepancy values on the most recent batch
    inputs X_t. To advance to X_{t+1}, it propagates this posterior through the GP prior using
    the cross-covariance K(X_t, X_{t+1}) and then applies the tempered Bayes update with noise
    lambda_delta * sigma_delta^2.
    """

    def __init__(
        self,
        X_init: torch.Tensor,
        y_init: torch.Tensor,
        hyper_spec: GPyTorchScaleRBFHyperSpec,
        *,
        lambda_delta: float = 1.0,
        obs_noise_var: Optional[float] = None,
        add_kernel_noise_to_predict: bool = True,
    ):
        if X_init.numel() == 0 or y_init.numel() == 0:
            raise ValueError("SharedBatchSupportOnlineBPCDeltaState requires non-empty initialization data")
        X_init = X_init.reshape(-1, X_init.shape[1])
        y_init = y_init.reshape(-1)
        self.hyper_spec = GPyTorchScaleRBFHyperSpec(
            lengthscale=hyper_spec.lengthscale.detach().clone(),
            variance=max(float(hyper_spec.variance), 1e-8),
            noise=max(float(hyper_spec.noise), 1e-8),
        )
        self.lambda_delta = max(float(lambda_delta), 1e-6)
        self.obs_noise_var = None if obs_noise_var is None else max(float(obs_noise_var), 1e-8)
        self.add_kernel_noise_to_predict = bool(add_kernel_noise_to_predict)
        self.device = X_init.device
        self.dtype = X_init.dtype
        self.dx = int(X_init.shape[1])
        self.support_X = torch.empty(0, self.dx, device=self.device, dtype=self.dtype)
        self.support_mean = torch.empty(0, device=self.device, dtype=self.dtype)
        self.support_cov = torch.empty(0, 0, device=self.device, dtype=self.dtype)
        self._support_prior_cov = torch.empty(0, 0, device=self.device, dtype=self.dtype)
        self._support_chol = None
        self._replace_support(self._posterior_from_prior_batch(X_init, y_init))

    def _kernel(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return _scale_rbf_cov_from_spec(
            x1.to(self.device, self.dtype).reshape(-1, self.dx),
            x2.to(self.device, self.dtype).reshape(-1, self.dx),
            self.hyper_spec,
        )

    def _stabilize_cov(self, cov: torch.Tensor, *, diag_floor: Optional[float] = None) -> torch.Tensor:
        cov = 0.5 * (cov + cov.transpose(0, 1))
        n = int(cov.shape[0])
        if n == 0:
            return cov
        floor = max(float(diag_floor if diag_floor is not None else self.hyper_spec.noise), 1e-8)
        eye = torch.eye(n, device=self.device, dtype=self.dtype)
        return cov + floor * 1e-6 * eye

    def _effective_obs_noise(self, kernel_noise: Optional[float] = None) -> float:
        if self.obs_noise_var is not None:
            return max(float(self.obs_noise_var), 1e-8)
        base = float(self.hyper_spec.noise if kernel_noise is None else kernel_noise)
        return max(base, 1e-8)

    def _support_noise(self, n: int) -> torch.Tensor:
        return (
            self.lambda_delta * self._effective_obs_noise()
        ) * torch.eye(n, device=self.device, dtype=self.dtype)

    def _replace_support(self, state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> None:
        X_support, mean_support, cov_support = state
        X_support = X_support.to(self.device, self.dtype).reshape(-1, self.dx)
        mean_support = mean_support.to(self.device, self.dtype).reshape(-1)
        cov_support = self._stabilize_cov(cov_support.to(self.device, self.dtype))
        self.support_X = X_support.clone()
        self.support_mean = mean_support.clone()
        self.support_cov = cov_support.clone()
        self._support_prior_cov = self._kernel(self.support_X, self.support_X)
        self._support_chol = torch.linalg.cholesky(self._stabilize_cov(self._support_prior_cov, diag_floor=1e-8))

    def _posterior_from_prior_batch(self, X_batch: torch.Tensor, y_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        X_batch = X_batch.to(self.device, self.dtype).reshape(-1, self.dx)
        y_batch = y_batch.to(self.device, self.dtype).reshape(-1)
        K = self._kernel(X_batch, X_batch)
        S = self._stabilize_cov(K + self._support_noise(X_batch.shape[0]))
        chol_s = torch.linalg.cholesky(S)
        alpha = torch.cholesky_solve(y_batch[:, None], chol_s).squeeze(-1)
        mean = K @ alpha
        solve_k = torch.cholesky_solve(K, chol_s)
        cov = K - K @ solve_k
        cov = self._stabilize_cov(cov)
        return X_batch, mean, cov

    def _propagate_to_batch(self, X_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        X_batch = X_batch.to(self.device, self.dtype).reshape(-1, self.dx)
        if self.support_X.numel() == 0:
            K = self._kernel(X_batch, X_batch)
            zeros = torch.zeros(X_batch.shape[0], device=self.device, dtype=self.dtype)
            return zeros, self._stabilize_cov(K), K
        K_old_new = self._kernel(self.support_X, X_batch)
        solve = torch.cholesky_solve(K_old_new, self._support_chol)
        A = solve.transpose(0, 1)
        K_new = self._kernel(X_batch, X_batch)
        schur = K_new - K_old_new.transpose(0, 1) @ solve
        prior_mean = A @ self.support_mean
        prior_cov = schur + A @ self.support_cov @ A.transpose(0, 1)
        prior_cov = self._stabilize_cov(prior_cov)
        return prior_mean, prior_cov, K_new

    def _posterior_from_propagated_batch(
        self,
        X_batch: torch.Tensor,
        y_batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        X_batch = X_batch.to(self.device, self.dtype).reshape(-1, self.dx)
        y_batch = y_batch.to(self.device, self.dtype).reshape(-1)
        prior_mean, prior_cov, _ = self._propagate_to_batch(X_batch)
        S = self._stabilize_cov(prior_cov + self._support_noise(X_batch.shape[0]))
        chol_s = torch.linalg.cholesky(S)
        innovation = y_batch - prior_mean
        solve_innov = torch.cholesky_solve(innovation[:, None], chol_s).squeeze(-1)
        mean = prior_mean + prior_cov @ solve_innov
        solve_prior = torch.cholesky_solve(prior_cov, chol_s)
        cov = prior_cov - prior_cov @ solve_prior
        cov = self._stabilize_cov(cov)
        return X_batch, mean, cov

    def append(self, X_new: torch.Tensor, y_new: torch.Tensor) -> "SharedBatchSupportOnlineBPCDeltaState":
        self._replace_support(self._posterior_from_propagated_batch(X_new, y_new))
        return self

    def predict(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        X = X.to(self.device, self.dtype).reshape(-1, self.dx)
        if self.support_X.numel() == 0:
            zeros = torch.zeros(X.shape[0], device=self.device, dtype=self.dtype)
            extra_noise = float(self.hyper_spec.noise) if self.add_kernel_noise_to_predict else 0.0
            base_var = torch.full((X.shape[0],), float(self.hyper_spec.variance + extra_noise), device=self.device, dtype=self.dtype)
            return zeros, base_var
        K_xs = self._kernel(X, self.support_X)
        solve = torch.cholesky_solve(K_xs.transpose(0, 1), self._support_chol)
        A = solve.transpose(0, 1)
        mu = A @ self.support_mean
        cond_var = float(self.hyper_spec.variance) - (K_xs * A).sum(dim=1)
        cond_var = cond_var.clamp_min(0.0)
        latent_var = (A @ self.support_cov * A).sum(dim=1).clamp_min(0.0)
        extra_noise = float(self.hyper_spec.noise) if self.add_kernel_noise_to_predict else 0.0
        var = (cond_var + latent_var + extra_noise).clamp_min(1e-12)
        return mu, var

    def copy(self) -> "SharedBatchSupportOnlineBPCDeltaState":
        out = object.__new__(SharedBatchSupportOnlineBPCDeltaState)
        out.hyper_spec = GPyTorchScaleRBFHyperSpec(
            lengthscale=self.hyper_spec.lengthscale.detach().clone(),
            variance=float(self.hyper_spec.variance),
            noise=float(self.hyper_spec.noise),
        )
        out.lambda_delta = float(self.lambda_delta)
        out.obs_noise_var = None if self.obs_noise_var is None else float(self.obs_noise_var)
        out.add_kernel_noise_to_predict = bool(self.add_kernel_noise_to_predict)
        out.device = self.device
        out.dtype = self.dtype
        out.dx = self.dx
        out.support_X = self.support_X.clone()
        out.support_mean = self.support_mean.clone()
        out.support_cov = self.support_cov.clone()
        out._support_prior_cov = self._support_prior_cov.clone()
        out._support_chol = None if self._support_chol is None else self._support_chol.clone()
        return out



class SharedExactOnlineBPCDeltaState:
    """Shared exact online-BPC discrepancy state on an expanding support."""

    def __init__(self, X_init: torch.Tensor, y_init: torch.Tensor, hyper_spec: GPyTorchScaleRBFHyperSpec, *, lambda_delta: float = 1.0, obs_noise_var: Optional[float] = None, add_kernel_noise_to_predict: bool = True):
        if X_init.numel() == 0 or y_init.numel() == 0:
            raise ValueError("SharedExactOnlineBPCDeltaState requires non-empty initialization data")
        X_init = X_init.reshape(-1, X_init.shape[1])
        y_init = y_init.reshape(-1)
        self.hyper_spec = GPyTorchScaleRBFHyperSpec(lengthscale=hyper_spec.lengthscale.detach().clone(), variance=max(float(hyper_spec.variance), 1e-8), noise=max(float(hyper_spec.noise), 1e-8))
        self.lambda_delta = max(float(lambda_delta), 1e-6)
        self.obs_noise_var = None if obs_noise_var is None else max(float(obs_noise_var), 1e-8)
        self.add_kernel_noise_to_predict = bool(add_kernel_noise_to_predict)
        self.device = X_init.device
        self.dtype = X_init.dtype
        self.dx = int(X_init.shape[1])
        self.support_X = torch.empty(0, self.dx, device=self.device, dtype=self.dtype)
        self.support_mean = torch.empty(0, device=self.device, dtype=self.dtype)
        self.support_cov = torch.empty(0, 0, device=self.device, dtype=self.dtype)
        self._support_prior_cov = torch.empty(0, 0, device=self.device, dtype=self.dtype)
        self._support_chol = None
        self._replace_support(*self._posterior_from_empty_prior(X_init, y_init))

    def _kernel(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return _scale_rbf_cov_from_spec(x1.to(self.device, self.dtype).reshape(-1, self.dx), x2.to(self.device, self.dtype).reshape(-1, self.dx), self.hyper_spec)

    def _stabilize_cov(self, cov: torch.Tensor, *, diag_floor: Optional[float] = None) -> torch.Tensor:
        cov = 0.5 * (cov + cov.transpose(0, 1))
        n = int(cov.shape[0])
        if n == 0:
            return cov
        floor = max(float(diag_floor if diag_floor is not None else self.hyper_spec.noise), 1e-8)
        return cov + floor * 1e-6 * torch.eye(n, device=self.device, dtype=self.dtype)

    def _safe_cholesky(self, cov: torch.Tensor, *, diag_floor: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        cov = self._stabilize_cov(cov, diag_floor=diag_floor)
        n = int(cov.shape[0])
        if n == 0:
            return torch.empty(0, 0, device=self.device, dtype=self.dtype), cov
        floor = max(float(diag_floor if diag_floor is not None else self.hyper_spec.noise), 1e-8)
        eye = torch.eye(n, device=self.device, dtype=self.dtype)
        diag_mean = float(torch.diag(cov).mean().detach().item()) if n > 0 else 1.0
        jitter = max(floor * 1e-8, abs(diag_mean) * 1e-10, 1e-10)
        for _ in range(8):
            try:
                repaired = cov + jitter * eye
                chol = torch.linalg.cholesky(repaired)
                return chol, repaired
            except RuntimeError:
                jitter *= 10.0
        evals, evecs = torch.linalg.eigh(0.5 * (cov + cov.transpose(0, 1)))
        eig_floor = max(jitter, floor * 1e-6, 1e-8)
        repaired = (evecs * evals.clamp_min(eig_floor).unsqueeze(0)) @ evecs.transpose(0, 1)
        repaired = self._stabilize_cov(repaired, diag_floor=eig_floor)
        chol = torch.linalg.cholesky(repaired)
        return chol, repaired

    def _effective_obs_noise(self, kernel_noise: Optional[float] = None) -> float:
        if self.obs_noise_var is not None:
            return max(float(self.obs_noise_var), 1e-8)
        base = float(self.hyper_spec.noise if kernel_noise is None else kernel_noise)
        return max(base, 1e-8)

    def _support_noise(self, n: int) -> torch.Tensor:
        return (self.lambda_delta * self._effective_obs_noise()) * torch.eye(n, device=self.device, dtype=self.dtype)

    def _replace_support(self, X_support: torch.Tensor, mean_support: torch.Tensor, cov_support: torch.Tensor) -> None:
        self.support_X = X_support.to(self.device, self.dtype).reshape(-1, self.dx).clone()
        self.support_mean = mean_support.to(self.device, self.dtype).reshape(-1).clone()
        self.support_cov = self._stabilize_cov(cov_support.to(self.device, self.dtype)).clone()
        prior_cov = self._kernel(self.support_X, self.support_X)
        self._support_chol, self._support_prior_cov = self._safe_cholesky(prior_cov, diag_floor=1e-8)

    def _posterior_from_prior(self, X_support: torch.Tensor, prior_mean: torch.Tensor, prior_cov: torch.Tensor, batch_indices: torch.Tensor, y_batch: torch.Tensor):
        idx = batch_indices.to(self.device).long()
        y_batch = y_batch.to(self.device, self.dtype).reshape(-1)
        prior_mean = prior_mean.to(self.device, self.dtype).reshape(-1)
        prior_cov = self._stabilize_cov(prior_cov.to(self.device, self.dtype))
        prior_obs_mean = prior_mean[idx]
        cross = prior_cov[:, idx]
        S = self._stabilize_cov(prior_cov[idx][:, idx] + self._support_noise(idx.shape[0]))
        chol_s, _ = self._safe_cholesky(S, diag_floor=self._effective_obs_noise())
        innovation = y_batch - prior_obs_mean
        mean = prior_mean + cross @ torch.cholesky_solve(innovation[:, None], chol_s).squeeze(-1)
        cov = prior_cov - cross @ torch.cholesky_solve(cross.transpose(0, 1), chol_s)
        return X_support, mean, self._stabilize_cov(cov)

    def _posterior_from_empty_prior(self, X_batch: torch.Tensor, y_batch: torch.Tensor):
        X_batch = X_batch.to(self.device, self.dtype).reshape(-1, self.dx)
        support_X, batch_indices, _ = _support_union_with_batch(torch.empty(0, self.dx, device=self.device, dtype=self.dtype), X_batch)
        prior_mean = torch.zeros(support_X.shape[0], device=self.device, dtype=self.dtype)
        prior_cov = self._stabilize_cov(self._kernel(support_X, support_X), diag_floor=1e-8)
        return self._posterior_from_prior(support_X, prior_mean, prior_cov, batch_indices, y_batch)

    def _build_predictive_prior_on_expanded_support(self, X_batch: torch.Tensor):
        X_batch = X_batch.to(self.device, self.dtype).reshape(-1, self.dx)
        support_X, batch_indices, _ = _support_union_with_batch(self.support_X, X_batch)
        if self.support_X.numel() == 0:
            prior_mean = torch.zeros(support_X.shape[0], device=self.device, dtype=self.dtype)
            prior_cov = self._stabilize_cov(self._kernel(support_X, support_X), diag_floor=1e-8)
            return support_X, batch_indices, prior_mean, prior_cov
        old_n = int(self.support_X.shape[0])
        if int(support_X.shape[0]) == old_n:
            return support_X, batch_indices, self.support_mean.clone(), self.support_cov.clone()
        Z_new = support_X[old_n:]
        K_old_new = self._kernel(self.support_X, Z_new)
        solve = torch.cholesky_solve(K_old_new, self._support_chol)
        A = solve.transpose(0, 1)
        schur = self._kernel(Z_new, Z_new) - K_old_new.transpose(0, 1) @ solve
        mean_new = A @ self.support_mean
        cov_cross = self.support_cov @ A.transpose(0, 1)
        cov_new = schur + A @ self.support_cov @ A.transpose(0, 1)
        total_n = int(support_X.shape[0])
        prior_mean = torch.cat([self.support_mean, mean_new], dim=0)
        prior_cov = torch.zeros(total_n, total_n, device=self.device, dtype=self.dtype)
        prior_cov[:old_n, :old_n] = self.support_cov
        prior_cov[:old_n, old_n:] = cov_cross
        prior_cov[old_n:, :old_n] = cov_cross.transpose(0, 1)
        prior_cov[old_n:, old_n:] = cov_new
        return support_X, batch_indices, prior_mean, self._stabilize_cov(prior_cov)

    def append(self, X_new: torch.Tensor, y_new: torch.Tensor) -> "SharedExactOnlineBPCDeltaState":
        support_X, batch_indices, prior_mean, prior_cov = self._build_predictive_prior_on_expanded_support(X_new)
        self._replace_support(*self._posterior_from_prior(support_X, prior_mean, prior_cov, batch_indices, y_new))
        return self

    def predict(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        X = X.to(self.device, self.dtype).reshape(-1, self.dx)
        if self.support_X.numel() == 0:
            zeros = torch.zeros(X.shape[0], device=self.device, dtype=self.dtype)
            extra_noise = float(self.hyper_spec.noise) if self.add_kernel_noise_to_predict else 0.0
            return zeros, torch.full((X.shape[0],), float(self.hyper_spec.variance + extra_noise), device=self.device, dtype=self.dtype)
        K_xs = self._kernel(X, self.support_X)
        A = torch.cholesky_solve(K_xs.transpose(0, 1), self._support_chol).transpose(0, 1)
        mu = A @ self.support_mean
        cond_var = (float(self.hyper_spec.variance) - (K_xs * A).sum(dim=1)).clamp_min(0.0)
        latent_var = (A @ self.support_cov * A).sum(dim=1).clamp_min(0.0)
        extra_noise = float(self.hyper_spec.noise) if self.add_kernel_noise_to_predict else 0.0
        return mu, (cond_var + latent_var + extra_noise).clamp_min(1e-12)

    def copy(self) -> "SharedExactOnlineBPCDeltaState":
        out = object.__new__(SharedExactOnlineBPCDeltaState)
        out.hyper_spec = GPyTorchScaleRBFHyperSpec(lengthscale=self.hyper_spec.lengthscale.detach().clone(), variance=float(self.hyper_spec.variance), noise=float(self.hyper_spec.noise))
        out.lambda_delta = float(self.lambda_delta)
        out.obs_noise_var = None if self.obs_noise_var is None else float(self.obs_noise_var)
        out.add_kernel_noise_to_predict = bool(self.add_kernel_noise_to_predict)
        out.device = self.device
        out.dtype = self.dtype
        out.dx = self.dx
        out.support_X = self.support_X.clone()
        out.support_mean = self.support_mean.clone()
        out.support_cov = self.support_cov.clone()
        out._support_prior_cov = self._support_prior_cov.clone()
        out._support_chol = None if self._support_chol is None else self._support_chol.clone()
        return out



class SharedProxyDatasetOnlineBPCDeltaState(SharedExactOnlineBPCDeltaState):
    """
    Shared online-BPC discrepancy state updated through the proxy-dataset form.

    The previous posterior on the historical support is converted into a pseudo-dataset,
    the current residual batch is appended as a tempered Gaussian observation, and one exact
    GP posterior is recomputed on the combined support. When ``refit_max_iter > 0``, the GP
    hyperparameters are re-optimized on the combined pseudo-dataset marginal likelihood before
    the posterior update. This is a heuristic once the hyperparameters change, but it is the
    proxy-dataset variant requested for prediction diagnostics.
    """

    def __init__(self, X_init: torch.Tensor, y_init: torch.Tensor, hyper_spec: GPyTorchScaleRBFHyperSpec, *, lambda_delta: float = 1.0, refit_max_iter: int = 0, obs_noise_var: Optional[float] = None, add_kernel_noise_to_predict: bool = True):
        self.refit_max_iter = max(int(refit_max_iter), 0)
        self.last_proxy_diag: Dict[str, Any] = {}
        super().__init__(X_init=X_init, y_init=y_init, hyper_spec=hyper_spec, lambda_delta=lambda_delta, obs_noise_var=obs_noise_var, add_kernel_noise_to_predict=add_kernel_noise_to_predict)

    def _inverse_spd(self, mat: torch.Tensor, *, diag_floor: float = 1e-8) -> torch.Tensor:
        chol, repaired = self._safe_cholesky(mat, diag_floor=diag_floor)
        return self._stabilize_cov(torch.cholesky_inverse(chol), diag_floor=max(diag_floor, float(torch.diag(repaired).mean().detach().item()) * 1e-12))

    def _project_psd(self, mat: torch.Tensor, *, floor: float = 1e-10) -> torch.Tensor:
        sym = 0.5 * (mat + mat.transpose(0, 1))
        evals, evecs = torch.linalg.eigh(sym)
        evals = evals.clamp_min(float(floor))
        return (evecs * evals.unsqueeze(0)) @ evecs.transpose(0, 1)

    def _proxy_dataset_from_state(self) -> Tuple[torch.Tensor, torch.Tensor]:
        h = int(self.support_X.shape[0])
        if h == 0:
            return torch.empty(0, device=self.device, dtype=self.dtype), torch.empty(0, 0, device=self.device, dtype=self.dtype)
        K_h = self._kernel(self.support_X, self.support_X)
        K_inv = self._inverse_spd(K_h, diag_floor=1e-8)
        C_inv = self._inverse_spd(self.support_cov, diag_floor=max(self.hyper_spec.noise, 1e-8))
        pseudo_precision = self._project_psd(C_inv - K_inv, floor=1e-10)
        pseudo_cov = self._inverse_spd(pseudo_precision, diag_floor=1e-10)
        pseudo_y = pseudo_cov @ (C_inv @ self.support_mean)
        return pseudo_y, pseudo_cov

    def _candidate_spec(self, log_lengthscale: torch.Tensor, log_variance: torch.Tensor, log_noise: torch.Tensor) -> GPyTorchScaleRBFHyperSpec:
        return GPyTorchScaleRBFHyperSpec(
            lengthscale=log_lengthscale.exp(),
            variance=float(log_variance.exp().detach().item()),
            noise=float(log_noise.exp().detach().item()),
        )

    def _build_proxy_marginal(self, spec: GPyTorchScaleRBFHyperSpec, support_X: torch.Tensor, batch_indices: torch.Tensor, pseudo_y: torch.Tensor, pseudo_cov: torch.Tensor, y_batch: torch.Tensor):
        h = int(pseudo_y.shape[0])
        idx = batch_indices.to(self.device).long()
        y_batch = y_batch.to(self.device, self.dtype).reshape(-1)
        K_s = _scale_rbf_cov_from_spec(support_X, support_X, spec)
        z = torch.cat([pseudo_y.to(self.device, self.dtype).reshape(-1), y_batch], dim=0)
        K_hist = K_s[:h, :h]
        K_cross = K_s[:h, idx]
        K_batch = K_s[idx][:, idx]
        upper = torch.cat([K_hist + pseudo_cov, K_cross], dim=1)
        obs_noise = self.lambda_delta * self._effective_obs_noise(float(spec.noise))
        lower = torch.cat([K_cross.transpose(0, 1), K_batch + obs_noise * torch.eye(idx.shape[0], device=self.device, dtype=self.dtype)], dim=1)
        sigma_obs = self._stabilize_cov(torch.cat([upper, lower], dim=0), diag_floor=1e-8)
        K_SA = torch.cat([K_s[:, :h], K_s[:, idx]], dim=1)
        return z, sigma_obs, K_s, K_SA

    def _refit_hyper_from_proxy(self, support_X: torch.Tensor, batch_indices: torch.Tensor, pseudo_y: torch.Tensor, pseudo_cov: torch.Tensor, y_batch: torch.Tensor) -> GPyTorchScaleRBFHyperSpec:
        if self.refit_max_iter <= 0:
            return GPyTorchScaleRBFHyperSpec(lengthscale=self.hyper_spec.lengthscale.detach().clone(), variance=float(self.hyper_spec.variance), noise=float(self.hyper_spec.noise))
        log_ls = torch.log(self.hyper_spec.lengthscale.detach().clone().to(self.device, self.dtype)).requires_grad_(True)
        log_var = torch.log(torch.tensor(float(self.hyper_spec.variance), device=self.device, dtype=self.dtype).clamp_min(1e-8)).requires_grad_(True)
        log_noise = torch.log(torch.tensor(float(self.hyper_spec.noise), device=self.device, dtype=self.dtype).clamp_min(1e-8)).requires_grad_(True)
        opt = torch.optim.Adam([log_ls, log_var, log_noise], lr=0.05)
        best = None
        best_loss = float('inf')
        for _ in range(self.refit_max_iter):
            opt.zero_grad()
            spec = GPyTorchScaleRBFHyperSpec(lengthscale=log_ls.exp(), variance=float(log_var.exp().detach().item()), noise=float(log_noise.exp().detach().item()))
            z, sigma_obs, _, _ = self._build_proxy_marginal(spec, support_X, batch_indices, pseudo_y, pseudo_cov, y_batch)
            chol, _ = self._safe_cholesky(sigma_obs, diag_floor=max(float(spec.noise), 1e-8))
            alpha = torch.cholesky_solve(z[:, None], chol).squeeze(-1)
            loss = 0.5 * (z * alpha).sum() + torch.log(torch.diag(chol)).sum() + 0.5 * z.shape[0] * math.log(2.0 * math.pi)
            loss.backward()
            opt.step()
            cur = float(loss.detach().item())
            if cur < best_loss:
                best_loss = cur
                best = GPyTorchScaleRBFHyperSpec(lengthscale=log_ls.exp().detach().clone(), variance=float(log_var.exp().detach().item()), noise=float(log_noise.exp().detach().item()))
        return best if best is not None else GPyTorchScaleRBFHyperSpec(lengthscale=self.hyper_spec.lengthscale.detach().clone(), variance=float(self.hyper_spec.variance), noise=float(self.hyper_spec.noise))

    def append(self, X_new: torch.Tensor, y_new: torch.Tensor) -> "SharedProxyDatasetOnlineBPCDeltaState":
        X_new = X_new.to(self.device, self.dtype).reshape(-1, self.dx)
        y_new = y_new.to(self.device, self.dtype).reshape(-1)
        if self.support_X.numel() == 0:
            if self.refit_max_iter > 0 and X_new.shape[0] >= 2:
                spec = fit_scale_rbf_gpytorch_hyper(X_new, y_new, noise=max(float(self.hyper_spec.noise), 1e-8), lengthscale_init=self.hyper_spec.lengthscale.detach().clone(), variance_init=float(self.hyper_spec.variance), max_iter=self.refit_max_iter)
                self.hyper_spec = GPyTorchScaleRBFHyperSpec(lengthscale=spec.lengthscale.detach().clone(), variance=float(spec.variance), noise=float(spec.noise))
            self._replace_support(*self._posterior_from_empty_prior(X_new, y_new))
            return self
        pseudo_y, pseudo_cov = self._proxy_dataset_from_state()
        history_support_X = self.support_X.clone()
        self.last_proxy_diag = {
            "history_support_X": history_support_X,
            "history_pseudo_y": pseudo_y.detach().clone(),
            "history_pseudo_cov_diag": torch.diag(pseudo_cov).detach().clone(),
            "batch_X": X_new.detach().clone(),
            "batch_y": y_new.detach().clone(),
        }
        support_X, batch_indices, _ = _support_union_with_batch(self.support_X, X_new)
        spec = self._refit_hyper_from_proxy(support_X, batch_indices, pseudo_y, pseudo_cov, y_new)
        self.hyper_spec = GPyTorchScaleRBFHyperSpec(lengthscale=spec.lengthscale.detach().clone(), variance=float(spec.variance), noise=float(spec.noise))
        z, sigma_obs, K_s, K_SA = self._build_proxy_marginal(self.hyper_spec, support_X, batch_indices, pseudo_y, pseudo_cov, y_new)
        chol, _ = self._safe_cholesky(sigma_obs, diag_floor=max(float(self.hyper_spec.noise), 1e-8))
        alpha = torch.cholesky_solve(z[:, None], chol).squeeze(-1)
        mean = K_SA @ alpha
        solve_cross = torch.cholesky_solve(K_SA.transpose(0, 1), chol)
        cov = self._stabilize_cov(K_s - K_SA @ solve_cross)
        self._replace_support(support_X, mean, cov)
        return self

    def copy(self) -> "SharedProxyDatasetOnlineBPCDeltaState":
        out = object.__new__(SharedProxyDatasetOnlineBPCDeltaState)
        out.refit_max_iter = int(self.refit_max_iter)
        out.hyper_spec = GPyTorchScaleRBFHyperSpec(lengthscale=self.hyper_spec.lengthscale.detach().clone(), variance=float(self.hyper_spec.variance), noise=float(self.hyper_spec.noise))
        out.lambda_delta = float(self.lambda_delta)
        out.obs_noise_var = None if self.obs_noise_var is None else float(self.obs_noise_var)
        out.add_kernel_noise_to_predict = bool(self.add_kernel_noise_to_predict)
        out.device = self.device
        out.dtype = self.dtype
        out.dx = self.dx
        out.support_X = self.support_X.clone()
        out.support_mean = self.support_mean.clone()
        out.support_cov = self.support_cov.clone()
        out._support_prior_cov = self._support_prior_cov.clone()
        out._support_chol = None if self._support_chol is None else self._support_chol.clone()
        out.last_proxy_diag = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in self.last_proxy_diag.items()}
        return out


class SharedStableMeanProxyOnlineBPCDeltaState(SharedProxyDatasetOnlineBPCDeltaState):
    """
    Mean-stable proxy-dataset online-BPC discrepancy state.

    Instead of reconstructing a pseudo-target through the unstable inversion

        y_tilde = (C_H^{-1} - K_H^{-1})^{-1} C_H^{-1} m_H,

    this variant keeps the historical posterior mean itself as the proxy target
    and retains only diagonal historical uncertainty. The result is no longer an
    exact pseudo-dataset representation of the previous posterior, but it is
    numerically stable and preserves the quantity we care about most for the
    next update: the historical discrepancy mean.
    """

    def _proxy_dataset_from_state(self) -> Tuple[torch.Tensor, torch.Tensor]:
        h = int(self.support_X.shape[0])
        if h == 0:
            return (
                torch.empty(0, device=self.device, dtype=self.dtype),
                torch.empty(0, 0, device=self.device, dtype=self.dtype),
            )
        noise_floor = max(self.lambda_delta * max(float(self.hyper_spec.noise), 1e-8), 1e-8)
        hist_var = torch.diag(self.support_cov).clamp_min(noise_floor) + noise_floor
        pseudo_y = self.support_mean.clone()
        pseudo_cov = torch.diag(hist_var.to(self.device, self.dtype))
        return pseudo_y, pseudo_cov

    def copy(self) -> "SharedStableMeanProxyOnlineBPCDeltaState":
        out = object.__new__(SharedStableMeanProxyOnlineBPCDeltaState)
        out.refit_max_iter = int(self.refit_max_iter)
        out.hyper_spec = GPyTorchScaleRBFHyperSpec(lengthscale=self.hyper_spec.lengthscale.detach().clone(), variance=float(self.hyper_spec.variance), noise=float(self.hyper_spec.noise))
        out.lambda_delta = float(self.lambda_delta)
        out.obs_noise_var = None if self.obs_noise_var is None else float(self.obs_noise_var)
        out.add_kernel_noise_to_predict = bool(self.add_kernel_noise_to_predict)
        out.device = self.device
        out.dtype = self.dtype
        out.dx = self.dx
        out.support_X = self.support_X.clone()
        out.support_mean = self.support_mean.clone()
        out.support_cov = self.support_cov.clone()
        out._support_prior_cov = self._support_prior_cov.clone()
        out._support_chol = None if self._support_chol is None else self._support_chol.clone()
        out.last_proxy_diag = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in self.last_proxy_diag.items()}
        return out


class SharedFixedSupportOnlineBPCDeltaState:
    """
    Shared fixed-support online-BPC discrepancy state.

    The latent state is the inducing-value vector u = delta(Z) on a fixed support Z.
    Online updates are exact Gaussian recursions in this projected model:

        r_t | u ~ N(A_t u, R_t + Sigma_r),

    where A_t and R_t come from the GP conditional on Z.
    """

    def __init__(
        self,
        X_init: torch.Tensor,
        y_init: torch.Tensor,
        hyper_spec: GPyTorchScaleRBFHyperSpec,
        *,
        num_support: int = 20,
        support_points: Optional[torch.Tensor] = None,
        lambda_delta: float = 1.0,
        obs_noise_var: Optional[float] = None,
        add_kernel_noise_to_predict: bool = True,
    ):
        if X_init.numel() == 0 or y_init.numel() == 0:
            raise ValueError("SharedFixedSupportOnlineBPCDeltaState requires non-empty initialization data")
        X_init = X_init.reshape(-1, X_init.shape[1])
        y_init = y_init.reshape(-1)
        self.hyper_spec = GPyTorchScaleRBFHyperSpec(
            lengthscale=hyper_spec.lengthscale.detach().clone(),
            variance=max(float(hyper_spec.variance), 1e-8),
            noise=max(float(hyper_spec.noise), 1e-8),
        )
        self.lambda_delta = max(float(lambda_delta), 1e-6)
        self.obs_noise_var = None if obs_noise_var is None else max(float(obs_noise_var), 1e-8)
        self.add_kernel_noise_to_predict = bool(add_kernel_noise_to_predict)
        self.device = X_init.device
        self.dtype = X_init.dtype
        self.dx = int(X_init.shape[1])
        self.num_support = max(int(num_support), 1)
        self.Z = (
            support_points.to(self.device, self.dtype).reshape(-1, self.dx).clone()
            if support_points is not None
            else _select_sparse_inducing_points(X_init, self.num_support)
        )
        self.num_support = int(self.Z.shape[0])
        self._eye_z = torch.eye(self.num_support, device=self.device, dtype=self.dtype)
        self._build_kernel_cache()
        self.mean_u = torch.zeros(self.num_support, device=self.device, dtype=self.dtype)
        self.cov_u = self.Kzz.clone()
        self.refresh_from_history(X_init, y_init)

    def _build_kernel_cache(self) -> None:
        Kzz = _scale_rbf_cov_from_spec(self.Z, self.Z, self.hyper_spec)
        Kzz = 0.5 * (Kzz + Kzz.transpose(0, 1)) + 1e-6 * self._eye_z
        self.Kzz = Kzz
        self._chol_zz = torch.linalg.cholesky(Kzz)
        self._Kzz_inv = torch.cholesky_inverse(self._chol_zz)

    def _stabilize_cov(self, cov: torch.Tensor, *, floor: float = 1e-8) -> torch.Tensor:
        cov = 0.5 * (cov + cov.transpose(0, 1))
        if cov.numel() == 0:
            return cov
        return cov + max(float(floor), 1e-8) * 1e-6 * torch.eye(cov.shape[0], device=self.device, dtype=self.dtype)

    def _effective_obs_noise(self) -> float:
        if self.obs_noise_var is not None:
            return max(float(self.obs_noise_var), 1e-8)
        return max(float(self.hyper_spec.noise), 1e-8)

    def _mapping(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        X = X.to(self.device, self.dtype).reshape(-1, self.dx)
        Kxz = _scale_rbf_cov_from_spec(X, self.Z, self.hyper_spec)
        A = torch.cholesky_solve(Kxz.transpose(0, 1), self._chol_zz).transpose(0, 1)
        Kxx = _scale_rbf_cov_from_spec(X, X, self.hyper_spec)
        cond_cov = self._stabilize_cov(Kxx - Kxz @ A.transpose(0, 1), floor=1e-8)
        cond_var = cond_cov.diag().clamp_min(0.0)
        return A, cond_cov, cond_var

    def _posterior_update(self, mean_u: torch.Tensor, cov_u: torch.Tensor, X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        X = X.to(self.device, self.dtype).reshape(-1, self.dx)
        y = y.to(self.device, self.dtype).reshape(-1)
        A, cond_cov, _ = self._mapping(X)
        obs_cov = self._stabilize_cov(
            cond_cov + self._effective_obs_noise() * torch.eye(X.shape[0], device=self.device, dtype=self.dtype),
            floor=1e-8,
        )
        chol_obs = torch.linalg.cholesky(obs_cov)
        obs_solve_A = torch.cholesky_solve(A, chol_obs)
        obs_solve_y = torch.cholesky_solve(y[:, None], chol_obs).squeeze(-1)
        cov_u = self._stabilize_cov(cov_u, floor=1e-8)
        chol_prior = torch.linalg.cholesky(cov_u)
        prior_precision = torch.cholesky_inverse(chol_prior)
        precision = prior_precision + (1.0 / self.lambda_delta) * (A.transpose(0, 1) @ obs_solve_A)
        precision = self._stabilize_cov(precision, floor=1e-8)
        rhs = prior_precision @ mean_u + (1.0 / self.lambda_delta) * (A.transpose(0, 1) @ obs_solve_y)
        chol_post = torch.linalg.cholesky(precision)
        mean_post = torch.cholesky_solve(rhs[:, None], chol_post).squeeze(-1)
        cov_post = self._stabilize_cov(torch.cholesky_inverse(chol_post), floor=1e-8)
        return mean_post, cov_post

    def refresh_from_history(self, X_hist: torch.Tensor, y_hist: torch.Tensor) -> None:
        X_hist = X_hist.to(self.device, self.dtype).reshape(-1, self.dx)
        y_hist = y_hist.to(self.device, self.dtype).reshape(-1)
        self.mean_u, self.cov_u = self._posterior_update(
            torch.zeros(self.num_support, device=self.device, dtype=self.dtype),
            self.Kzz.clone(),
            X_hist,
            y_hist,
        )

    def append(self, X_new: torch.Tensor, y_new: torch.Tensor) -> "SharedFixedSupportOnlineBPCDeltaState":
        self.mean_u, self.cov_u = self._posterior_update(self.mean_u, self.cov_u, X_new, y_new)
        return self

    def predict(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        X = X.to(self.device, self.dtype).reshape(-1, self.dx)
        A, _, cond_var = self._mapping(X)
        mu = A @ self.mean_u
        latent_var = (A @ self.cov_u * A).sum(dim=1).clamp_min(0.0)
        extra_noise = float(self.hyper_spec.noise) if self.add_kernel_noise_to_predict else 0.0
        var = (cond_var + latent_var + extra_noise).clamp_min(1e-12)
        return mu, var

    def copy(self) -> "SharedFixedSupportOnlineBPCDeltaState":
        out = object.__new__(SharedFixedSupportOnlineBPCDeltaState)
        out.hyper_spec = GPyTorchScaleRBFHyperSpec(
            lengthscale=self.hyper_spec.lengthscale.detach().clone(),
            variance=float(self.hyper_spec.variance),
            noise=float(self.hyper_spec.noise),
        )
        out.lambda_delta = float(self.lambda_delta)
        out.obs_noise_var = None if self.obs_noise_var is None else float(self.obs_noise_var)
        out.add_kernel_noise_to_predict = bool(self.add_kernel_noise_to_predict)
        out.device = self.device
        out.dtype = self.dtype
        out.dx = self.dx
        out.num_support = int(self.num_support)
        out.Z = self.Z.clone()
        out._eye_z = self._eye_z.clone()
        out.Kzz = self.Kzz.clone()
        out._chol_zz = self._chol_zz.clone()
        out._Kzz_inv = self._Kzz_inv.clone()
        out.mean_u = self.mean_u.clone()
        out.cov_u = self.cov_u.clone()
        return out


class SharedMCInducingDeltaState:
    """
    Shared expert-level MC-inducing discrepancy posterior.

    The discrepancy is represented through inducing values u = delta(Z) and a weighted
    empirical posterior q(u) = sum_m alpha_m delta_{u_m}. Prediction uses moment matching,
    while BOCPD/PF scoring can use the exact discrepancy-particle mixture through
    ``loglik_for_particles``.
    """

    def __init__(
        self,
        X_init: torch.Tensor,
        y_init: torch.Tensor,
        hyper_spec: GPyTorchScaleRBFHyperSpec,
        *,
        num_inducing: int = 16,
        num_particles: int = 8,
        resample_ess_ratio: float = 0.5,
        include_conditional_var: bool = True,
        inducing_points: Optional[torch.Tensor] = None,
    ):
        if X_init.numel() == 0 or y_init.numel() == 0:
            raise ValueError("SharedMCInducingDeltaState requires non-empty initialization data")
        X_init = X_init.reshape(-1, X_init.shape[1])
        y_init = y_init.reshape(-1)
        self.hyper_spec = GPyTorchScaleRBFHyperSpec(
            lengthscale=hyper_spec.lengthscale.detach().clone(),
            variance=max(float(hyper_spec.variance), 1e-8),
            noise=max(float(hyper_spec.noise), 1e-8),
        )
        self.device = X_init.device
        self.dtype = X_init.dtype
        self.dx = int(X_init.shape[1])
        self.num_inducing = max(int(num_inducing), 1)
        self.num_particles = max(int(num_particles), 1)
        self.resample_ess_ratio = min(max(float(resample_ess_ratio), 1e-6), 1.0)
        self.include_conditional_var = bool(include_conditional_var)
        self.Z = (
            inducing_points.to(self.device, self.dtype).reshape(-1, self.dx).clone()
            if inducing_points is not None
            else _select_sparse_inducing_points(X_init, self.num_inducing)
        )
        self.num_inducing = int(self.Z.shape[0])
        self._eye_z = torch.eye(self.num_inducing, device=self.device, dtype=self.dtype)
        self._build_kernel_cache()
        self.u_particles = torch.empty(self.num_particles, self.num_inducing, device=self.device, dtype=self.dtype)
        self.weights = torch.full((self.num_particles,), 1.0 / self.num_particles, device=self.device, dtype=self.dtype)
        self.last_ess = float(self.num_particles)
        self.num_resamples = 0
        self.num_refreshes = 0
        self.refresh_from_history(X_init, y_init)

    def _build_kernel_cache(self) -> None:
        Kzz = _scale_rbf_cov_from_spec(self.Z, self.Z, self.hyper_spec)
        Kzz = Kzz + 1e-6 * self._eye_z
        self.Kzz = Kzz
        self._chol_zz = torch.linalg.cholesky(Kzz)
        self._Kzz_inv = torch.cholesky_inverse(self._chol_zz)

    def _mapping(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        X = X.to(self.device, self.dtype).reshape(-1, self.dx)
        Kxz = _scale_rbf_cov_from_spec(X, self.Z, self.hyper_spec)
        A = torch.cholesky_solve(Kxz.transpose(0, 1), self._chol_zz).transpose(0, 1)
        if self.include_conditional_var:
            cond_var = self.hyper_spec.variance - (Kxz * A).sum(dim=1)
            cond_var = cond_var.clamp_min(0.0)
        else:
            cond_var = torch.zeros(X.shape[0], device=self.device, dtype=self.dtype)
        obs_var = (cond_var + self.hyper_spec.noise).clamp_min(1e-12)
        return A, obs_var

    def _posterior_from_history(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        A, obs_var = self._mapping(X)
        inv_obs = 1.0 / obs_var.clamp_min(1e-12)
        precision = self._Kzz_inv + A.transpose(0, 1) @ (A * inv_obs[:, None])
        precision = 0.5 * (precision + precision.transpose(0, 1)) + 1e-8 * self._eye_z
        chol = torch.linalg.cholesky(precision)
        rhs = A.transpose(0, 1) @ (y.reshape(-1) * inv_obs)
        mean = torch.cholesky_solve(rhs[:, None], chol).squeeze(-1)
        cov = torch.cholesky_inverse(chol)
        cov = 0.5 * (cov + cov.transpose(0, 1)) + 1e-8 * self._eye_z
        return mean, cov

    def refresh_from_history(self, X_hist: torch.Tensor, y_hist: torch.Tensor) -> None:
        X_hist = X_hist.to(self.device, self.dtype).reshape(-1, self.dx)
        y_hist = y_hist.to(self.device, self.dtype).reshape(-1)
        mean, cov = self._posterior_from_history(X_hist, y_hist)
        chol = torch.linalg.cholesky(cov)
        eps = torch.randn(self.num_particles, self.num_inducing, device=self.device, dtype=self.dtype)
        self.u_particles = mean.view(1, -1) + eps @ chol.transpose(0, 1)
        self.weights = torch.full((self.num_particles,), 1.0 / self.num_particles, device=self.device, dtype=self.dtype)
        self.last_ess = float(self.num_particles)
        self.num_refreshes += 1

    def ess(self) -> float:
        return float(1.0 / self.weights.clamp_min(1e-30).pow(2).sum().item())

    def resample(self) -> None:
        idx = torch.multinomial(self.weights, self.num_particles, replacement=True)
        self.u_particles = self.u_particles[idx].clone()
        self.weights = torch.full((self.num_particles,), 1.0 / self.num_particles, device=self.device, dtype=self.dtype)
        self.last_ess = float(self.num_particles)
        self.num_resamples += 1

    def update_weights_from_residual(self, X_batch: torch.Tensor, resid_batch: torch.Tensor) -> None:
        A, obs_var = self._mapping(X_batch)
        resid_batch = resid_batch.to(self.device, self.dtype).reshape(-1)
        mu_particles = A @ self.u_particles.transpose(0, 1)
        loglik = normal_logpdf(resid_batch[:, None], mu_particles, obs_var[:, None]).sum(dim=0)
        logw = torch.log(self.weights.clamp_min(1e-30)) + loglik
        logw = logw - torch.logsumexp(logw, dim=0)
        self.weights = torch.exp(logw)
        self.last_ess = self.ess()
        if self.last_ess < self.resample_ess_ratio * self.num_particles:
            self.resample()

    def predict(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        A, obs_var = self._mapping(X)
        mu_particles = A @ self.u_particles.transpose(0, 1)
        w = self.weights.view(1, -1)
        mu = (w * mu_particles).sum(dim=1)
        second = (w * (mu_particles.pow(2) + obs_var[:, None])).sum(dim=1)
        var = (second - mu.pow(2)).clamp_min(1e-12)
        return mu, var

    def loglik_for_particles(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        theta_particles: torch.Tensor,
        *,
        emulator=None,
        rho: float,
        sigma_eps: float,
        mu_eta: Optional[torch.Tensor] = None,
        var_eta: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        del theta_particles, emulator
        if mu_eta is None or var_eta is None:
            return None
        if mu_eta.dim() != 2 or var_eta.dim() != 2:
            return None
        A, obs_var = self._mapping(x)
        mu_delta_particles = A @ self.u_particles.transpose(0, 1)  # [b,M]
        total_mu = rho * mu_eta[:, :, None] + mu_delta_particles[:, None, :]
        total_var = ((rho ** 2) * var_eta[:, :, None] + obs_var[:, None, None] + (sigma_eps ** 2)).clamp_min(1e-12)
        diff = y.reshape(-1, 1, 1) - total_mu
        loglik_bnm = -0.5 * (torch.log(2.0 * math.pi * total_var) + diff.pow(2) / total_var)
        loglik_nm = loglik_bnm.sum(dim=0)
        return torch.logsumexp(loglik_nm + torch.log(self.weights.clamp_min(1e-30))[None, :], dim=1)

    def copy(self) -> "SharedMCInducingDeltaState":
        out = object.__new__(SharedMCInducingDeltaState)
        out.hyper_spec = GPyTorchScaleRBFHyperSpec(
            lengthscale=self.hyper_spec.lengthscale.detach().clone(),
            variance=float(self.hyper_spec.variance),
            noise=float(self.hyper_spec.noise),
        )
        out.device = self.device
        out.dtype = self.dtype
        out.dx = self.dx
        out.num_inducing = self.num_inducing
        out.num_particles = self.num_particles
        out.resample_ess_ratio = self.resample_ess_ratio
        out.include_conditional_var = self.include_conditional_var
        out.Z = self.Z.clone()
        out._eye_z = self._eye_z.clone()
        out.Kzz = self.Kzz.clone()
        out._chol_zz = self._chol_zz.clone()
        out._Kzz_inv = self._Kzz_inv.clone()
        out.u_particles = self.u_particles.clone()
        out.weights = self.weights.clone()
        out.last_ess = float(self.last_ess)
        out.num_resamples = int(self.num_resamples)
        out.num_refreshes = int(self.num_refreshes)
        return out


# ------------------------- Minimal test in __main__ -------------------------
if __name__ == "__main__":
    # Minimal self-test comparing exact_full vs exact_rank1, and optional SVGP if gpytorch exists.
    torch.manual_seed(0)
    device, dtype = "cpu", torch.float64

    # Synthetic 2D function for delta: delta(x) = sin(2*pi*x0) + 0.3*x1
    def f_delta(X):
        return torch.sin(2.0 * math.pi * X[:, 0]) + 0.3 * X[:, 1]

    # Training set
    n0, dx = 8, 2
    X0 = torch.rand(n0, dx, dtype=dtype, device=device)
    y0 = f_delta(X0) + 0.01 * torch.randn(n0, dtype=dtype, device=device)

    # A query batch
    Xq = torch.rand(5, dx, dtype=dtype, device=device)

    # Kernel with vector lengthscale
    from .kernels import RBFKernel
    k = RBFKernel(lengthscale=torch.tensor([0.4, 0.7], dtype=dtype), variance=1.2)

    print("=== exact_full ===")
    gp_full = OnlineGPState(X=torch.empty(0, dx, dtype=dtype), y=torch.empty(0, dtype=dtype),
                            kernel=k, noise=1e-3, update_mode="exact_full", hyperparam_mode="fixed")
    gp_full.append_batch(X0, y0)
    mu_f, var_f = gp_full.predict(Xq)
    print("mu_f:", mu_f.detach().cpu().numpy())
    print("var_f:", var_f.detach().cpu().numpy())

    print("\n=== exact_rank1 (single appends) ===")
    gp_rank1 = OnlineGPState(X=torch.empty(0, dx, dtype=dtype), y=torch.empty(0, dtype=dtype),
                             kernel=k, noise=1e-3, update_mode="exact_rank1", hyperparam_mode="fixed")
    gp_rank1.append_batch(X0, y0)
    mu_r, var_r = gp_rank1.predict(Xq)
    print("mu_r:", mu_r.detach().cpu().numpy())
    print("var_r:", var_r.detach().cpu().numpy())

    # Sanity: results should be very close
    print("\nmax |mu_f - mu_r| =", (mu_f - mu_r).abs().max().item())
    print("max |var_f - var_r| =", (var_f - var_r).abs().max().item())

    # Test batch append additional points
    X1 = torch.rand(3, dx, dtype=dtype, device=device)
    y1 = f_delta(X1) + 0.01 * torch.randn(3, dtype=dtype, device=device)
    gp_rank1.append_batch(X1, y1)
    mu_r2, var_r2 = gp_rank1.predict(Xq)
    print("\nAfter batch append, mu_r2:", mu_r2.detach().cpu().numpy())

    # Optional: hyperparameter refit (ML-II)
    gp_rank1.hyperparam_mode = "fit"
    gp_rank1.refit_hyperparams(max_iter=50, lr=0.05, fit_noise=True)
    mu_r3, var_r3 = gp_rank1.predict(Xq)
    print("\nAfter refit, mu_r3:", mu_r3.detach().cpu().numpy())

    # Optional: SVGP (requires gpytorch)
    try:
        import gpytorch  # noqa
        print("\n=== SVGP (gpytorch) ===")
        Z = torch.rand(16, dx, dtype=dtype, device=device)  # inducing points
        svgp = SVGPState(X=torch.empty(0, dx, dtype=dtype), y=torch.empty(0, dtype=dtype),
                         inducing_points=Z, noise=1e-3, lengthscale=torch.tensor([0.4, 0.7], dtype=dtype), variance=1.2)
        svgp.append(X0, y0, steps=100, lr=0.05)
        mu_s, var_s = svgp.predict(Xq)
        print("mu_s:", mu_s.detach().cpu().numpy())
    except Exception as e:
        print("\nSVGP not available:", e)
