from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
import math
import torch

from .emulator import Emulator
from .utils import normal_logpdf


def _to_lengthscale_tensor(lengthscale, dx: int, *, device, dtype) -> torch.Tensor:
    ls = torch.as_tensor(lengthscale, device=device, dtype=dtype).reshape(-1)
    if ls.numel() == 1:
        ls = ls.repeat(dx)
    if ls.numel() != dx:
        raise ValueError(f"lengthscale dimension mismatch: expected {dx}, got {ls.numel()}")
    return ls.clamp_min(1e-8)


def _rbf_cov(
    x1: torch.Tensor,
    x2: torch.Tensor,
    *,
    lengthscale: torch.Tensor,
    variance: float,
) -> torch.Tensor:
    x1s = x1 / lengthscale
    x2s = x2 / lengthscale
    dist2 = (x1s[:, None, :] - x2s[None, :, :]).pow(2).sum(dim=-1)
    return float(variance) * torch.exp(-0.5 * dist2)


@dataclass
class KernelHyperSpec:
    lengthscale: torch.Tensor
    variance: float
    noise: float


class ParticleSpecificGPDeltaState:
    def __init__(
        self,
        X_hist: torch.Tensor,
        Y_hist: torch.Tensor,
        theta_particles: torch.Tensor,
        emulator: Emulator,
        rho: float,
        hyper_specs: Sequence[KernelHyperSpec],
    ):
        self.X_hist = X_hist
        self.Y_hist = Y_hist.reshape(-1)
        self.theta_particles = theta_particles
        self.emulator = emulator
        self.rho = float(rho)
        self.device = X_hist.device
        self.dtype = X_hist.dtype
        self.dx = int(X_hist.shape[1])
        self.hyper_specs = [
            KernelHyperSpec(
                lengthscale=_to_lengthscale_tensor(spec.lengthscale, self.dx, device=self.device, dtype=self.dtype),
                variance=float(spec.variance),
                noise=max(float(spec.noise), 1e-8),
            )
            for spec in hyper_specs
        ]
        self._kernel_caches = [self._build_kernel_cache(spec) for spec in self.hyper_specs]
        self._current_stats = self._build_particle_stats(self.theta_particles)

    def _build_kernel_cache(self, spec: KernelHyperSpec):
        K = _rbf_cov(self.X_hist, self.X_hist, lengthscale=spec.lengthscale, variance=spec.variance)
        K = K + (spec.noise + 1e-6) * torch.eye(K.shape[0], device=self.device, dtype=self.dtype)
        chol = torch.linalg.cholesky(K)
        logdet = 2.0 * torch.log(torch.diag(chol)).sum()
        return {"spec": spec, "chol": chol, "logdet": logdet}

    def _residual_matrix(self, theta_particles: torch.Tensor) -> torch.Tensor:
        mu_eta_hist, _ = self.emulator.predict(self.X_hist, theta_particles)
        if mu_eta_hist.dim() == 3:
            mu_eta_hist = mu_eta_hist.mean(dim=-1)
        return self.Y_hist[:, None] - self.rho * mu_eta_hist

    def _build_particle_stats(self, theta_particles: torch.Tensor):
        R = self._residual_matrix(theta_particles)
        alphas: List[torch.Tensor] = []
        log_evidences: List[torch.Tensor] = []
        n = int(self.X_hist.shape[0])
        const = -0.5 * n * math.log(2.0 * math.pi)
        for cache in self._kernel_caches:
            chol = cache["chol"]
            alpha = torch.cholesky_solve(R, chol)
            quad = (R * alpha).sum(dim=0)
            logev = const - 0.5 * quad - 0.5 * cache["logdet"]
            alphas.append(alpha)
            log_evidences.append(logev)
        if len(alphas) == 1:
            weights = torch.ones(1, theta_particles.shape[0], device=self.device, dtype=self.dtype)
        else:
            weights = torch.softmax(torch.stack(log_evidences, dim=0), dim=0)
        return {"alphas": alphas, "weights": weights}

    def _predict_with_stats(self, x: torch.Tensor, stats) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.to(self.device, self.dtype)
        mus = []
        vars_ = []
        N = int(stats["weights"].shape[1])
        for alpha, cache in zip(stats["alphas"], self._kernel_caches):
            spec = cache["spec"]
            K_qh = _rbf_cov(x, self.X_hist, lengthscale=spec.lengthscale, variance=spec.variance)
            mu_h = K_qh @ alpha
            solve = torch.cholesky_solve(K_qh.transpose(0, 1), cache["chol"])
            base_var = spec.variance - (K_qh * solve.transpose(0, 1)).sum(dim=1)
            base_var = (base_var + spec.noise).clamp_min(1e-12)
            var_h = base_var[:, None].expand(-1, N)
            mus.append(mu_h)
            vars_.append(var_h)
        if len(mus) == 1:
            return mus[0], vars_[0]
        weights = stats["weights"]
        mix_mu = torch.zeros_like(mus[0])
        mix_second = torch.zeros_like(mus[0])
        for h, (mu_h, var_h) in enumerate(zip(mus, vars_)):
            w = weights[h][None, :]
            mix_mu = mix_mu + w * mu_h
            mix_second = mix_second + w * (var_h + mu_h.pow(2))
        mix_var = (mix_second - mix_mu.pow(2)).clamp_min(1e-12)
        return mix_mu, mix_var

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._predict_with_stats(x, self._current_stats)

    def predict_for_particles(
        self,
        x: torch.Tensor,
        theta_particles: torch.Tensor,
        *,
        emulator: Optional[Emulator] = None,
        rho: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del rho
        if emulator is not None:
            self.emulator = emulator
        stats = self._build_particle_stats(theta_particles.to(self.device, self.dtype))
        return self._predict_with_stats(x, stats)


class ParticleSpecificBasisDeltaState:
    def __init__(
        self,
        X_hist: torch.Tensor,
        Y_hist: torch.Tensor,
        theta_particles: torch.Tensor,
        emulator: Emulator,
        rho: float,
        *,
        basis_kind: str = "rbf",
        num_features: int = 8,
        lengthscale: float = 0.25,
        ridge: float = 1e-2,
        noise: float = 1e-3,
    ):
        self.X_hist = X_hist
        self.Y_hist = Y_hist.reshape(-1)
        self.theta_particles = theta_particles
        self.emulator = emulator
        self.rho = float(rho)
        self.device = X_hist.device
        self.dtype = X_hist.dtype
        self.basis_kind = str(basis_kind).lower()
        self.num_features = max(1, int(num_features))
        self.lengthscale = max(float(lengthscale), 1e-8)
        self.ridge = max(float(ridge), 1e-8)
        self.noise = max(float(noise), 1e-8)
        self.centers = self._select_centers()
        self.Phi_hist = self._basis(self.X_hist)
        self._chol_A = self._build_precision_chol(self.Phi_hist)
        self._current_beta = self._solve_beta(self.theta_particles)

    def _select_centers(self) -> Optional[torch.Tensor]:
        if self.basis_kind != "rbf":
            return None
        n = int(self.X_hist.shape[0])
        m = min(self.num_features, n)
        idx = torch.linspace(0, n - 1, m, device=self.device).round().long().unique(sorted=True)
        return self.X_hist[idx]

    def _basis(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device, self.dtype)
        if self.basis_kind == "linear":
            return torch.cat([torch.ones(x.shape[0], 1, device=self.device, dtype=self.dtype), x], dim=1)
        if self.centers is None:
            raise ValueError("rbf basis requires centers")
        dist2 = (x[:, None, :] - self.centers[None, :, :]).pow(2).sum(dim=-1)
        feats = torch.exp(-0.5 * dist2 / (self.lengthscale ** 2))
        return torch.cat([torch.ones(x.shape[0], 1, device=self.device, dtype=self.dtype), feats], dim=1)

    def _build_precision_chol(self, Phi: torch.Tensor) -> torch.Tensor:
        p = int(Phi.shape[1])
        A = (Phi.transpose(0, 1) @ Phi) / self.noise + self.ridge * torch.eye(p, device=self.device, dtype=self.dtype)
        return torch.linalg.cholesky(A)

    def _residual_matrix(self, theta_particles: torch.Tensor) -> torch.Tensor:
        mu_eta_hist, _ = self.emulator.predict(self.X_hist, theta_particles)
        if mu_eta_hist.dim() == 3:
            mu_eta_hist = mu_eta_hist.mean(dim=-1)
        return self.Y_hist[:, None] - self.rho * mu_eta_hist

    def _solve_beta(self, theta_particles: torch.Tensor) -> torch.Tensor:
        R = self._residual_matrix(theta_particles)
        rhs = (self.Phi_hist.transpose(0, 1) @ R) / self.noise
        return torch.cholesky_solve(rhs, self._chol_A)

    def _predict_with_beta(self, x: torch.Tensor, beta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        Phi_x = self._basis(x)
        mu = Phi_x @ beta
        solve = torch.cholesky_solve(Phi_x.transpose(0, 1), self._chol_A)
        base_var = (Phi_x * solve.transpose(0, 1)).sum(dim=1)
        var = (base_var + self.noise)[:, None].expand(-1, beta.shape[1]).clamp_min(1e-12)
        return mu, var

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._predict_with_beta(x, self._current_beta)

    def predict_for_particles(
        self,
        x: torch.Tensor,
        theta_particles: torch.Tensor,
        *,
        emulator: Optional[Emulator] = None,
        rho: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del rho
        if emulator is not None:
            self.emulator = emulator
        beta = self._solve_beta(theta_particles.to(self.device, self.dtype))
        return self._predict_with_beta(x, beta)

class ParticleSpecificOnlineGPDeltaState:
    """
    Frozen-residual, lineage-bound online particle discrepancy state.

    Residual histories are attached to current particle lineages. After resampling, children
    inherit their parent's residual history and then append their own new residuals.
    """

    def __init__(
        self,
        theta_particles: torch.Tensor,
        emulator: Emulator,
        rho: float,
        *,
        hyper_specs: Optional[Sequence[KernelHyperSpec]] = None,
        min_points: int = 3,
    ):
        self.theta_particles = theta_particles.detach().clone()
        self.emulator = emulator
        self.rho = float(rho)
        self.device = theta_particles.device
        self.dtype = theta_particles.dtype
        self.dx = None
        self.min_points = max(int(min_points), 1)
        self.X_hist = torch.empty(0, 0, device=self.device, dtype=self.dtype)
        self.R_hist = torch.empty(0, int(theta_particles.shape[0]), device=self.device, dtype=self.dtype)
        self.hyper_specs: Optional[List[KernelHyperSpec]] = None
        self._kernel_caches = []
        self._current_stats = None
        if hyper_specs:
            self.set_hyper_specs(hyper_specs)

    def copy(self) -> 'ParticleSpecificOnlineGPDeltaState':
        out = ParticleSpecificOnlineGPDeltaState(
            theta_particles=self.theta_particles.clone(),
            emulator=self.emulator,
            rho=self.rho,
            hyper_specs=self.hyper_specs,
            min_points=self.min_points,
        )
        out.X_hist = self.X_hist.clone()
        out.R_hist = self.R_hist.clone()
        if out.hyper_specs is not None and out.X_hist.numel() > 0:
            out._rebuild_kernel_caches()
            out._rebuild_current_stats()
        return out

    def set_hyper_specs(self, hyper_specs: Sequence[KernelHyperSpec]) -> None:
        if self.dx is None:
            if self.X_hist.numel() == 0:
                raise ValueError('Cannot set particle online hyperparameters before any X history exists')
            self.dx = int(self.X_hist.shape[1])
        self.hyper_specs = [
            KernelHyperSpec(
                lengthscale=_to_lengthscale_tensor(spec.lengthscale, self.dx, device=self.device, dtype=self.dtype),
                variance=float(spec.variance),
                noise=max(float(spec.noise), 1e-8),
            )
            for spec in hyper_specs
        ]
        self._rebuild_kernel_caches()
        self._rebuild_current_stats()

    def _rebuild_kernel_caches(self) -> None:
        if self.hyper_specs is None or self.X_hist.numel() == 0:
            self._kernel_caches = []
            return
        self._kernel_caches = []
        for spec in self.hyper_specs:
            K = _rbf_cov(self.X_hist, self.X_hist, lengthscale=spec.lengthscale, variance=spec.variance)
            K = K + (spec.noise + 1e-6) * torch.eye(K.shape[0], device=self.device, dtype=self.dtype)
            chol = torch.linalg.cholesky(K)
            logdet = 2.0 * torch.log(torch.diag(chol)).sum()
            self._kernel_caches.append({'spec': spec, 'chol': chol, 'logdet': logdet})

    def _rebuild_current_stats(self) -> None:
        if self.hyper_specs is None or self.X_hist.shape[0] < self.min_points or self.R_hist.numel() == 0:
            self._current_stats = None
            return
        self._current_stats = self._build_stats_from_residual_history()

    def _build_stats_from_residual_history(self):
        alphas: List[torch.Tensor] = []
        log_evidences: List[torch.Tensor] = []
        n = int(self.X_hist.shape[0])
        const = -0.5 * n * math.log(2.0 * math.pi)
        for cache in self._kernel_caches:
            chol = cache['chol']
            alpha = torch.cholesky_solve(self.R_hist, chol)
            quad = (self.R_hist * alpha).sum(dim=0)
            logev = const - 0.5 * quad - 0.5 * cache['logdet']
            alphas.append(alpha)
            log_evidences.append(logev)
        if len(alphas) == 1:
            weights = torch.ones(1, self.R_hist.shape[1], device=self.device, dtype=self.dtype)
        else:
            weights = torch.softmax(torch.stack(log_evidences, dim=0), dim=0)
        return {'alphas': alphas, 'weights': weights}

    def _predict_with_stats(self, x: torch.Tensor, stats) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.to(self.device, self.dtype)
        mus = []
        vars_ = []
        N = int(stats['weights'].shape[1])
        for alpha, cache in zip(stats['alphas'], self._kernel_caches):
            spec = cache['spec']
            K_qh = _rbf_cov(x, self.X_hist, lengthscale=spec.lengthscale, variance=spec.variance)
            mu_h = K_qh @ alpha
            solve = torch.cholesky_solve(K_qh.transpose(0, 1), cache['chol'])
            base_var = spec.variance - (K_qh * solve.transpose(0, 1)).sum(dim=1)
            base_var = (base_var + spec.noise).clamp_min(1e-12)
            var_h = base_var[:, None].expand(-1, N)
            mus.append(mu_h)
            vars_.append(var_h)
        if len(mus) == 1:
            return mus[0], vars_[0]
        weights = stats['weights']
        mix_mu = torch.zeros_like(mus[0])
        mix_second = torch.zeros_like(mus[0])
        for h, (mu_h, var_h) in enumerate(zip(mus, vars_)):
            w = weights[h][None, :]
            mix_mu = mix_mu + w * mu_h
            mix_second = mix_second + w * (var_h + mu_h.pow(2))
        mix_var = (mix_second - mix_mu.pow(2)).clamp_min(1e-12)
        return mix_mu, mix_var

    def append_batch(
        self,
        X_batch: torch.Tensor,
        Y_batch: torch.Tensor,
        theta_particles: torch.Tensor,
        *,
        ancestor_indices: Optional[torch.Tensor] = None,
        emulator: Optional[Emulator] = None,
        rho: Optional[float] = None,
    ) -> None:
        X_batch = X_batch.to(self.device, self.dtype)
        Y_batch = Y_batch.to(self.device, self.dtype).reshape(-1)
        theta_particles = theta_particles.to(self.device, self.dtype)
        if emulator is not None:
            self.emulator = emulator
        if rho is not None:
            self.rho = float(rho)
        if self.dx is None:
            self.dx = int(X_batch.shape[1])
        if ancestor_indices is not None and self.R_hist.numel() > 0:
            idx = ancestor_indices.to(self.device).long()
            self.R_hist = self.R_hist[:, idx]
        self.theta_particles = theta_particles.detach().clone()
        mu_eta_batch, _ = self.emulator.predict(X_batch, theta_particles)
        if mu_eta_batch.dim() == 3:
            mu_eta_batch = mu_eta_batch.mean(dim=-1)
        R_batch = Y_batch[:, None] - self.rho * mu_eta_batch
        if self.X_hist.numel() == 0:
            self.X_hist = X_batch.clone()
        else:
            self.X_hist = torch.cat([self.X_hist, X_batch], dim=0)
        if self.R_hist.numel() == 0:
            self.R_hist = R_batch.clone()
        else:
            self.R_hist = torch.cat([self.R_hist, R_batch], dim=0)
        if self.hyper_specs is not None:
            self._rebuild_kernel_caches()
        self._rebuild_current_stats()

    def truncate_recent(self, keep_n: int) -> None:
        keep_n = max(int(keep_n), 0)
        if keep_n <= 0 or self.X_hist.numel() == 0:
            self.X_hist = torch.empty(0, 0 if self.dx is None else self.dx, device=self.device, dtype=self.dtype)
            self.R_hist = torch.empty(0, self.theta_particles.shape[0], device=self.device, dtype=self.dtype)
            self._kernel_caches = []
            self._current_stats = None
            return
        if keep_n < self.X_hist.shape[0]:
            self.X_hist = self.X_hist[-keep_n:].clone()
            self.R_hist = self.R_hist[-keep_n:].clone()
        if self.hyper_specs is not None and self.X_hist.numel() > 0:
            self._rebuild_kernel_caches()
        self._rebuild_current_stats()

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._current_stats is None:
            b = x.shape[0]
            n = int(self.theta_particles.shape[0])
            zeros = torch.zeros(b, n, device=self.device, dtype=self.dtype)
            return zeros, zeros
        return self._predict_with_stats(x, self._current_stats)

    def predict_for_particles(
        self,
        x: torch.Tensor,
        theta_particles: torch.Tensor,
        *,
        emulator: Optional[Emulator] = None,
        rho: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del theta_particles
        if emulator is not None:
            self.emulator = emulator
        if rho is not None:
            self.rho = float(rho)
        return self.predict(x)



def _support_union_with_batch_rows(
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


class ParticleSpecificBatchSupportOnlineBPCDeltaState:
    """
    Lineage-bound exact online-BPC particle discrepancy state with one-batch support.

    Each particle lineage carries a Gaussian posterior over discrepancy values on the most
    recent batch inputs. The support inputs are shared across lineages, so only the support
    posterior means differ by particle; the support covariance remains shared. After PF
    resampling, children copy their parent's support posterior mean before assimilating the
    new residual batch.
    """

    def __init__(
        self,
        theta_particles: torch.Tensor,
        emulator: Emulator,
        rho: float,
        *,
        hyper_spec: Optional[KernelHyperSpec] = None,
        min_points: int = 3,
        lambda_delta: float = 1.0,
    ):
        self.theta_particles = theta_particles.detach().clone()
        self.emulator = emulator
        self.rho = float(rho)
        self.device = theta_particles.device
        self.dtype = theta_particles.dtype
        self.dx: Optional[int] = None
        self.min_points = max(int(min_points), 1)
        self.lambda_delta = max(float(lambda_delta), 1e-6)
        self.support_X = torch.empty(0, 0, device=self.device, dtype=self.dtype)
        self.mean_particles: Optional[torch.Tensor] = None
        self.cov: Optional[torch.Tensor] = None
        self._support_prior_cov = torch.empty(0, 0, device=self.device, dtype=self.dtype)
        self._support_chol = None
        self.hyper_spec: Optional[KernelHyperSpec] = None
        if hyper_spec is not None:
            self.set_hyper_spec(hyper_spec)

    def _stabilize_cov(self, cov: torch.Tensor, *, diag_floor: Optional[float] = None) -> torch.Tensor:
        cov = 0.5 * (cov + cov.transpose(0, 1))
        n = int(cov.shape[0])
        if n == 0:
            return cov
        floor = max(float(diag_floor if diag_floor is not None else (self.hyper_spec.noise if self.hyper_spec is not None else 1e-8)), 1e-8)
        eye = torch.eye(n, device=self.device, dtype=self.dtype)
        return cov + floor * 1e-6 * eye

    def _kernel(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        if self.hyper_spec is None:
            raise ValueError('hyper_spec must be set before using particle online BPC discrepancy')
        return _rbf_cov(
            x1.to(self.device, self.dtype).reshape(-1, self.dx),
            x2.to(self.device, self.dtype).reshape(-1, self.dx),
            lengthscale=self.hyper_spec.lengthscale,
            variance=self.hyper_spec.variance,
        )

    def _support_noise(self, n: int) -> torch.Tensor:
        if self.hyper_spec is None:
            raise ValueError('hyper_spec must be set before using particle online BPC discrepancy')
        return (self.lambda_delta * max(self.hyper_spec.noise, 1e-8)) * torch.eye(n, device=self.device, dtype=self.dtype)

    def _replace_support(self, X_support: torch.Tensor, mean_particles: torch.Tensor, cov: torch.Tensor) -> None:
        self.support_X = X_support.to(self.device, self.dtype).reshape(-1, self.dx).clone()
        self.mean_particles = mean_particles.to(self.device, self.dtype).clone()
        self.cov = self._stabilize_cov(cov.to(self.device, self.dtype)).clone()
        self._support_prior_cov = self._kernel(self.support_X, self.support_X)
        self._support_chol = torch.linalg.cholesky(self._stabilize_cov(self._support_prior_cov, diag_floor=1e-8))

    def _posterior_from_prior_batch(self, X_batch: torch.Tensor, R_batch: torch.Tensor):
        X_batch = X_batch.to(self.device, self.dtype).reshape(-1, self.dx)
        R_batch = R_batch.to(self.device, self.dtype).reshape(X_batch.shape[0], -1)
        K = self._kernel(X_batch, X_batch)
        S = self._stabilize_cov(K + self._support_noise(X_batch.shape[0]))
        chol_s = torch.linalg.cholesky(S)
        alpha = torch.cholesky_solve(R_batch, chol_s)
        mean_particles = K @ alpha
        solve_k = torch.cholesky_solve(K, chol_s)
        cov = K - K @ solve_k
        cov = self._stabilize_cov(cov)
        return X_batch, mean_particles, cov

    def _propagate_to_batch(self, X_batch: torch.Tensor):
        X_batch = X_batch.to(self.device, self.dtype).reshape(-1, self.dx)
        if self.support_X.numel() == 0 or self.mean_particles is None or self.cov is None:
            K = self._kernel(X_batch, X_batch)
            zeros = torch.zeros(X_batch.shape[0], self.theta_particles.shape[0], device=self.device, dtype=self.dtype)
            return zeros, self._stabilize_cov(K), K
        K_old_new = self._kernel(self.support_X, X_batch)
        solve = torch.cholesky_solve(K_old_new, self._support_chol)
        A = solve.transpose(0, 1)
        K_new = self._kernel(X_batch, X_batch)
        schur = K_new - K_old_new.transpose(0, 1) @ solve
        prior_mean_particles = A @ self.mean_particles
        prior_cov = schur + A @ self.cov @ A.transpose(0, 1)
        prior_cov = self._stabilize_cov(prior_cov)
        return prior_mean_particles, prior_cov, K_new

    def _posterior_from_propagated_batch(self, X_batch: torch.Tensor, R_batch: torch.Tensor):
        X_batch = X_batch.to(self.device, self.dtype).reshape(-1, self.dx)
        R_batch = R_batch.to(self.device, self.dtype).reshape(X_batch.shape[0], -1)
        prior_mean_particles, prior_cov, _ = self._propagate_to_batch(X_batch)
        S = self._stabilize_cov(prior_cov + self._support_noise(X_batch.shape[0]))
        chol_s = torch.linalg.cholesky(S)
        innovation = R_batch - prior_mean_particles
        solve_innov = torch.cholesky_solve(innovation, chol_s)
        mean_particles = prior_mean_particles + prior_cov @ solve_innov
        solve_prior = torch.cholesky_solve(prior_cov, chol_s)
        cov = prior_cov - prior_cov @ solve_prior
        cov = self._stabilize_cov(cov)
        return X_batch, mean_particles, cov

    def copy(self) -> 'ParticleSpecificBatchSupportOnlineBPCDeltaState':
        out = ParticleSpecificBatchSupportOnlineBPCDeltaState(
            theta_particles=self.theta_particles.clone(),
            emulator=self.emulator,
            rho=self.rho,
            hyper_spec=self.hyper_spec,
            min_points=self.min_points,
            lambda_delta=self.lambda_delta,
        )
        out.dx = self.dx
        out.support_X = self.support_X.clone()
        out.mean_particles = None if self.mean_particles is None else self.mean_particles.clone()
        out.cov = None if self.cov is None else self.cov.clone()
        out._support_prior_cov = self._support_prior_cov.clone()
        out._support_chol = None if self._support_chol is None else self._support_chol.clone()
        return out

    def set_hyper_spec(self, hyper_spec: KernelHyperSpec) -> None:
        if self.dx is None:
            self.dx = int(torch.as_tensor(hyper_spec.lengthscale, device=self.device, dtype=self.dtype).reshape(-1).numel())
        self.hyper_spec = KernelHyperSpec(
            lengthscale=_to_lengthscale_tensor(hyper_spec.lengthscale, self.dx, device=self.device, dtype=self.dtype),
            variance=max(float(hyper_spec.variance), 1e-8),
            noise=max(float(hyper_spec.noise), 1e-8),
        )
        if self.support_X.numel() > 0 and self.mean_particles is not None and self.cov is not None:
            self._replace_support(self.support_X, self.mean_particles, self.cov)

    def append_batch(
        self,
        X_batch: torch.Tensor,
        Y_batch: torch.Tensor,
        theta_particles: torch.Tensor,
        *,
        ancestor_indices: Optional[torch.Tensor] = None,
        emulator: Optional[Emulator] = None,
        rho: Optional[float] = None,
    ) -> None:
        X_batch = X_batch.to(self.device, self.dtype)
        Y_batch = Y_batch.to(self.device, self.dtype).reshape(-1)
        theta_particles = theta_particles.to(self.device, self.dtype)
        if emulator is not None:
            self.emulator = emulator
        if rho is not None:
            self.rho = float(rho)
        if self.dx is None:
            self.dx = int(X_batch.shape[1])
        if ancestor_indices is not None and self.mean_particles is not None:
            idx = ancestor_indices.to(self.device).long()
            self.mean_particles = self.mean_particles[:, idx]
        self.theta_particles = theta_particles.detach().clone()
        if self.hyper_spec is None:
            return
        mu_eta_batch, _ = self.emulator.predict(X_batch, theta_particles)
        if mu_eta_batch.dim() == 3:
            mu_eta_batch = mu_eta_batch.mean(dim=-1)
        R_batch = Y_batch[:, None] - self.rho * mu_eta_batch
        if self.support_X.numel() == 0 or self.mean_particles is None or self.cov is None:
            self._replace_support(*self._posterior_from_prior_batch(X_batch, R_batch))
            return
        self._replace_support(*self._posterior_from_propagated_batch(X_batch, R_batch))

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b = int(x.shape[0])
        n = int(self.theta_particles.shape[0])
        if self.hyper_spec is None or self.support_X.numel() == 0 or self.mean_particles is None or self.cov is None:
            zeros = torch.zeros(b, n, device=self.device, dtype=self.dtype)
            return zeros, zeros
        x = x.to(self.device, self.dtype).reshape(-1, self.dx)
        K_xs = self._kernel(x, self.support_X)
        solve = torch.cholesky_solve(K_xs.transpose(0, 1), self._support_chol)
        A = solve.transpose(0, 1)
        mu = A @ self.mean_particles
        cond_var = float(self.hyper_spec.variance) - (K_xs * A).sum(dim=1)
        cond_var = cond_var.clamp_min(0.0)
        latent_var = (A @ self.cov * A).sum(dim=1).clamp_min(0.0)
        var = (cond_var + latent_var + self.hyper_spec.noise)[:, None].expand(-1, n).clamp_min(1e-12)
        return mu, var

    def predict_for_particles(
        self,
        x: torch.Tensor,
        theta_particles: torch.Tensor,
        *,
        emulator: Optional[Emulator] = None,
        rho: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del theta_particles
        if emulator is not None:
            self.emulator = emulator
        if rho is not None:
            self.rho = float(rho)
        return self.predict(x)



class ParticleSpecificExactOnlineBPCDeltaState:
    """Lineage-bound exact online-BPC particle discrepancy state on an expanding support."""

    def __init__(self, theta_particles: torch.Tensor, emulator: Emulator, rho: float, *, hyper_spec: Optional[KernelHyperSpec] = None, min_points: int = 3, lambda_delta: float = 1.0):
        self.theta_particles = theta_particles.detach().clone()
        self.emulator = emulator
        self.rho = float(rho)
        self.device = theta_particles.device
        self.dtype = theta_particles.dtype
        self.dx: Optional[int] = None
        self.min_points = max(int(min_points), 1)
        self.lambda_delta = max(float(lambda_delta), 1e-6)
        self.support_X = torch.empty(0, 0, device=self.device, dtype=self.dtype)
        self.mean_particles: Optional[torch.Tensor] = None
        self.cov: Optional[torch.Tensor] = None
        self._support_prior_cov = torch.empty(0, 0, device=self.device, dtype=self.dtype)
        self._support_chol = None
        self.hyper_spec: Optional[KernelHyperSpec] = None
        if hyper_spec is not None:
            self.set_hyper_spec(hyper_spec)

    def _stabilize_cov(self, cov: torch.Tensor, *, diag_floor: Optional[float] = None) -> torch.Tensor:
        cov = 0.5 * (cov + cov.transpose(0, 1))
        n = int(cov.shape[0])
        if n == 0:
            return cov
        floor = max(float(diag_floor if diag_floor is not None else (self.hyper_spec.noise if self.hyper_spec is not None else 1e-8)), 1e-8)
        return cov + floor * 1e-6 * torch.eye(n, device=self.device, dtype=self.dtype)

    def _kernel(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        if self.hyper_spec is None:
            raise ValueError('hyper_spec must be set before using particle exact online BPC discrepancy')
        return _rbf_cov(x1.to(self.device, self.dtype).reshape(-1, self.dx), x2.to(self.device, self.dtype).reshape(-1, self.dx), lengthscale=self.hyper_spec.lengthscale, variance=self.hyper_spec.variance)

    def _support_noise(self, n: int) -> torch.Tensor:
        if self.hyper_spec is None:
            raise ValueError('hyper_spec must be set before using particle exact online BPC discrepancy')
        return (self.lambda_delta * max(self.hyper_spec.noise, 1e-8)) * torch.eye(n, device=self.device, dtype=self.dtype)

    def _replace_support(self, X_support: torch.Tensor, mean_particles: torch.Tensor, cov: torch.Tensor) -> None:
        self.support_X = X_support.to(self.device, self.dtype).reshape(-1, self.dx).clone()
        self.mean_particles = mean_particles.to(self.device, self.dtype).clone()
        self.cov = self._stabilize_cov(cov.to(self.device, self.dtype)).clone()
        self._support_prior_cov = self._kernel(self.support_X, self.support_X)
        self._support_chol = torch.linalg.cholesky(self._stabilize_cov(self._support_prior_cov, diag_floor=1e-8))

    def _posterior_from_prior(self, X_support: torch.Tensor, prior_mean_particles: torch.Tensor, prior_cov: torch.Tensor, batch_indices: torch.Tensor, R_batch: torch.Tensor):
        idx = batch_indices.to(self.device).long()
        prior_mean_particles = prior_mean_particles.to(self.device, self.dtype)
        prior_cov = self._stabilize_cov(prior_cov.to(self.device, self.dtype))
        R_batch = R_batch.to(self.device, self.dtype).reshape(idx.shape[0], -1)
        prior_obs_mean = prior_mean_particles[idx, :]
        cross = prior_cov[:, idx]
        S = self._stabilize_cov(prior_cov[idx][:, idx] + self._support_noise(idx.shape[0]))
        chol_s = torch.linalg.cholesky(S)
        innovation = R_batch - prior_obs_mean
        mean_particles = prior_mean_particles + cross @ torch.cholesky_solve(innovation, chol_s)
        cov = prior_cov - cross @ torch.cholesky_solve(cross.transpose(0, 1), chol_s)
        return X_support, mean_particles, self._stabilize_cov(cov)

    def _posterior_from_empty_prior(self, X_batch: torch.Tensor, R_batch: torch.Tensor):
        X_batch = X_batch.to(self.device, self.dtype).reshape(-1, self.dx)
        support_X, batch_indices, _ = _support_union_with_batch_rows(torch.empty(0, self.dx, device=self.device, dtype=self.dtype), X_batch)
        prior_mean_particles = torch.zeros(support_X.shape[0], R_batch.shape[1], device=self.device, dtype=self.dtype)
        prior_cov = self._stabilize_cov(self._kernel(support_X, support_X), diag_floor=1e-8)
        return self._posterior_from_prior(support_X, prior_mean_particles, prior_cov, batch_indices, R_batch)

    def _build_predictive_prior_on_expanded_support(self, X_batch: torch.Tensor):
        X_batch = X_batch.to(self.device, self.dtype).reshape(-1, self.dx)
        support_X, batch_indices, _ = _support_union_with_batch_rows(self.support_X, X_batch)
        if self.support_X.numel() == 0 or self.mean_particles is None or self.cov is None:
            prior_mean_particles = torch.zeros(support_X.shape[0], self.theta_particles.shape[0], device=self.device, dtype=self.dtype)
            prior_cov = self._stabilize_cov(self._kernel(support_X, support_X), diag_floor=1e-8)
            return support_X, batch_indices, prior_mean_particles, prior_cov
        old_n = int(self.support_X.shape[0])
        if int(support_X.shape[0]) == old_n:
            return support_X, batch_indices, self.mean_particles.clone(), self.cov.clone()
        Z_new = support_X[old_n:]
        K_old_new = self._kernel(self.support_X, Z_new)
        solve = torch.cholesky_solve(K_old_new, self._support_chol)
        A = solve.transpose(0, 1)
        schur = self._kernel(Z_new, Z_new) - K_old_new.transpose(0, 1) @ solve
        mean_new = A @ self.mean_particles
        cov_cross = self.cov @ A.transpose(0, 1)
        cov_new = schur + A @ self.cov @ A.transpose(0, 1)
        total_n = int(support_X.shape[0])
        prior_mean_particles = torch.cat([self.mean_particles, mean_new], dim=0)
        prior_cov = torch.zeros(total_n, total_n, device=self.device, dtype=self.dtype)
        prior_cov[:old_n, :old_n] = self.cov
        prior_cov[:old_n, old_n:] = cov_cross
        prior_cov[old_n:, :old_n] = cov_cross.transpose(0, 1)
        prior_cov[old_n:, old_n:] = cov_new
        return support_X, batch_indices, prior_mean_particles, self._stabilize_cov(prior_cov)

    def copy(self) -> 'ParticleSpecificExactOnlineBPCDeltaState':
        out = ParticleSpecificExactOnlineBPCDeltaState(theta_particles=self.theta_particles.clone(), emulator=self.emulator, rho=self.rho, hyper_spec=self.hyper_spec, min_points=self.min_points, lambda_delta=self.lambda_delta)
        out.dx = self.dx
        out.support_X = self.support_X.clone()
        out.mean_particles = None if self.mean_particles is None else self.mean_particles.clone()
        out.cov = None if self.cov is None else self.cov.clone()
        out._support_prior_cov = self._support_prior_cov.clone()
        out._support_chol = None if self._support_chol is None else self._support_chol.clone()
        return out

    def set_hyper_spec(self, hyper_spec: KernelHyperSpec) -> None:
        if self.dx is None:
            self.dx = int(torch.as_tensor(hyper_spec.lengthscale, device=self.device, dtype=self.dtype).reshape(-1).numel())
        self.hyper_spec = KernelHyperSpec(lengthscale=_to_lengthscale_tensor(hyper_spec.lengthscale, self.dx, device=self.device, dtype=self.dtype), variance=max(float(hyper_spec.variance), 1e-8), noise=max(float(hyper_spec.noise), 1e-8))
        if self.support_X.numel() > 0 and self.mean_particles is not None and self.cov is not None:
            self._replace_support(self.support_X, self.mean_particles, self.cov)

    def append_batch(self, X_batch: torch.Tensor, Y_batch: torch.Tensor, theta_particles: torch.Tensor, *, ancestor_indices: Optional[torch.Tensor] = None, emulator: Optional[Emulator] = None, rho: Optional[float] = None) -> None:
        X_batch = X_batch.to(self.device, self.dtype)
        Y_batch = Y_batch.to(self.device, self.dtype).reshape(-1)
        theta_particles = theta_particles.to(self.device, self.dtype)
        if emulator is not None:
            self.emulator = emulator
        if rho is not None:
            self.rho = float(rho)
        if self.dx is None:
            self.dx = int(X_batch.shape[1])
        if ancestor_indices is not None and self.mean_particles is not None:
            self.mean_particles = self.mean_particles[:, ancestor_indices.to(self.device).long()]
        self.theta_particles = theta_particles.detach().clone()
        if self.hyper_spec is None:
            return
        mu_eta_batch, _ = self.emulator.predict(X_batch, theta_particles)
        if mu_eta_batch.dim() == 3:
            mu_eta_batch = mu_eta_batch.mean(dim=-1)
        R_batch = Y_batch[:, None] - self.rho * mu_eta_batch
        if self.support_X.numel() == 0 or self.mean_particles is None or self.cov is None:
            self._replace_support(*self._posterior_from_empty_prior(X_batch, R_batch))
            return
        support_X, batch_indices, prior_mean_particles, prior_cov = self._build_predictive_prior_on_expanded_support(X_batch)
        self._replace_support(*self._posterior_from_prior(support_X, prior_mean_particles, prior_cov, batch_indices, R_batch))

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b = int(x.shape[0])
        n = int(self.theta_particles.shape[0])
        if self.hyper_spec is None or self.support_X.numel() == 0 or self.mean_particles is None or self.cov is None:
            zeros = torch.zeros(b, n, device=self.device, dtype=self.dtype)
            return zeros, zeros
        x = x.to(self.device, self.dtype).reshape(-1, self.dx)
        K_xs = self._kernel(x, self.support_X)
        A = torch.cholesky_solve(K_xs.transpose(0, 1), self._support_chol).transpose(0, 1)
        mu = A @ self.mean_particles
        cond_var = (float(self.hyper_spec.variance) - (K_xs * A).sum(dim=1)).clamp_min(0.0)
        latent_var = (A @ self.cov * A).sum(dim=1).clamp_min(0.0)
        var = (cond_var + latent_var + self.hyper_spec.noise)[:, None].expand(-1, n).clamp_min(1e-12)
        return mu, var

    def predict_for_particles(self, x: torch.Tensor, theta_particles: torch.Tensor, *, emulator: Optional[Emulator] = None, rho: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        del theta_particles
        if emulator is not None:
            self.emulator = emulator
        if rho is not None:
            self.rho = float(rho)
        return self.predict(x)


class ParticleSpecificFixedSupportOnlineBPCDeltaState:
    """Lineage-bound fixed-support exact online-BPC particle discrepancy state."""

    def __init__(self, theta_particles: torch.Tensor, emulator: Emulator, rho: float, *, hyper_spec: Optional[KernelHyperSpec] = None, min_points: int = 3, num_support: int = 20, lambda_delta: float = 1.0, obs_noise_var: Optional[float] = None, add_kernel_noise_to_predict: bool = True):
        self.theta_particles = theta_particles.detach().clone()
        self.emulator = emulator
        self.rho = float(rho)
        self.device = theta_particles.device
        self.dtype = theta_particles.dtype
        self.dx: Optional[int] = None
        self.min_points = max(int(min_points), 1)
        self.num_support = max(int(num_support), 1)
        self.lambda_delta = max(float(lambda_delta), 1e-6)
        self.obs_noise_var = None if obs_noise_var is None else max(float(obs_noise_var), 1e-8)
        self.add_kernel_noise_to_predict = bool(add_kernel_noise_to_predict)
        self.hyper_spec: Optional[KernelHyperSpec] = None
        self.Z = torch.empty(0, 0, device=self.device, dtype=self.dtype)
        self._eye_z = torch.empty(0, 0, device=self.device, dtype=self.dtype)
        self.Kzz = torch.empty(0, 0, device=self.device, dtype=self.dtype)
        self._chol_zz = None
        self._Kzz_inv = None
        self.mean_particles: Optional[torch.Tensor] = None
        self.cov_u: Optional[torch.Tensor] = None
        if hyper_spec is not None:
            self.set_hyper_spec(hyper_spec)

    def _stabilize_cov(self, cov: torch.Tensor, *, diag_floor: Optional[float] = None) -> torch.Tensor:
        cov = 0.5 * (cov + cov.transpose(0, 1))
        n = int(cov.shape[0])
        if n == 0:
            return cov
        floor = max(float(diag_floor if diag_floor is not None else (self.hyper_spec.noise if self.hyper_spec is not None else 1e-8)), 1e-8)
        return cov + floor * 1e-6 * torch.eye(n, device=self.device, dtype=self.dtype)

    def _effective_obs_noise(self) -> float:
        if self.obs_noise_var is not None:
            return max(float(self.obs_noise_var), 1e-8)
        if self.hyper_spec is None:
            return 1e-8
        return max(float(self.hyper_spec.noise), 1e-8)

    def _select_support(self, X_hist: torch.Tensor) -> torch.Tensor:
        n = int(X_hist.shape[0])
        m = max(1, min(self.num_support, n))
        if m == n:
            idx = torch.arange(n, device=self.device)
        else:
            idx = torch.linspace(0, n - 1, m, device=self.device).round().long().unique(sorted=True)
            if idx.numel() < m:
                idx = torch.arange(m, device=self.device)
        return X_hist[idx].clone()

    def set_hyper_spec(self, hyper_spec: KernelHyperSpec) -> None:
        if self.dx is None:
            self.dx = int(torch.as_tensor(hyper_spec.lengthscale, device=self.device, dtype=self.dtype).reshape(-1).numel())
        self.hyper_spec = KernelHyperSpec(
            lengthscale=_to_lengthscale_tensor(hyper_spec.lengthscale, self.dx, device=self.device, dtype=self.dtype),
            variance=max(float(hyper_spec.variance), 1e-8),
            noise=max(float(hyper_spec.noise), 1e-8),
        )
        if self.Z.numel() > 0:
            self._build_kernel_cache()

    def _build_kernel_cache(self) -> None:
        if self.hyper_spec is None or self.Z.numel() == 0:
            return
        J = int(self.Z.shape[0])
        self._eye_z = torch.eye(J, device=self.device, dtype=self.dtype)
        self.Kzz = _rbf_cov(self.Z, self.Z, lengthscale=self.hyper_spec.lengthscale, variance=self.hyper_spec.variance)
        self.Kzz = self._stabilize_cov(self.Kzz + 1e-6 * self._eye_z, diag_floor=1e-8)
        self._chol_zz = torch.linalg.cholesky(self.Kzz)
        self._Kzz_inv = torch.cholesky_inverse(self._chol_zz)

    def _mapping(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.hyper_spec is None or self._chol_zz is None:
            raise ValueError('particle fixed-support online BPC state is not initialized')
        X = X.to(self.device, self.dtype).reshape(-1, self.dx)
        Kxz = _rbf_cov(X, self.Z, lengthscale=self.hyper_spec.lengthscale, variance=self.hyper_spec.variance)
        A = torch.cholesky_solve(Kxz.transpose(0, 1), self._chol_zz).transpose(0, 1)
        Kxx = _rbf_cov(X, X, lengthscale=self.hyper_spec.lengthscale, variance=self.hyper_spec.variance)
        cond_cov = self._stabilize_cov(Kxx - Kxz @ A.transpose(0, 1), diag_floor=1e-8)
        cond_var = cond_cov.diag().clamp_min(0.0)
        return A, cond_cov, cond_var

    def _posterior_update(self, mean_particles: torch.Tensor, cov_u: torch.Tensor, X: torch.Tensor, R_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        X = X.to(self.device, self.dtype).reshape(-1, self.dx)
        R_batch = R_batch.to(self.device, self.dtype).reshape(X.shape[0], -1)
        A, cond_cov, _ = self._mapping(X)
        obs_cov = self._stabilize_cov(cond_cov + self._effective_obs_noise() * torch.eye(X.shape[0], device=self.device, dtype=self.dtype), diag_floor=1e-8)
        chol_obs = torch.linalg.cholesky(obs_cov)
        obs_solve_A = torch.cholesky_solve(A, chol_obs)
        obs_solve_R = torch.cholesky_solve(R_batch, chol_obs)
        cov_u = self._stabilize_cov(cov_u, diag_floor=1e-8)
        chol_prior = torch.linalg.cholesky(cov_u)
        prior_precision = torch.cholesky_inverse(chol_prior)
        precision = prior_precision + (1.0 / self.lambda_delta) * (A.transpose(0, 1) @ obs_solve_A)
        precision = self._stabilize_cov(precision, diag_floor=1e-8)
        rhs = prior_precision @ mean_particles + (1.0 / self.lambda_delta) * (A.transpose(0, 1) @ obs_solve_R)
        chol_post = torch.linalg.cholesky(precision)
        mean_post = torch.cholesky_solve(rhs, chol_post)
        cov_post = self._stabilize_cov(torch.cholesky_inverse(chol_post), diag_floor=1e-8)
        return mean_post, cov_post

    def copy(self) -> 'ParticleSpecificFixedSupportOnlineBPCDeltaState':
        out = ParticleSpecificFixedSupportOnlineBPCDeltaState(theta_particles=self.theta_particles.clone(), emulator=self.emulator, rho=self.rho, hyper_spec=self.hyper_spec, min_points=self.min_points, num_support=self.num_support, lambda_delta=self.lambda_delta, obs_noise_var=self.obs_noise_var, add_kernel_noise_to_predict=self.add_kernel_noise_to_predict)
        out.dx = self.dx
        out.Z = self.Z.clone()
        out._eye_z = self._eye_z.clone()
        out.Kzz = self.Kzz.clone()
        out._chol_zz = None if self._chol_zz is None else self._chol_zz.clone()
        out._Kzz_inv = None if self._Kzz_inv is None else self._Kzz_inv.clone()
        out.mean_particles = None if self.mean_particles is None else self.mean_particles.clone()
        out.cov_u = None if self.cov_u is None else self.cov_u.clone()
        return out

    def append_batch(self, X_batch: torch.Tensor, Y_batch: torch.Tensor, theta_particles: torch.Tensor, *, ancestor_indices: Optional[torch.Tensor] = None, emulator: Optional[Emulator] = None, rho: Optional[float] = None) -> None:
        X_batch = X_batch.to(self.device, self.dtype)
        Y_batch = Y_batch.to(self.device, self.dtype).reshape(-1)
        theta_particles = theta_particles.to(self.device, self.dtype)
        if emulator is not None:
            self.emulator = emulator
        if rho is not None:
            self.rho = float(rho)
        if self.dx is None:
            self.dx = int(X_batch.shape[1])
        if self.Z.numel() == 0:
            self.Z = self._select_support(X_batch)
            self._build_kernel_cache()
        if ancestor_indices is not None and self.mean_particles is not None:
            self.mean_particles = self.mean_particles[:, ancestor_indices.to(self.device).long()]
        self.theta_particles = theta_particles.detach().clone()
        if self.hyper_spec is None or self._chol_zz is None:
            return
        mu_eta_batch, _ = self.emulator.predict(X_batch, theta_particles)
        if mu_eta_batch.dim() == 3:
            mu_eta_batch = mu_eta_batch.mean(dim=-1)
        R_batch = Y_batch[:, None] - self.rho * mu_eta_batch
        if self.mean_particles is None or self.cov_u is None:
            self.mean_particles, self.cov_u = self._posterior_update(
                torch.zeros(int(self.Z.shape[0]), theta_particles.shape[0], device=self.device, dtype=self.dtype),
                self.Kzz.clone(),
                X_batch,
                R_batch,
            )
            return
        self.mean_particles, self.cov_u = self._posterior_update(self.mean_particles, self.cov_u, X_batch, R_batch)

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b = int(x.shape[0])
        n = int(self.theta_particles.shape[0])
        if self.hyper_spec is None or self.Z.numel() == 0 or self.mean_particles is None or self.cov_u is None or self._chol_zz is None:
            zeros = torch.zeros(b, n, device=self.device, dtype=self.dtype)
            return zeros, zeros
        x = x.to(self.device, self.dtype).reshape(-1, self.dx)
        A, _, cond_var = self._mapping(x)
        mu = A @ self.mean_particles
        latent_var = (A @ self.cov_u * A).sum(dim=1).clamp_min(0.0)
        extra_noise = float(self.hyper_spec.noise) if self.add_kernel_noise_to_predict else 0.0
        var = (cond_var + latent_var + extra_noise)[:, None].expand(-1, n).clamp_min(1e-12)
        return mu, var

    def predict_for_particles(self, x: torch.Tensor, theta_particles: torch.Tensor, *, emulator: Optional[Emulator] = None, rho: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        del theta_particles
        if emulator is not None:
            self.emulator = emulator
        if rho is not None:
            self.rho = float(rho)
        return self.predict(x)


class ParticleSpecificMCInducingDeltaState:
    """
    Particle-specific MC-inducing discrepancy posterior with lineage-bound updates.

    Each theta particle maintains its own weighted empirical posterior over inducing
    values. Resampling copies the parent's discrepancy posterior into each child and
    the children then update with their own residuals.
    """

    def __init__(
        self,
        theta_particles: torch.Tensor,
        emulator: Emulator,
        rho: float,
        *,
        hyper_spec: Optional[KernelHyperSpec] = None,
        min_points: int = 3,
        num_inducing: int = 8,
        num_mc_particles: int = 4,
        resample_ess_ratio: float = 0.5,
        include_conditional_var: bool = True,
    ):
        self.theta_particles = theta_particles.detach().clone()
        self.emulator = emulator
        self.rho = float(rho)
        self.device = theta_particles.device
        self.dtype = theta_particles.dtype
        self.dx: Optional[int] = None
        self.min_points = max(int(min_points), 1)
        self.num_inducing = max(int(num_inducing), 1)
        self.num_mc_particles = max(int(num_mc_particles), 1)
        self.resample_ess_ratio = min(max(float(resample_ess_ratio), 1e-6), 1.0)
        self.include_conditional_var = bool(include_conditional_var)
        self.hyper_spec: Optional[KernelHyperSpec] = None
        self.Z: Optional[torch.Tensor] = None
        self._eye_z: Optional[torch.Tensor] = None
        self.Kzz: Optional[torch.Tensor] = None
        self._chol_zz: Optional[torch.Tensor] = None
        self._Kzz_inv: Optional[torch.Tensor] = None
        self.u_particles = torch.empty(int(theta_particles.shape[0]), self.num_mc_particles, self.num_inducing, device=self.device, dtype=self.dtype)
        self.weights = torch.full((int(theta_particles.shape[0]), self.num_mc_particles), 1.0 / self.num_mc_particles, device=self.device, dtype=self.dtype)
        self.last_ess = torch.full((int(theta_particles.shape[0]),), float(self.num_mc_particles), device=self.device, dtype=self.dtype)
        self.num_resamples = 0
        self.num_refreshes = 0
        if hyper_spec is not None:
            self.set_hyper_spec(hyper_spec)

    def _select_inducing(self, X_hist: torch.Tensor) -> torch.Tensor:
        n = int(X_hist.shape[0])
        m = max(1, min(self.num_inducing, n))
        if m == n:
            idx = torch.arange(n, device=self.device)
        else:
            idx = torch.linspace(0, n - 1, m, device=self.device).round().long().unique(sorted=True)
            if idx.numel() < m:
                idx = torch.arange(m, device=self.device)
        return X_hist[idx].clone()

    def set_hyper_spec(self, hyper_spec: KernelHyperSpec) -> None:
        if self.dx is None:
            self.dx = int(torch.as_tensor(hyper_spec.lengthscale, device=self.device, dtype=self.dtype).reshape(-1).numel())
        self.hyper_spec = KernelHyperSpec(
            lengthscale=_to_lengthscale_tensor(hyper_spec.lengthscale, self.dx, device=self.device, dtype=self.dtype),
            variance=max(float(hyper_spec.variance), 1e-8),
            noise=max(float(hyper_spec.noise), 1e-8),
        )

    def _build_kernel_cache(self) -> None:
        if self.Z is None or self.hyper_spec is None:
            return
        J = int(self.Z.shape[0])
        self._eye_z = torch.eye(J, device=self.device, dtype=self.dtype)
        self.Kzz = _rbf_cov(self.Z, self.Z, lengthscale=self.hyper_spec.lengthscale, variance=self.hyper_spec.variance)
        self.Kzz = self.Kzz + 1e-6 * self._eye_z
        self._chol_zz = torch.linalg.cholesky(self.Kzz)
        self._Kzz_inv = torch.cholesky_inverse(self._chol_zz)

    def _mapping(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.Z is None or self.hyper_spec is None or self._chol_zz is None:
            raise ValueError('particle MC-inducing state is not initialized')
        X = X.to(self.device, self.dtype).reshape(-1, self.dx)
        Kxz = _rbf_cov(X, self.Z, lengthscale=self.hyper_spec.lengthscale, variance=self.hyper_spec.variance)
        A = torch.cholesky_solve(Kxz.transpose(0, 1), self._chol_zz).transpose(0, 1)
        if self.include_conditional_var:
            cond_var = self.hyper_spec.variance - (Kxz * A).sum(dim=1)
            cond_var = cond_var.clamp_min(0.0)
        else:
            cond_var = torch.zeros(X.shape[0], device=self.device, dtype=self.dtype)
        obs_var = (cond_var + self.hyper_spec.noise).clamp_min(1e-12)
        return A, obs_var

    def refresh_from_history(
        self,
        X_hist: torch.Tensor,
        Y_hist: torch.Tensor,
        theta_particles: torch.Tensor,
        *,
        emulator: Optional[Emulator] = None,
        rho: Optional[float] = None,
    ) -> None:
        X_hist = X_hist.to(self.device, self.dtype).reshape(-1, X_hist.shape[1])
        Y_hist = Y_hist.to(self.device, self.dtype).reshape(-1)
        theta_particles = theta_particles.to(self.device, self.dtype)
        if emulator is not None:
            self.emulator = emulator
        if rho is not None:
            self.rho = float(rho)
        self.theta_particles = theta_particles.detach().clone()
        self.dx = int(X_hist.shape[1])
        if self.hyper_spec is None:
            raise ValueError('hyper_spec must be set before refreshing particle MC-inducing state')
        self.Z = self._select_inducing(X_hist)
        self.num_inducing = int(self.Z.shape[0])
        self._build_kernel_cache()
        A, obs_var = self._mapping(X_hist)
        inv_obs = 1.0 / obs_var.clamp_min(1e-12)
        precision = self._Kzz_inv + A.transpose(0, 1) @ (A * inv_obs[:, None])
        precision = 0.5 * (precision + precision.transpose(0, 1)) + 1e-8 * self._eye_z
        chol = torch.linalg.cholesky(precision)
        mu_eta_hist, _ = self.emulator.predict(X_hist, theta_particles)
        if mu_eta_hist.dim() == 3:
            mu_eta_hist = mu_eta_hist.mean(dim=-1)
        R_hist = Y_hist[:, None] - self.rho * mu_eta_hist
        rhs = A.transpose(0, 1) @ (R_hist * inv_obs[:, None])
        mean_particles = torch.cholesky_solve(rhs, chol)  # [J,N]
        post_cov = torch.cholesky_inverse(chol)
        post_cov = 0.5 * (post_cov + post_cov.transpose(0, 1)) + 1e-8 * self._eye_z
        chol_cov = torch.linalg.cholesky(post_cov)
        eps = torch.randn(int(theta_particles.shape[0]), self.num_mc_particles, self.num_inducing, device=self.device, dtype=self.dtype)
        means = mean_particles.transpose(0, 1)
        self.u_particles = means[:, None, :] + torch.einsum('nmj,kj->nmk', eps, chol_cov.transpose(0, 1))
        self.weights = torch.full((int(theta_particles.shape[0]), self.num_mc_particles), 1.0 / self.num_mc_particles, device=self.device, dtype=self.dtype)
        self.last_ess = torch.full((int(theta_particles.shape[0]),), float(self.num_mc_particles), device=self.device, dtype=self.dtype)
        self.num_refreshes += 1

    def _resample_rows(self, mask: torch.Tensor) -> None:
        rows = torch.nonzero(mask, as_tuple=False).reshape(-1)
        if rows.numel() == 0:
            return
        for idx in rows.tolist():
            picks = torch.multinomial(self.weights[idx], self.num_mc_particles, replacement=True)
            self.u_particles[idx] = self.u_particles[idx, picks].clone()
            self.weights[idx] = 1.0 / self.num_mc_particles
        self.num_resamples += int(rows.numel())

    def append_batch(
        self,
        X_batch: torch.Tensor,
        Y_batch: torch.Tensor,
        theta_particles: torch.Tensor,
        *,
        ancestor_indices: Optional[torch.Tensor] = None,
        emulator: Optional[Emulator] = None,
        rho: Optional[float] = None,
    ) -> None:
        if self.hyper_spec is None or self.Z is None:
            return
        X_batch = X_batch.to(self.device, self.dtype)
        Y_batch = Y_batch.to(self.device, self.dtype).reshape(-1)
        theta_particles = theta_particles.to(self.device, self.dtype)
        if emulator is not None:
            self.emulator = emulator
        if rho is not None:
            self.rho = float(rho)
        if ancestor_indices is not None:
            idx = ancestor_indices.to(self.device).long()
            self.u_particles = self.u_particles[idx].clone()
            self.weights = self.weights[idx].clone()
            self.last_ess = self.last_ess[idx].clone()
        self.theta_particles = theta_particles.detach().clone()
        mu_eta_batch, _ = self.emulator.predict(X_batch, theta_particles)
        if mu_eta_batch.dim() == 3:
            mu_eta_batch = mu_eta_batch.mean(dim=-1)
        R_batch = Y_batch[:, None] - self.rho * mu_eta_batch
        A, obs_var = self._mapping(X_batch)
        mu_particles = torch.einsum('bj,nmj->bnm', A, self.u_particles)
        obs = obs_var[:, None, None].clamp_min(1e-12)
        diff = R_batch[:, :, None] - mu_particles
        loglik = (-0.5 * (torch.log(2.0 * math.pi * obs) + diff.pow(2) / obs)).sum(dim=0)
        logw = torch.log(self.weights.clamp_min(1e-30)) + loglik
        logw = logw - torch.logsumexp(logw, dim=1, keepdim=True)
        self.weights = torch.exp(logw)
        self.last_ess = 1.0 / self.weights.clamp_min(1e-30).pow(2).sum(dim=1)
        self._resample_rows(self.last_ess < self.resample_ess_ratio * self.num_mc_particles)
        self.last_ess = 1.0 / self.weights.clamp_min(1e-30).pow(2).sum(dim=1)

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.predict_for_particles(x, self.theta_particles)

    def predict_for_particles(
        self,
        x: torch.Tensor,
        theta_particles: torch.Tensor,
        *,
        emulator: Optional[Emulator] = None,
        rho: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del theta_particles
        if emulator is not None:
            self.emulator = emulator
        if rho is not None:
            self.rho = float(rho)
        if self.hyper_spec is None or self.Z is None:
            b = int(x.shape[0])
            n = int(self.theta_particles.shape[0])
            zeros = torch.zeros(b, n, device=self.device, dtype=self.dtype)
            return zeros, zeros
        A, obs_var = self._mapping(x)
        mu_particles = torch.einsum('bj,nmj->bnm', A, self.u_particles)
        w = self.weights.unsqueeze(0)
        mu = (w * mu_particles).sum(dim=2)
        second = (w * (mu_particles.pow(2) + obs_var[:, None, None])).sum(dim=2)
        var = (second - mu.pow(2)).clamp_min(1e-12)
        return mu, var

    def loglik_for_particles(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        theta_particles: torch.Tensor,
        *,
        emulator: Optional[Emulator] = None,
        rho: float,
        sigma_eps: float,
        mu_eta: Optional[torch.Tensor] = None,
        var_eta: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        del theta_particles
        if emulator is not None:
            self.emulator = emulator
        if mu_eta is None or var_eta is None:
            return None
        if mu_eta.dim() != 2 or var_eta.dim() != 2 or self.hyper_spec is None or self.Z is None:
            return None
        A, obs_var = self._mapping(x)
        mu_delta_particles = torch.einsum('bj,nmj->bnm', A, self.u_particles)
        total_mu = rho * mu_eta[:, :, None] + mu_delta_particles
        total_var = ((rho ** 2) * var_eta[:, :, None] + obs_var[:, None, None] + (sigma_eps ** 2)).clamp_min(1e-12)
        diff = y.reshape(-1, 1, 1) - total_mu
        loglik_bnm = -0.5 * (torch.log(2.0 * math.pi * total_var) + diff.pow(2) / total_var)
        loglik_nm = loglik_bnm.sum(dim=0)
        return torch.logsumexp(loglik_nm + torch.log(self.weights.clamp_min(1e-30)), dim=1)

    def copy(self) -> 'ParticleSpecificMCInducingDeltaState':
        out = ParticleSpecificMCInducingDeltaState(
            theta_particles=self.theta_particles.clone(),
            emulator=self.emulator,
            rho=self.rho,
            hyper_spec=self.hyper_spec,
            min_points=self.min_points,
            num_inducing=self.num_inducing,
            num_mc_particles=self.num_mc_particles,
            resample_ess_ratio=self.resample_ess_ratio,
            include_conditional_var=self.include_conditional_var,
        )
        if self.Z is not None:
            out.Z = self.Z.clone()
            out.num_inducing = int(self.Z.shape[0])
            out._build_kernel_cache()
        out.u_particles = self.u_particles.clone()
        out.weights = self.weights.clone()
        out.last_ess = self.last_ess.clone()
        out.num_resamples = int(self.num_resamples)
        out.num_refreshes = int(self.num_refreshes)
        return out


class ParticleSpecificDynamicBasisDeltaState:
    """
    Lineage-bound dynamic discrepancy filter for particle-specific residuals.

    Each particle lineage carries its own coefficient mean, while the basis covariance is
    shared because the observation design is the same across particles. Resampling copies a
    parent's coefficient mean into each child, and children continue updating online.
    """

    def __init__(
        self,
        theta_particles: torch.Tensor,
        emulator: Emulator,
        rho: float,
        *,
        hyper_spec: Optional[KernelHyperSpec] = None,
        min_points: int = 3,
        num_features: int = 8,
        forgetting: float = 0.98,
        process_noise_scale: float = 1e-3,
        prior_var_scale: float = 1.0,
        buffer_max_points: int = 256,
    ):
        self.theta_particles = theta_particles.detach().clone()
        self.emulator = emulator
        self.rho = float(rho)
        self.device = theta_particles.device
        self.dtype = theta_particles.dtype
        self.dx: Optional[int] = None
        self.min_points = max(int(min_points), 1)
        self.num_features = max(int(num_features), 1)
        self.forgetting = min(max(float(forgetting), 1e-4), 1.0)
        self.process_noise_scale = max(float(process_noise_scale), 0.0)
        self.prior_var_scale = max(float(prior_var_scale), 1e-6)
        self.buffer_max_points = max(int(buffer_max_points), self.min_points)
        self.X_hist = torch.empty(0, 0, device=self.device, dtype=self.dtype)
        self.R_hist = torch.empty(0, int(theta_particles.shape[0]), device=self.device, dtype=self.dtype)
        self.batch_sizes: List[int] = []
        self.hyper_spec: Optional[KernelHyperSpec] = None
        self.centers: Optional[torch.Tensor] = None
        self.mean_particles: Optional[torch.Tensor] = None
        self.cov: Optional[torch.Tensor] = None
        self._identity: Optional[torch.Tensor] = None
        if hyper_spec is not None:
            self.set_hyper_spec(hyper_spec)

    def _trim_batch_sizes(self, keep_n: int) -> None:
        keep_n = max(int(keep_n), 0)
        if keep_n <= 0 or len(self.batch_sizes) == 0:
            self.batch_sizes = []
            return
        rem = keep_n
        kept: List[int] = []
        for size in reversed(self.batch_sizes):
            size_i = int(size)
            if rem <= 0:
                break
            take = min(size_i, rem)
            kept.append(take)
            rem -= take
        self.batch_sizes = list(reversed(kept))

    def _append_batch_size(self, size: int) -> None:
        self.batch_sizes.append(int(size))

    def _trim_history(self, keep_n: int) -> None:
        keep_n = max(int(keep_n), 0)
        if keep_n <= 0 or self.X_hist.numel() == 0:
            self.X_hist = torch.empty(0, 0 if self.dx is None else self.dx, device=self.device, dtype=self.dtype)
            self.R_hist = torch.empty(0, self.theta_particles.shape[0], device=self.device, dtype=self.dtype)
            self.batch_sizes = []
            return
        if keep_n < int(self.X_hist.shape[0]):
            self.X_hist = self.X_hist[-keep_n:].clone()
            self.R_hist = self.R_hist[-keep_n:].clone()
        self._trim_batch_sizes(keep_n)

    def _select_centers(self, X_hist: torch.Tensor) -> torch.Tensor:
        n = int(X_hist.shape[0])
        m = max(1, min(self.num_features, n))
        if m == n:
            idx = torch.arange(n, device=self.device)
        else:
            idx = torch.linspace(0, n - 1, m, device=self.device).round().long().unique(sorted=True)
            if idx.numel() < m:
                idx = torch.arange(m, device=self.device)
        return X_hist[idx].clone()

    def _basis(self, X: torch.Tensor) -> torch.Tensor:
        if self.centers is None or self.hyper_spec is None:
            raise ValueError('dynamic particle discrepancy basis is not initialized')
        X = X.to(self.device, self.dtype)
        ls = _to_lengthscale_tensor(self.hyper_spec.lengthscale, self.dx, device=self.device, dtype=self.dtype)
        scaled = (X[:, None, :] - self.centers[None, :, :]) / ls.view(1, 1, -1)
        dist2 = scaled.pow(2).sum(dim=-1)
        feats = torch.exp(-0.5 * dist2)
        ones = torch.ones(X.shape[0], 1, device=self.device, dtype=self.dtype)
        return torch.cat([ones, feats], dim=1)

    def _initial_covariance(self) -> torch.Tensor:
        if self.hyper_spec is None:
            raise ValueError('hyper_spec must be set before initializing particle dynamic state')
        prior_var = max(self.prior_var_scale * self.hyper_spec.variance, self.hyper_spec.noise)
        p = 1 + int(self.centers.shape[0])
        return prior_var * torch.eye(p, device=self.device, dtype=self.dtype)

    def _symmetrize_cov(self) -> None:
        if self.cov is None or self._identity is None or self.hyper_spec is None:
            return
        self.cov = 0.5 * (self.cov + self.cov.transpose(0, 1))
        self.cov = self.cov + max(self.hyper_spec.noise, 1e-8) * 1e-8 * self._identity

    def _propagate(self) -> None:
        if self.cov is None or self._identity is None or self.hyper_spec is None:
            return
        if self.forgetting < 1.0:
            self.cov = self.cov / self.forgetting
        if self.process_noise_scale > 0.0:
            q = self.process_noise_scale * max(self.hyper_spec.variance, self.hyper_spec.noise)
            self.cov = self.cov + q * self._identity
        self._symmetrize_cov()

    def _update_with_residual_batch(self, X_batch: torch.Tensor, R_batch: torch.Tensor, *, propagate_first: bool) -> None:
        if self.hyper_spec is None or self.centers is None or self.mean_particles is None or self.cov is None:
            return
        X_batch = X_batch.to(self.device, self.dtype).reshape(-1, self.dx)
        R_batch = R_batch.to(self.device, self.dtype).reshape(X_batch.shape[0], -1)
        if propagate_first:
            self._propagate()
        Phi = self._basis(X_batch)
        for i in range(Phi.shape[0]):
            phi = Phi[i]
            Pphi = self.cov @ phi
            innovation_var = torch.clamp(phi @ Pphi + self.hyper_spec.noise, min=1e-10)
            gain = Pphi / innovation_var
            innovation = R_batch[i].reshape(1, -1) - (phi @ self.mean_particles).reshape(1, -1)
            self.mean_particles = self.mean_particles + gain[:, None] * innovation
            KH = torch.outer(gain, phi)
            self.cov = (self._identity - KH) @ self.cov @ (self._identity - KH).transpose(0, 1)
            self.cov = self.cov + self.hyper_spec.noise * torch.outer(gain, gain)
            self._symmetrize_cov()

    def _rebuild_state_from_history(self) -> None:
        if self.hyper_spec is None or self.X_hist.shape[0] < self.min_points or self.R_hist.numel() == 0:
            self.centers = None
            self.mean_particles = None
            self.cov = None
            self._identity = None
            return
        self.centers = self._select_centers(self.X_hist)
        p = 1 + int(self.centers.shape[0])
        self.mean_particles = torch.zeros(p, self.theta_particles.shape[0], device=self.device, dtype=self.dtype)
        self.cov = self._initial_covariance()
        self._identity = torch.eye(p, device=self.device, dtype=self.dtype)
        offset = 0
        sizes = self.batch_sizes if len(self.batch_sizes) > 0 else [int(self.X_hist.shape[0])]
        for i, size in enumerate(sizes):
            end = min(offset + int(size), int(self.X_hist.shape[0]))
            if end <= offset:
                continue
            self._update_with_residual_batch(self.X_hist[offset:end], self.R_hist[offset:end], propagate_first=(i > 0))
            offset = end
            if offset >= int(self.X_hist.shape[0]):
                break
        if offset < int(self.X_hist.shape[0]):
            self._update_with_residual_batch(self.X_hist[offset:], self.R_hist[offset:], propagate_first=(offset > 0))

    def copy(self) -> 'ParticleSpecificDynamicBasisDeltaState':
        out = ParticleSpecificDynamicBasisDeltaState(
            theta_particles=self.theta_particles.clone(),
            emulator=self.emulator,
            rho=self.rho,
            hyper_spec=self.hyper_spec,
            min_points=self.min_points,
            num_features=self.num_features,
            forgetting=self.forgetting,
            process_noise_scale=self.process_noise_scale,
            prior_var_scale=self.prior_var_scale,
            buffer_max_points=self.buffer_max_points,
        )
        out.dx = self.dx
        out.X_hist = self.X_hist.clone()
        out.R_hist = self.R_hist.clone()
        out.batch_sizes = list(self.batch_sizes)
        out.centers = None if self.centers is None else self.centers.clone()
        out.mean_particles = None if self.mean_particles is None else self.mean_particles.clone()
        out.cov = None if self.cov is None else self.cov.clone()
        out._identity = None if self._identity is None else self._identity.clone()
        return out

    def set_hyper_spec(self, hyper_spec: KernelHyperSpec) -> None:
        if self.dx is None:
            self.dx = int(torch.as_tensor(hyper_spec.lengthscale, device=self.device, dtype=self.dtype).reshape(-1).numel())
        self.hyper_spec = KernelHyperSpec(
            lengthscale=_to_lengthscale_tensor(hyper_spec.lengthscale, self.dx, device=self.device, dtype=self.dtype),
            variance=max(float(hyper_spec.variance), 1e-8),
            noise=max(float(hyper_spec.noise), 1e-8),
        )
        self._rebuild_state_from_history()

    def append_batch(
        self,
        X_batch: torch.Tensor,
        Y_batch: torch.Tensor,
        theta_particles: torch.Tensor,
        *,
        ancestor_indices: Optional[torch.Tensor] = None,
        emulator: Optional[Emulator] = None,
        rho: Optional[float] = None,
    ) -> None:
        X_batch = X_batch.to(self.device, self.dtype)
        Y_batch = Y_batch.to(self.device, self.dtype).reshape(-1)
        theta_particles = theta_particles.to(self.device, self.dtype)
        if emulator is not None:
            self.emulator = emulator
        if rho is not None:
            self.rho = float(rho)
        if self.dx is None:
            self.dx = int(X_batch.shape[1])
        if ancestor_indices is not None:
            idx = ancestor_indices.to(self.device).long()
            if self.R_hist.numel() > 0:
                self.R_hist = self.R_hist[:, idx]
            if self.mean_particles is not None:
                self.mean_particles = self.mean_particles[:, idx]
        self.theta_particles = theta_particles.detach().clone()
        mu_eta_batch, _ = self.emulator.predict(X_batch, theta_particles)
        if mu_eta_batch.dim() == 3:
            mu_eta_batch = mu_eta_batch.mean(dim=-1)
        R_batch = Y_batch[:, None] - self.rho * mu_eta_batch
        if self.X_hist.numel() == 0:
            self.X_hist = X_batch.clone()
            self.R_hist = R_batch.clone()
        else:
            self.X_hist = torch.cat([self.X_hist, X_batch], dim=0)
            self.R_hist = torch.cat([self.R_hist, R_batch], dim=0)
        self._append_batch_size(int(X_batch.shape[0]))
        if self.X_hist.shape[0] > self.buffer_max_points:
            self._trim_history(self.buffer_max_points)
        if self.hyper_spec is None:
            return
        if self.mean_particles is None or self.cov is None or self.centers is None:
            self._rebuild_state_from_history()
            return
        self._update_with_residual_batch(X_batch, R_batch, propagate_first=True)

    def truncate_recent(self, keep_n: int) -> None:
        self._trim_history(keep_n)
        self._rebuild_state_from_history()

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b = int(x.shape[0])
        n = int(self.theta_particles.shape[0])
        if self.mean_particles is None or self.cov is None or self.centers is None or self.hyper_spec is None:
            zeros = torch.zeros(b, n, device=self.device, dtype=self.dtype)
            return zeros, zeros
        Phi = self._basis(x)
        mu = Phi @ self.mean_particles
        cov_proj = Phi @ self.cov
        base_var = (cov_proj * Phi).sum(dim=1).clamp_min(0.0) + self.hyper_spec.noise
        var = base_var[:, None].expand(-1, n).clamp_min(1e-12)
        return mu, var

    def predict_for_particles(
        self,
        x: torch.Tensor,
        theta_particles: torch.Tensor,
        *,
        emulator: Optional[Emulator] = None,
        rho: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del theta_particles
        if emulator is not None:
            self.emulator = emulator
        if rho is not None:
            self.rho = float(rho)
        return self.predict(x)

