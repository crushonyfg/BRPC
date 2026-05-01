import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch


def _pairwise_sq(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    diff = a[:, None, :] - b[None, :, :]
    return (diff * diff).sum(dim=-1)


def _safe_cholesky_batched(
    K: torch.Tensor,
    jitter_init: float = 1e-8,
    jitter_mult: float = 10.0,
    max_tries: int = 6,
) -> Tuple[torch.Tensor, float]:
    eye = torch.eye(K.shape[-1], device=K.device, dtype=K.dtype)
    jitter = float(jitter_init)
    for _ in range(max_tries):
        try:
            L = torch.linalg.cholesky(K + jitter * eye.unsqueeze(0))
            return L, jitter
        except RuntimeError:
            jitter *= float(jitter_mult)
    L = torch.linalg.cholesky(K + jitter * eye.unsqueeze(0))
    return L, jitter


@dataclass
class WardPaperPFConfig:
    num_particles: int = 1024
    theta_lo: float = 0.0
    theta_hi: float = 3.0
    rho: float = 1.0
    emulator_var: float = 1.0
    discrepancy_var: float = 1.0 / (10.0 / 0.3)
    sigma_obs_var: float = 1.0 / (10.0 / 0.03)
    design_x_points: int = 5
    design_theta_points: int = 7
    x_domain: object = None
    x_design_np: object = None
    prior_l_median: float = 0.30
    prior_l_logsd: float = 0.50
    l_min: float = 0.05
    l_max: float = 3.00
    move_theta_std: float = 0.0
    move_logl_std: float = 0.0
    jitter: float = 1e-8
    device: str = "cpu"
    dtype: torch.dtype = torch.float64
    seed: int = 0


@dataclass
class WardPaperPFVectorConfig:
    num_particles: int = 1024
    theta_dim: int = 5
    theta_lo: float = -2.0
    theta_hi: float = 2.0
    rho: float = 1.0
    emulator_var: float = 1.0
    discrepancy_var: float = 0.09
    sigma_obs_var: float = 0.0025
    design_x_points: int = 8
    design_theta_points: int = 8
    x_domain: object = None
    x_design_np: object = None
    theta_design_np: object = None
    prior_l_median: float = 2.0
    prior_l_logsd: float = 0.50
    l_min: float = 0.20
    l_max: float = 8.00
    move_theta_std: float = 0.15
    move_logl_std: float = 0.10
    jitter: float = 1e-8
    device: str = "cpu"
    dtype: torch.dtype = torch.float64
    seed: int = 0


class WardPaperParticleFilter:
    """
    Particle filter following Ward et al. (2021) as closely as the paper
    specifies for the synthetic toy setting:
    - particles over (theta, l)
    - rho fixed to 1
    - measurement variance fixed
    - GP emulator over prior simulator runs
    - GP discrepancy term in likelihood
    - weight update then resample at every step
    - no rejuvenation / move step

    One paper detail is not specified: the lognormal parameters of the
    length-scale prior. Those are therefore exposed explicitly in the config.
    """

    def __init__(self, sim_func_np, cfg: WardPaperPFConfig):
        self.sim = sim_func_np
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.dtype = cfg.dtype
        self.rng = np.random.default_rng(int(cfg.seed))

        self.theta = self.rng.uniform(
            low=float(cfg.theta_lo),
            high=float(cfg.theta_hi),
            size=int(cfg.num_particles),
        ).astype(np.float64)
        self.lengthscale = self.rng.lognormal(
            mean=math.log(float(cfg.prior_l_median)),
            sigma=float(cfg.prior_l_logsd),
            size=int(cfg.num_particles),
        ).astype(np.float64)
        self.lengthscale = np.clip(
            self.lengthscale,
            float(cfg.l_min),
            float(cfg.l_max),
        )

        if cfg.x_design_np is not None:
            Xd_base_np = np.asarray(cfg.x_design_np, dtype=np.float64)
            if Xd_base_np.ndim == 1:
                Xd_base_np = Xd_base_np.reshape(-1, 1)
        elif cfg.x_domain is not None:
            x_domain = list(cfg.x_domain)
            x_dim = len(x_domain)
            Xd_base_np = np.empty((int(cfg.design_x_points), x_dim), dtype=np.float64)
            for j, (lo, hi) in enumerate(x_domain):
                Xd_base_np[:, j] = self.rng.uniform(float(lo), float(hi), size=int(cfg.design_x_points))
        else:
            x_nodes = np.linspace(0.0, 1.0, int(cfg.design_x_points), dtype=np.float64)
            Xd_base_np = x_nodes.reshape(-1, 1)
        theta_nodes = np.linspace(
            float(cfg.theta_lo),
            float(cfg.theta_hi),
            int(cfg.design_theta_points),
            dtype=np.float64,
        )
        Xd = []
        Td = []
        for th in theta_nodes:
            for x_row in Xd_base_np:
                Xd.append(np.asarray(x_row, dtype=np.float64).reshape(-1).tolist())
                Td.append([th])
        Xd_np = np.asarray(Xd, dtype=np.float64).reshape(-1, Xd_base_np.shape[1])
        Td_np = np.asarray(Td, dtype=np.float64)
        d_np = self.sim(Xd_np, Td_np).reshape(-1).astype(np.float64)

        self.Zd = torch.tensor(
            np.concatenate([Xd_np, Td_np], axis=1),
            dtype=self.dtype,
            device=self.device,
        )
        self.Zd_x = self.Zd[:, :-1]
        self.Zd_theta = self.Zd[:, -1:]
        self.d = torch.tensor(d_np, dtype=self.dtype, device=self.device)
        self.D_dd = _pairwise_sq(self.Zd, self.Zd)

    def _current_particles(self) -> Tuple[torch.Tensor, torch.Tensor]:
        theta = torch.tensor(self.theta, dtype=self.dtype, device=self.device)
        l = torch.tensor(self.lengthscale, dtype=self.dtype, device=self.device)
        l = torch.clamp(l, min=float(self.cfg.l_min), max=float(self.cfg.l_max))
        return theta, l

    def _particle_predictive(
        self,
        X_batch_np: np.ndarray,
        Y_batch_np: np.ndarray = None,
    ) -> Dict[str, torch.Tensor]:
        X_np = np.asarray(X_batch_np, dtype=np.float64).reshape(-1, 1)
        if np.asarray(X_batch_np).ndim > 1:
            X_np = np.asarray(X_batch_np, dtype=np.float64)
        x = torch.tensor(X_np, dtype=self.dtype, device=self.device)
        theta, l = self._current_particles()
        n_particles = theta.shape[0]
        batch_size = x.shape[0]
        M = self.Zd.shape[0]

        l2 = (l * l).view(n_particles, 1, 1)
        eye_M = torch.eye(M, dtype=self.dtype, device=self.device).unsqueeze(0)
        eye_B = torch.eye(batch_size, dtype=self.dtype, device=self.device).unsqueeze(0)

        Kdd = float(self.cfg.emulator_var) * torch.exp(-0.5 * self.D_dd.unsqueeze(0) / l2)
        Kdd = Kdd + float(self.cfg.jitter) * eye_M
        Ldd, _ = _safe_cholesky_batched(Kdd, jitter_init=float(self.cfg.jitter))

        d_rhs = self.d.view(1, M, 1).expand(n_particles, -1, -1)
        alpha = torch.cholesky_solve(d_rhs, Ldd).squeeze(-1)

        x_batch = x.view(1, 1, batch_size, -1)
        zd_x = self.Zd_x.view(1, M, 1, -1)
        theta_batch = theta.view(n_particles, 1, 1, 1)
        zd_theta = self.Zd_theta.view(1, M, 1, 1)
        D_ds = ((zd_x - x_batch) ** 2).sum(dim=-1) + ((zd_theta - theta_batch) ** 2).squeeze(-1)
        Kds = float(self.cfg.emulator_var) * torch.exp(-0.5 * D_ds / l2)
        mu = torch.einsum("nmb,nm->nb", Kds, alpha)

        x_aug = x
        D_xx = _pairwise_sq(x_aug, x_aug)
        Kss_eta = float(self.cfg.emulator_var) * torch.exp(-0.5 * D_xx.unsqueeze(0) / l2)
        V = torch.cholesky_solve(Kds, Ldd)
        C_eta = Kss_eta - torch.einsum("nmb,nmc->nbc", Kds, V)
        C_eta = 0.5 * (C_eta + C_eta.transpose(-1, -2))

        K_delta = float(self.cfg.discrepancy_var) * torch.exp(-0.5 * D_xx.unsqueeze(0) / l2)
        Sigma = C_eta + K_delta + float(self.cfg.sigma_obs_var) * eye_B
        Sigma = 0.5 * (Sigma + Sigma.transpose(-1, -2))

        out = {
            "mu_particles": mu,
            "Sigma_particles": Sigma,
            "mu_mix": mu.mean(dim=0),
        }
        second = (
            torch.diagonal(Sigma, dim1=-2, dim2=-1)
            + mu * mu
        ).mean(dim=0)
        out["var_mix"] = torch.clamp(second - out["mu_mix"] * out["mu_mix"], min=1e-12)

        if Y_batch_np is not None:
            y = torch.tensor(
                np.asarray(Y_batch_np, dtype=np.float64).reshape(-1),
                dtype=self.dtype,
                device=self.device,
            )
            diff = y.view(1, batch_size) - mu
            LSigma, _ = _safe_cholesky_batched(Sigma, jitter_init=float(self.cfg.jitter))
            solve = torch.cholesky_solve(diff.unsqueeze(-1), LSigma).squeeze(-1)
            quad = (diff * solve).sum(dim=1)
            logdet = 2.0 * torch.log(torch.diagonal(LSigma, dim1=-2, dim2=-1)).sum(dim=1)
            ll = -0.5 * (
                quad
                + logdet
                + batch_size * math.log(2.0 * math.pi)
            )
            out["loglik_particles"] = ll
        return out

    def _systematic_resample(self, weights: np.ndarray) -> np.ndarray:
        N = int(weights.shape[0])
        positions = (self.rng.random() + np.arange(N)) / N
        cumsum = np.cumsum(weights)
        indexes = np.zeros(N, dtype=np.int64)
        i = 0
        j = 0
        while i < N:
            if positions[i] < cumsum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        return indexes

    def _move_step(self) -> None:
        theta_std = float(self.cfg.move_theta_std)
        logl_std = float(self.cfg.move_logl_std)
        if theta_std > 0.0:
            self.theta = self.theta + self.rng.normal(
                loc=0.0,
                scale=theta_std,
                size=self.theta.shape,
            )
            self.theta = np.clip(
                self.theta,
                float(self.cfg.theta_lo),
                float(self.cfg.theta_hi),
            )
        if logl_std > 0.0:
            self.lengthscale = self.lengthscale * np.exp(
                self.rng.normal(
                    loc=0.0,
                    scale=logl_std,
                    size=self.lengthscale.shape,
                )
            )
            self.lengthscale = np.clip(
                self.lengthscale,
                float(self.cfg.l_min),
                float(self.cfg.l_max),
            )

    def predict_batch(self, X_batch_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        out = self._particle_predictive(X_batch_np, None)
        return (
            out["mu_mix"].detach().cpu().numpy().reshape(-1),
            out["var_mix"].detach().cpu().numpy().reshape(-1),
        )

    def compute_loglik_batch(self, X_batch_np: np.ndarray, Y_batch_np: np.ndarray) -> np.ndarray:
        out = self._particle_predictive(X_batch_np, Y_batch_np)
        return out["loglik_particles"].detach().cpu().numpy().reshape(-1).copy()

    def step_batch(self, X_batch_np: np.ndarray, Y_batch_np: np.ndarray) -> Dict[str, np.ndarray]:
        out = self._particle_predictive(X_batch_np, Y_batch_np)
        ll = out["loglik_particles"].detach().cpu().numpy()
        ll = ll - np.max(ll)
        w = np.exp(ll)
        w = w / np.clip(w.sum(), 1e-300, None)
        idx = self._systematic_resample(w)
        self.theta = self.theta[idx]
        self.lengthscale = self.lengthscale[idx]
        self._move_step()
        return {
            "pred_mu": out["mu_mix"].detach().cpu().numpy().reshape(-1),
            "pred_var": out["var_mix"].detach().cpu().numpy().reshape(-1),
            "weights_pre_resample": w.copy(),
            "theta_particles": self.theta.copy(),
            "lengthscale_particles": self.lengthscale.copy(),
        }

    def posterior_mean_var(self) -> Tuple[float, float]:
        mean = float(np.mean(self.theta))
        var = float(np.var(self.theta))
        return mean, var


class WardPaperParticleFilterVector:
    """
    High-dimensional generalization of WardPaperParticleFilter.

    This keeps the same structural ingredients as the 1D implementation:
    - particles over (theta, l)
    - fixed rho
    - fixed sigma_obs_var and discrepancy_var
    - GP emulator over prior simulator design runs on the joint (x, theta) input
    - GP discrepancy covariance term in the likelihood
    - weight update, systematic resampling, optional move step on theta and log l

    The only unavoidable generalization is that the theta-design is a set of
    sampled theta vectors rather than a scalar linspace grid.
    """

    def __init__(self, sim_func_np, cfg: WardPaperPFVectorConfig):
        self.sim = sim_func_np
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.dtype = cfg.dtype
        self.rng = np.random.default_rng(int(cfg.seed))
        self.theta_dim = int(cfg.theta_dim)

        self.theta = self.rng.uniform(
            low=float(cfg.theta_lo),
            high=float(cfg.theta_hi),
            size=(int(cfg.num_particles), self.theta_dim),
        ).astype(np.float64)
        self.lengthscale = self.rng.lognormal(
            mean=math.log(float(cfg.prior_l_median)),
            sigma=float(cfg.prior_l_logsd),
            size=int(cfg.num_particles),
        ).astype(np.float64)
        self.lengthscale = np.clip(
            self.lengthscale,
            float(cfg.l_min),
            float(cfg.l_max),
        )

        if cfg.x_design_np is not None:
            Xd_base_np = np.asarray(cfg.x_design_np, dtype=np.float64)
            if Xd_base_np.ndim == 1:
                Xd_base_np = Xd_base_np.reshape(-1, 1)
        elif cfg.x_domain is not None:
            x_domain = list(cfg.x_domain)
            x_dim = len(x_domain)
            Xd_base_np = np.empty((int(cfg.design_x_points), x_dim), dtype=np.float64)
            for j, (lo, hi) in enumerate(x_domain):
                Xd_base_np[:, j] = self.rng.uniform(float(lo), float(hi), size=int(cfg.design_x_points))
        else:
            x_nodes = np.linspace(0.0, 1.0, int(cfg.design_x_points), dtype=np.float64)
            Xd_base_np = x_nodes.reshape(-1, 1)

        if cfg.theta_design_np is not None:
            Td_base_np = np.asarray(cfg.theta_design_np, dtype=np.float64)
            if Td_base_np.ndim == 1:
                Td_base_np = Td_base_np.reshape(-1, self.theta_dim)
        else:
            Td_base_np = self.rng.uniform(
                low=float(cfg.theta_lo),
                high=float(cfg.theta_hi),
                size=(int(cfg.design_theta_points), self.theta_dim),
            ).astype(np.float64)

        Xd = []
        Td = []
        for theta_row in Td_base_np:
            for x_row in Xd_base_np:
                Xd.append(np.asarray(x_row, dtype=np.float64).reshape(-1).tolist())
                Td.append(np.asarray(theta_row, dtype=np.float64).reshape(-1).tolist())
        Xd_np = np.asarray(Xd, dtype=np.float64).reshape(-1, Xd_base_np.shape[1])
        Td_np = np.asarray(Td, dtype=np.float64).reshape(-1, self.theta_dim)
        d_np = self.sim(Xd_np, Td_np).reshape(-1).astype(np.float64)

        self.Zd = torch.tensor(
            np.concatenate([Xd_np, Td_np], axis=1),
            dtype=self.dtype,
            device=self.device,
        )
        self.Zd_x = self.Zd[:, :-self.theta_dim]
        self.Zd_theta = self.Zd[:, -self.theta_dim:]
        self.d = torch.tensor(d_np, dtype=self.dtype, device=self.device)
        self.D_dd = _pairwise_sq(self.Zd, self.Zd)

    def _current_particles(self) -> Tuple[torch.Tensor, torch.Tensor]:
        theta = torch.tensor(self.theta, dtype=self.dtype, device=self.device)
        l = torch.tensor(self.lengthscale, dtype=self.dtype, device=self.device)
        l = torch.clamp(l, min=float(self.cfg.l_min), max=float(self.cfg.l_max))
        return theta, l

    def _particle_predictive(
        self,
        X_batch_np: np.ndarray,
        Y_batch_np: np.ndarray = None,
    ) -> Dict[str, torch.Tensor]:
        X_np = np.asarray(X_batch_np, dtype=np.float64)
        if X_np.ndim == 1:
            X_np = X_np.reshape(-1, 1)
        x = torch.tensor(X_np, dtype=self.dtype, device=self.device)
        theta, l = self._current_particles()
        n_particles = theta.shape[0]
        batch_size = x.shape[0]
        M = self.Zd.shape[0]

        l2 = (l * l).view(n_particles, 1, 1)
        eye_M = torch.eye(M, dtype=self.dtype, device=self.device).unsqueeze(0)
        eye_B = torch.eye(batch_size, dtype=self.dtype, device=self.device).unsqueeze(0)

        Kdd = float(self.cfg.emulator_var) * torch.exp(-0.5 * self.D_dd.unsqueeze(0) / l2)
        Kdd = Kdd + float(self.cfg.jitter) * eye_M
        Ldd, _ = _safe_cholesky_batched(Kdd, jitter_init=float(self.cfg.jitter))

        d_rhs = self.d.view(1, M, 1).expand(n_particles, -1, -1)
        alpha = torch.cholesky_solve(d_rhs, Ldd).squeeze(-1)

        x_batch = x.view(1, 1, batch_size, -1)
        zd_x = self.Zd_x.view(1, M, 1, -1)
        theta_batch = theta.view(n_particles, 1, 1, self.theta_dim)
        zd_theta = self.Zd_theta.view(1, M, 1, self.theta_dim)
        D_ds = ((zd_x - x_batch) ** 2).sum(dim=-1) + ((zd_theta - theta_batch) ** 2).sum(dim=-1)
        Kds = float(self.cfg.emulator_var) * torch.exp(-0.5 * D_ds / l2)
        mu = torch.einsum("nmb,nm->nb", Kds, alpha)

        D_xx = _pairwise_sq(x, x)
        Kss_eta = float(self.cfg.emulator_var) * torch.exp(-0.5 * D_xx.unsqueeze(0) / l2)
        V = torch.cholesky_solve(Kds, Ldd)
        C_eta = Kss_eta - torch.einsum("nmb,nmc->nbc", Kds, V)
        C_eta = 0.5 * (C_eta + C_eta.transpose(-1, -2))

        K_delta = float(self.cfg.discrepancy_var) * torch.exp(-0.5 * D_xx.unsqueeze(0) / l2)
        Sigma = C_eta + K_delta + float(self.cfg.sigma_obs_var) * eye_B
        Sigma = 0.5 * (Sigma + Sigma.transpose(-1, -2))

        out = {
            "mu_particles": mu,
            "Sigma_particles": Sigma,
            "mu_mix": mu.mean(dim=0),
        }
        second = (torch.diagonal(Sigma, dim1=-2, dim2=-1) + mu * mu).mean(dim=0)
        out["var_mix"] = torch.clamp(second - out["mu_mix"] * out["mu_mix"], min=1e-12)

        if Y_batch_np is not None:
            y = torch.tensor(
                np.asarray(Y_batch_np, dtype=np.float64).reshape(-1),
                dtype=self.dtype,
                device=self.device,
            )
            diff = y.view(1, batch_size) - mu
            LSigma, _ = _safe_cholesky_batched(Sigma, jitter_init=float(self.cfg.jitter))
            solve = torch.cholesky_solve(diff.unsqueeze(-1), LSigma).squeeze(-1)
            quad = (diff * solve).sum(dim=1)
            logdet = 2.0 * torch.log(torch.diagonal(LSigma, dim1=-2, dim2=-1)).sum(dim=1)
            ll = -0.5 * (
                quad
                + logdet
                + batch_size * math.log(2.0 * math.pi)
            )
            out["loglik_particles"] = ll
        return out

    def _systematic_resample(self, weights: np.ndarray) -> np.ndarray:
        N = int(weights.shape[0])
        positions = (self.rng.random() + np.arange(N)) / N
        cumsum = np.cumsum(weights)
        indexes = np.zeros(N, dtype=np.int64)
        i = 0
        j = 0
        while i < N:
            if positions[i] < cumsum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        return indexes

    def _move_step(self) -> None:
        theta_std = float(self.cfg.move_theta_std)
        logl_std = float(self.cfg.move_logl_std)
        if theta_std > 0.0:
            self.theta = self.theta + self.rng.normal(
                loc=0.0,
                scale=theta_std,
                size=self.theta.shape,
            )
            self.theta = np.clip(
                self.theta,
                float(self.cfg.theta_lo),
                float(self.cfg.theta_hi),
            )
        if logl_std > 0.0:
            self.lengthscale = self.lengthscale * np.exp(
                self.rng.normal(
                    loc=0.0,
                    scale=logl_std,
                    size=self.lengthscale.shape,
                )
            )
            self.lengthscale = np.clip(
                self.lengthscale,
                float(self.cfg.l_min),
                float(self.cfg.l_max),
            )

    def predict_batch(self, X_batch_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        out = self._particle_predictive(X_batch_np, None)
        return (
            out["mu_mix"].detach().cpu().numpy().reshape(-1),
            out["var_mix"].detach().cpu().numpy().reshape(-1),
        )

    def compute_loglik_batch(self, X_batch_np: np.ndarray, Y_batch_np: np.ndarray) -> np.ndarray:
        out = self._particle_predictive(X_batch_np, Y_batch_np)
        return out["loglik_particles"].detach().cpu().numpy().reshape(-1).copy()

    def step_batch(self, X_batch_np: np.ndarray, Y_batch_np: np.ndarray) -> Dict[str, np.ndarray]:
        out = self._particle_predictive(X_batch_np, Y_batch_np)
        ll = out["loglik_particles"].detach().cpu().numpy()
        ll = ll - np.max(ll)
        w = np.exp(ll)
        w = w / np.clip(w.sum(), 1e-300, None)
        idx = self._systematic_resample(w)
        self.theta = self.theta[idx]
        self.lengthscale = self.lengthscale[idx]
        self._move_step()
        return {
            "pred_mu": out["mu_mix"].detach().cpu().numpy().reshape(-1),
            "pred_var": out["var_mix"].detach().cpu().numpy().reshape(-1),
            "weights_pre_resample": w.copy(),
            "theta_particles": self.theta.copy(),
            "lengthscale_particles": self.lengthscale.copy(),
        }

    def posterior_mean_var_diag(self) -> Tuple[np.ndarray, np.ndarray]:
        mean = np.asarray(np.mean(self.theta, axis=0), dtype=np.float64)
        var_diag = np.asarray(np.var(self.theta, axis=0), dtype=np.float64)
        return mean, var_diag
