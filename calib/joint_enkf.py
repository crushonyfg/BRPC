from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np


def rbf_basis_1d(
    x: np.ndarray,
    centers: np.ndarray,
    lengthscale: float,
    include_linear: bool = True,
) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    if x_arr.ndim == 1:
        x_arr = x_arr[:, None]
    xx = x_arr[:, [0]]
    cc = np.asarray(centers, dtype=float).reshape(1, -1)
    ls = max(float(lengthscale), 1e-8)
    rbf = np.exp(-0.5 * ((xx - cc) / ls) ** 2)
    if not include_linear:
        return rbf
    return np.concatenate(
        [
            np.ones((x_arr.shape[0], 1), dtype=float),
            xx,
            rbf,
        ],
        axis=1,
    )


@dataclass
class JointEnKFConfig:
    n_ensemble: int = 512
    theta_lo: float = 0.0
    theta_hi: float = 3.0
    sigma_obs: float = 0.2
    theta_init_mean: float = 1.5
    theta_init_sd: float = 0.65
    beta_init_sd: float = 0.25
    theta_rw_sd: float = 0.035
    beta_rw_sd: float = 0.015
    beta_damping: float = 0.995
    num_basis: int = 12
    basis_lengthscale: float = 0.16
    covariance_inflation: float = 1.02
    seed: int = 0


class JointEnKF1D:
    """
    Joint EnKF baseline over calibration and discrepancy state.

    State per ensemble member:
        z = [theta, beta_0, ..., beta_{q-1}]

    Observation model:
        y(x) = y_s(x, theta) + b(x)^T beta + eps

    The EnKF update is applied to the full joint state, so discrepancy
    coefficients and calibration parameter are corrected together by each batch.
    """

    def __init__(
        self,
        sim_func_np: Callable[[np.ndarray, np.ndarray], np.ndarray],
        cfg: JointEnKFConfig,
    ):
        self.sim_func_np = sim_func_np
        self.cfg = cfg
        self.rng = np.random.RandomState(int(cfg.seed))
        self.centers = np.linspace(0.0, 1.0, int(cfg.num_basis), dtype=float)
        q = self.basis_dim
        theta = self.rng.normal(float(cfg.theta_init_mean), float(cfg.theta_init_sd), size=(cfg.n_ensemble, 1))
        theta = np.clip(theta, float(cfg.theta_lo), float(cfg.theta_hi))
        beta = self.rng.normal(0.0, float(cfg.beta_init_sd), size=(cfg.n_ensemble, q))
        self.state = np.concatenate([theta, beta], axis=1).astype(float)

    @property
    def basis_dim(self) -> int:
        return int(self.cfg.num_basis) + 2

    def _basis(self, x: np.ndarray) -> np.ndarray:
        return rbf_basis_1d(
            x,
            centers=self.centers,
            lengthscale=float(self.cfg.basis_lengthscale),
            include_linear=True,
        )

    def _forecast(self) -> None:
        n = self.state.shape[0]
        self.state[:, 0] += self.rng.normal(0.0, float(self.cfg.theta_rw_sd), size=n)
        self.state[:, 0] = np.clip(self.state[:, 0], float(self.cfg.theta_lo), float(self.cfg.theta_hi))
        self.state[:, 1:] *= float(self.cfg.beta_damping)
        self.state[:, 1:] += self.rng.normal(0.0, float(self.cfg.beta_rw_sd), size=self.state[:, 1:].shape)

    def _predict_ensemble(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim == 1:
            x_arr = x_arr[:, None]
        n, b = self.state.shape[0], x_arr.shape[0]
        theta = self.state[:, [0]]
        x_rep = np.repeat(x_arr[None, :, :], n, axis=0).reshape(n * b, x_arr.shape[1])
        theta_rep = np.repeat(theta, b, axis=0)
        sim = self.sim_func_np(x_rep, theta_rep).reshape(n, b)
        basis = self._basis(x_arr)
        disc = self.state[:, 1:] @ basis.T
        return sim + disc

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        y_ens = self._predict_ensemble(x)
        mu = y_ens.mean(axis=0)
        var = y_ens.var(axis=0, ddof=1) + float(self.cfg.sigma_obs) ** 2
        return mu, np.maximum(var, 1e-10)

    def update_batch(self, x: np.ndarray, y: np.ndarray) -> None:
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim == 1:
            x_arr = x_arr[:, None]
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        self._forecast()

        y_ens = self._predict_ensemble(x_arr)
        n = self.state.shape[0]
        z_mean = self.state.mean(axis=0, keepdims=True)
        y_mean = y_ens.mean(axis=0, keepdims=True)
        z_anom = (self.state - z_mean) * float(self.cfg.covariance_inflation)
        y_anom = y_ens - y_mean

        denom = max(n - 1, 1)
        c_zy = (z_anom.T @ y_anom) / denom
        c_yy = (y_anom.T @ y_anom) / denom
        c_yy += (float(self.cfg.sigma_obs) ** 2 + 1e-8) * np.eye(y_arr.shape[0], dtype=float)

        try:
            gain_t = np.linalg.solve(c_yy, c_zy.T)
        except np.linalg.LinAlgError:
            gain_t = np.linalg.solve(c_yy + 1e-5 * np.eye(c_yy.shape[0]), c_zy.T)
        gain = gain_t.T

        obs_pert = y_arr[None, :] + self.rng.normal(0.0, float(self.cfg.sigma_obs), size=y_ens.shape)
        innovation = obs_pert - y_ens
        self.state += innovation @ gain.T
        self.state[:, 0] = np.clip(self.state[:, 0], float(self.cfg.theta_lo), float(self.cfg.theta_hi))

    def mean_theta(self) -> float:
        return float(np.mean(self.state[:, 0]))

    def var_theta(self) -> float:
        return float(np.var(self.state[:, 0], ddof=1))


class JointEnKFGeneric:
    """
    Joint EnKF baseline for arbitrary x dimension.

    The state remains joint:
        z = [theta, beta_0, ..., beta_{q-1}]

    with observation law:
        y(x) = y_s(x, theta) + b(x)^T beta + eps.
    """

    def __init__(
        self,
        sim_func_np: Callable[[np.ndarray, np.ndarray], np.ndarray],
        cfg: JointEnKFConfig,
        x_dim: int,
        theta_init_sampler: Callable[[int, np.random.RandomState], np.ndarray] | None = None,
    ):
        self.sim_func_np = sim_func_np
        self.cfg = cfg
        self.x_dim = int(x_dim)
        self.rng = np.random.RandomState(int(cfg.seed))
        self.rff_w = self.rng.normal(
            0.0,
            1.0 / max(float(cfg.basis_lengthscale), 1e-8),
            size=(self.x_dim, int(cfg.num_basis)),
        )
        self.rff_b = self.rng.uniform(0.0, 2.0 * np.pi, size=int(cfg.num_basis))
        q = self.basis_dim
        if theta_init_sampler is None:
            theta = self.rng.normal(float(cfg.theta_init_mean), float(cfg.theta_init_sd), size=(cfg.n_ensemble, 1))
            theta = np.clip(theta, float(cfg.theta_lo), float(cfg.theta_hi))
        else:
            theta = np.asarray(theta_init_sampler(int(cfg.n_ensemble), self.rng), dtype=float).reshape(-1, 1)
        beta = self.rng.normal(0.0, float(cfg.beta_init_sd), size=(cfg.n_ensemble, q))
        self.state = np.concatenate([theta, beta], axis=1).astype(float)

    @property
    def basis_dim(self) -> int:
        return 1 + self.x_dim + int(self.cfg.num_basis)

    def _basis(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim == 1:
            x_arr = x_arr[:, None]
        rff = np.sqrt(2.0 / max(int(self.cfg.num_basis), 1)) * np.cos(x_arr @ self.rff_w + self.rff_b)
        return np.concatenate(
            [
                np.ones((x_arr.shape[0], 1), dtype=float),
                x_arr,
                rff,
            ],
            axis=1,
        )

    def _forecast(self) -> None:
        n = self.state.shape[0]
        self.state[:, 0] += self.rng.normal(0.0, float(self.cfg.theta_rw_sd), size=n)
        self.state[:, 0] = np.clip(self.state[:, 0], float(self.cfg.theta_lo), float(self.cfg.theta_hi))
        self.state[:, 1:] *= float(self.cfg.beta_damping)
        self.state[:, 1:] += self.rng.normal(0.0, float(self.cfg.beta_rw_sd), size=self.state[:, 1:].shape)

    def _predict_ensemble(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim == 1:
            x_arr = x_arr[:, None]
        n, b = self.state.shape[0], x_arr.shape[0]
        theta = self.state[:, [0]]
        x_rep = np.repeat(x_arr[None, :, :], n, axis=0).reshape(n * b, x_arr.shape[1])
        theta_rep = np.repeat(theta, b, axis=0)
        sim = self.sim_func_np(x_rep, theta_rep).reshape(n, b)
        basis = self._basis(x_arr)
        disc = self.state[:, 1:] @ basis.T
        return sim + disc

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        y_ens = self._predict_ensemble(x)
        mu = y_ens.mean(axis=0)
        var = y_ens.var(axis=0, ddof=1) + float(self.cfg.sigma_obs) ** 2
        return mu, np.maximum(var, 1e-10)

    def update_batch(self, x: np.ndarray, y: np.ndarray) -> None:
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim == 1:
            x_arr = x_arr[:, None]
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        self._forecast()

        y_ens = self._predict_ensemble(x_arr)
        n = self.state.shape[0]
        z_mean = self.state.mean(axis=0, keepdims=True)
        y_mean = y_ens.mean(axis=0, keepdims=True)
        z_anom = (self.state - z_mean) * float(self.cfg.covariance_inflation)
        y_anom = y_ens - y_mean

        denom = max(n - 1, 1)
        c_zy = (z_anom.T @ y_anom) / denom
        c_yy = (y_anom.T @ y_anom) / denom
        c_yy += (float(self.cfg.sigma_obs) ** 2 + 1e-8) * np.eye(y_arr.shape[0], dtype=float)

        try:
            gain_t = np.linalg.solve(c_yy, c_zy.T)
        except np.linalg.LinAlgError:
            gain_t = np.linalg.solve(c_yy + 1e-5 * np.eye(c_yy.shape[0]), c_zy.T)
        self.state += (y_arr[None, :] + self.rng.normal(0.0, float(self.cfg.sigma_obs), size=y_ens.shape) - y_ens) @ gain_t
        self.state[:, 0] = np.clip(self.state[:, 0], float(self.cfg.theta_lo), float(self.cfg.theta_hi))

    def mean_theta(self) -> float:
        return float(np.mean(self.state[:, 0]))

    def var_theta(self) -> float:
        return float(np.var(self.state[:, 0], ddof=1))
