from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist

from .configs import CalibrationConfig
from .emulator import DeterministicSimulator
from .online_calibrator import OnlineBayesCalibrator, crps_gaussian
from .paper_pf_digital_twin import WardPaperPFVectorConfig, WardPaperParticleFilterVector
from .method_names import method_aliases, paper_method_name


PAPER_LABELS = {
    "DA": "DA",
    "BRPC-P": "BRPC-P",
    "BRPC-E": "BRPC-E",
    "BRPC-F": "BRPC-F",
    "B-BRPC-F": "B-BRPC-F",
    "C-BRPC-F": "C-BRPC-F",
    "B-BRPC-E": "B-BRPC-E",
    "C-BRPC-E": "C-BRPC-E",
    "B-BRPC-P": "B-BRPC-P",
    "C-BRPC-P": "C-BRPC-P",
    "B-BRPC-RRA": "B-BRPC-RRA",
    "BC": "BC",
}


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(-1)
    return v / max(float(np.linalg.norm(v)), 1e-12)


V1 = _unit(np.array([1.0, -0.5, 0.7, -0.3, 0.4], dtype=float))
V2 = _unit(np.array([0.2, 0.8, -0.4, 0.5, -0.1], dtype=float))
THETA0 = np.array([1.0, -0.8, 0.6, 0.9, -0.5], dtype=float)


def phi_features_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    feats = []
    for j in range(5):
        feats.append(
            np.sin(2.0 * np.pi * x[:, j])
            + 0.5 * np.cos(2.0 * np.pi * x[:, j + 5])
            + 0.25 * np.sin(2.0 * np.pi * (x[:, j] + x[:, j + 10]))
        )
    return np.stack(feats, axis=1)


def phi_features_torch(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 1:
        x = x[None, :]
    feats = []
    for j in range(5):
        feats.append(
            torch.sin(2.0 * torch.pi * x[:, j])
            + 0.5 * torch.cos(2.0 * torch.pi * x[:, j + 5])
            + 0.25 * torch.sin(2.0 * torch.pi * (x[:, j] + x[:, j + 10]))
        )
    return torch.stack(feats, dim=1)


def g_features_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    feats = []
    for j in range(5):
        feats.append(
            np.sin(2.0 * np.pi * x[:, j] + 0.4 * x[:, j + 5])
            + 0.4 * np.cos(2.0 * np.pi * x[:, j + 10])
            + 0.3 * x[:, j + 15] * x[:, j]
        )
    return np.stack(feats, axis=1)


def additive_ridge_simulator_np(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    theta = np.asarray(theta, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    if theta.ndim == 1:
        theta = theta[None, :]
    phi = phi_features_np(x)
    if theta.shape[0] == 1:
        theta = np.repeat(theta, x.shape[0], axis=0)
    return np.sum(phi * theta[:, :5], axis=1)


def additive_ridge_simulator_torch(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    if x.dim() == 1:
        x = x[None, :]
    if theta.dim() == 1:
        theta = theta[None, :]
    phi = phi_features_torch(x)
    if theta.shape[0] == 1:
        theta = theta.expand(x.shape[0], -1)
    return torch.sum(phi * theta[:, :5], dim=1, keepdim=True)


class OrthogonalizedRFFDiscrepancy:
    def __init__(
        self,
        x_dim: int,
        theta_dim: int,
        num_features: int,
        amplitude: float,
        rff_lengthscale: float,
        ref_size: int,
        ridge: float,
        seed: int,
    ):
        self.x_dim = int(x_dim)
        self.theta_dim = int(theta_dim)
        self.num_features = int(num_features)
        self.amplitude = float(amplitude)
        self.rff_lengthscale = float(rff_lengthscale)
        self.ref_size = int(ref_size)
        self.ridge = float(ridge)
        rng = np.random.RandomState(int(seed))
        self.omega = rng.normal(
            loc=0.0,
            scale=1.0 / max(self.rff_lengthscale, 1e-8),
            size=(self.num_features, self.x_dim),
        )
        self.phase = rng.uniform(0.0, 2.0 * np.pi, size=(self.num_features,))
        self.alpha = rng.normal(loc=0.0, scale=1.0 / math.sqrt(self.num_features), size=(self.num_features,))

        sobol = torch.quasirandom.SobolEngine(dimension=self.x_dim, scramble=True, seed=int(seed) + 17)
        x_ref = sobol.draw(self.ref_size).cpu().numpy()
        phi_ref = phi_features_np(x_ref)
        psi_raw_ref = self._raw_features_np(x_ref)
        gram = phi_ref.T @ phi_ref + self.ridge * np.eye(self.theta_dim, dtype=float)
        self.proj_coef = np.linalg.solve(gram, phi_ref.T @ psi_raw_ref)
        psi_ref = psi_raw_ref - phi_ref @ self.proj_coef
        proj_energy = np.linalg.norm(phi_ref.T @ psi_ref, ord="fro")
        ref_energy = max(np.linalg.norm(phi_ref, ord="fro") * np.linalg.norm(psi_ref, ord="fro"), 1e-12)
        self.orthog_ratio = float(proj_energy / ref_energy)

    def _raw_features_np(self, x: np.ndarray) -> np.ndarray:
        return math.sqrt(2.0) * np.cos(x @ self.omega.T + self.phase[None, :])

    def eval_np(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x[None, :]
        psi_raw = self._raw_features_np(x)
        phi = phi_features_np(x)
        psi = psi_raw - phi @ self.proj_coef
        return self.amplitude * (psi @ self.alpha)


class LocalHeterogeneousRFF:
    def __init__(
        self,
        x_dim: int,
        num_features: int,
        rff_lengthscale: float,
        local_center: Sequence[float],
        local_scale: float,
        seed: int,
    ):
        self.x_dim = int(x_dim)
        self.num_features = int(num_features)
        self.local_scale = float(local_scale)
        center = np.asarray(local_center, dtype=float).reshape(-1)
        if center.size != 3:
            raise ValueError("local_center must have length 3")
        self.local_center = center
        rng = np.random.RandomState(int(seed))
        self.omega = rng.normal(
            loc=0.0,
            scale=1.0 / max(float(rff_lengthscale), 1e-8),
            size=(self.num_features, self.x_dim),
        )
        self.phase = rng.uniform(0.0, 2.0 * np.pi, size=(self.num_features,))
        self.alpha = rng.normal(loc=0.0, scale=1.0 / math.sqrt(self.num_features), size=(self.num_features,))

    def eval_np(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x[None, :]
        raw = math.sqrt(2.0) * np.cos(x @ self.omega.T + self.phase[None, :]) @ self.alpha
        dist_sq = np.sum((x[:, :3] - self.local_center[None, :]) ** 2, axis=1)
        local_weight = 1.0 + 1.5 * np.exp(-dist_sq / max(2.0 * self.local_scale ** 2, 1e-12))
        return local_weight * raw


class L2ProjectedPhysicalResponse:
    def __init__(
        self,
        x_dim: int,
        theta_dim: int,
        total_batches: int,
        scenario: str,
        seed: int,
        ref_size: int,
        projection_ridge: float,
        num_rff: int,
        rff_lengthscale: float,
        h_amp: float,
        local_scale: float,
        sudden_jump_scale: float = 0.8,
        mixed_jump_scale: float = 0.6,
    ):
        self.x_dim = int(x_dim)
        self.theta_dim = int(theta_dim)
        self.h_amp = float(h_amp)
        self.a_batches, self.cp_batches = build_theta_path(
            str(scenario),
            int(total_batches),
            int(seed),
            sudden_jump_scale=float(sudden_jump_scale),
            mixed_jump_scale=float(mixed_jump_scale),
        )
        self.h = LocalHeterogeneousRFF(
            x_dim=self.x_dim,
            num_features=int(num_rff),
            rff_lengthscale=float(rff_lengthscale),
            local_center=(0.25, 0.60, 0.40),
            local_scale=float(local_scale),
            seed=int(seed) + 404,
        )
        sobol = torch.quasirandom.SobolEngine(dimension=self.x_dim, scramble=True, seed=int(seed) + 505)
        x_ref = sobol.draw(int(ref_size)).cpu().numpy()
        phi_ref = phi_features_np(x_ref)
        g_ref = g_features_np(x_ref)
        h_ref = self.h.eval_np(x_ref)
        gram = phi_ref.T @ phi_ref + float(projection_ridge) * np.eye(self.theta_dim, dtype=float)
        self.theta_from_a = np.linalg.solve(gram, phi_ref.T @ g_ref)
        self.theta_h = np.linalg.solve(gram, phi_ref.T @ (self.h_amp * h_ref))
        self.theta_dagger_batches = self.a_batches @ self.theta_from_a.T + self.theta_h[None, :]
        self.projection_ref_size = int(ref_size)

    def zeta_np(self, x: np.ndarray, batch_idx: int) -> np.ndarray:
        a_t = self.a_batches[int(batch_idx)]
        return g_features_np(x) @ a_t + self.h_amp * self.h.eval_np(x)


def _random_unit(dim: int, rng: np.random.RandomState) -> np.ndarray:
    return _unit(rng.normal(size=(dim,)))


def build_theta_path(
    scenario: str,
    total_batches: int,
    seed: int,
    sudden_jump_scale: float = 0.8,
    mixed_jump_scale: float = 0.6,
) -> Tuple[np.ndarray, List[int]]:
    rng = np.random.RandomState(int(seed))
    B = int(total_batches)
    t = np.arange(B, dtype=float)
    theta = np.zeros((B, 5), dtype=float)
    cp_batches: List[int] = []

    if scenario == "slope":
        for b in range(B):
            theta[b] = THETA0 + 0.8 * (b / max(B - 1, 1)) * V1 + 0.15 * math.sin(2.0 * math.pi * b / max(B, 1)) * V2
        return theta, cp_batches

    if scenario == "sudden":
        cp_batches = [max(1, B // 4), max(2, B // 2), max(3, (3 * B) // 4)]
        jumps = [_random_unit(5, rng) * float(sudden_jump_scale) for _ in cp_batches]
        cur = THETA0.copy()
        for b in range(B):
            if b in cp_batches:
                cur = cur + jumps[cp_batches.index(b)]
            theta[b] = cur
        return theta, cp_batches

    if scenario == "mixed":
        cp_batches = [max(2, int(round(0.35 * B))), max(3, int(round(0.70 * B)))]
        jumps = [_random_unit(5, rng) * float(mixed_jump_scale) for _ in cp_batches]
        for b in range(B):
            theta[b] = THETA0 + 0.4 * (b / max(B - 1, 1)) * V1 + 0.10 * math.sin(2.0 * math.pi * b / max(B, 1)) * V2
        for cpb, jump in zip(cp_batches, jumps):
            theta[cpb:] += jump[None, :]
        return theta, cp_batches

    raise ValueError(f"Unknown scenario: {scenario}")


@dataclass
class HighDimDiagSpec:
    x_dim: int = 20
    theta_dim: int = 5
    total_batches: int = 60
    batch_size: int = 64
    noise_sd: float = 0.05
    num_rff: int = 10
    discrepancy_amp: float = 0.3
    rff_lengthscale: float = 0.6
    orthog_ref_size: int = 1024
    orthog_ridge: float = 1e-6
    data_mode: str = "orthogonalized_rff"
    projection_ref_size: int = 50000
    projection_ridge: float = 1e-6
    physical_h_amp: float = 0.4
    physical_local_scale: float = 0.12
    sudden_jump_scale: float = 0.8
    mixed_jump_scale: float = 0.6
    bc_window_batches: float = 2.5


class HighDimProjectedDataStream:
    def __init__(self, scenario: str, spec: HighDimDiagSpec, seed: int):
        self.scenario = str(scenario)
        self.spec = spec
        self.seed = int(seed)
        self.total_batches = int(spec.total_batches)
        self.batch_size = int(spec.batch_size)
        self.noise_sd = float(spec.noise_sd)
        self.data_mode = str(spec.data_mode)
        self.discrepancy = None
        self.physical = None
        if self.data_mode == "orthogonalized_rff":
            self.theta_batches, self.cp_batches = build_theta_path(
                self.scenario,
                self.total_batches,
                self.seed,
                sudden_jump_scale=spec.sudden_jump_scale,
                mixed_jump_scale=spec.mixed_jump_scale,
            )
            self.discrepancy = OrthogonalizedRFFDiscrepancy(
                x_dim=spec.x_dim,
                theta_dim=spec.theta_dim,
                num_features=spec.num_rff,
                amplitude=spec.discrepancy_amp,
                rff_lengthscale=spec.rff_lengthscale,
                ref_size=spec.orthog_ref_size,
                ridge=spec.orthog_ridge,
                seed=self.seed + 101,
            )
        elif self.data_mode == "physical_projected":
            self.physical = L2ProjectedPhysicalResponse(
                x_dim=spec.x_dim,
                theta_dim=spec.theta_dim,
                total_batches=self.total_batches,
                scenario=self.scenario,
                seed=self.seed,
                ref_size=spec.projection_ref_size,
                projection_ridge=spec.projection_ridge,
                num_rff=spec.num_rff,
                rff_lengthscale=spec.rff_lengthscale,
                h_amp=spec.physical_h_amp,
                local_scale=spec.physical_local_scale,
                sudden_jump_scale=spec.sudden_jump_scale,
                mixed_jump_scale=spec.mixed_jump_scale,
            )
            self.theta_batches = self.physical.theta_dagger_batches
            self.cp_batches = self.physical.cp_batches
        else:
            raise ValueError(f"Unknown highdim data_mode: {self.data_mode}")
        self.cp_times = [int(b * self.batch_size) for b in self.cp_batches]
        self.sobol = torch.quasirandom.SobolEngine(dimension=spec.x_dim, scramble=True, seed=self.seed + 202)
        self.rng = np.random.RandomState(self.seed + 303)
        self.batch_idx = 0
        self.theta_true_hist: List[np.ndarray] = []
        self.y_noiseless_hist: List[np.ndarray] = []

    def noiseless_np(self, x_np: np.ndarray, batch_idx: int) -> np.ndarray:
        if self.data_mode == "orthogonalized_rff":
            if self.discrepancy is None:
                raise RuntimeError("orthogonalized_rff discrepancy is not initialized")
            return additive_ridge_simulator_np(x_np, self.theta_batches[int(batch_idx)]) + self.discrepancy.eval_np(x_np)
        if self.physical is None:
            raise RuntimeError("physical_projected response is not initialized")
        return self.physical.zeta_np(x_np, int(batch_idx))

    def next(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.batch_idx >= self.total_batches:
            raise StopIteration
        x_np = self.sobol.draw(self.batch_size).cpu().numpy()
        theta_true = self.theta_batches[self.batch_idx]
        y0 = self.noiseless_np(x_np, self.batch_idx)
        y = y0 + self.noise_sd * self.rng.randn(self.batch_size)
        self.theta_true_hist.append(theta_true.copy())
        self.y_noiseless_hist.append(np.asarray(y0, dtype=float).reshape(-1).copy())
        self.batch_idx += 1
        return (
            torch.tensor(x_np, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )


class KOHSlidingWindowVector:
    def __init__(
        self,
        theta_dim: int,
        window_points: int = 320,
        sigma_obs: float = 0.05,
        gp_lengthscale: float = 1.5,
        gp_signal_var: float = 0.09,
        theta_lo: float = -2.0,
        theta_hi: float = 2.0,
        ridge: float = 1e-6,
    ):
        self.theta_dim = int(theta_dim)
        self.W = int(window_points)
        self.sigma2 = float(sigma_obs) ** 2
        self.ls = float(gp_lengthscale)
        self.sv = float(gp_signal_var)
        self.theta_lo = float(theta_lo)
        self.theta_hi = float(theta_hi)
        self.ridge = float(ridge)
        self.current_theta = np.zeros(self.theta_dim, dtype=float)
        self.X_buf: List[np.ndarray] = []
        self.Y_buf: List[np.ndarray] = []
        self._gp_L = None
        self._gp_alpha = None
        self._gp_X = None

    def _kernel(self, Xa: np.ndarray, Xb: np.ndarray) -> np.ndarray:
        dist_sq = cdist(Xa, Xb, metric="sqeuclidean")
        return self.sv * np.exp(-0.5 * dist_sq / max(self.ls ** 2, 1e-12))

    def _trim(self) -> Tuple[np.ndarray, np.ndarray]:
        X_all = np.concatenate(self.X_buf, axis=0)
        Y_all = np.concatenate(self.Y_buf, axis=0)
        if len(X_all) > self.W:
            X_all = X_all[-self.W:]
            Y_all = Y_all[-self.W:]
            self.X_buf, self.Y_buf = [X_all], [Y_all]
        return X_all, Y_all

    def _fit(self) -> None:
        X_all, Y_all = self._trim()
        n = len(X_all)
        if n < max(8, self.theta_dim + 2):
            self._gp_L = None
            self._gp_alpha = None
            self._gp_X = None
            return
        Phi = phi_features_np(X_all)
        K = self._kernel(X_all, X_all) + (self.sigma2 + 1e-6) * np.eye(n, dtype=float)
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            K = K + 1e-4 * np.eye(n, dtype=float)
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                self._gp_L = None
                self._gp_alpha = None
                self._gp_X = None
                return
        Kinv_Phi = np.linalg.solve(L.T, np.linalg.solve(L, Phi))
        Kinv_Y = np.linalg.solve(L.T, np.linalg.solve(L, Y_all))
        A = Phi.T @ Kinv_Phi + self.ridge * np.eye(self.theta_dim, dtype=float)
        b = Phi.T @ Kinv_Y
        theta_hat = np.linalg.solve(A, b)
        theta_hat = np.clip(theta_hat, self.theta_lo, self.theta_hi)
        resid = Y_all - Phi @ theta_hat
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, resid))
        self.current_theta = theta_hat
        self._gp_L = L
        self._gp_alpha = alpha
        self._gp_X = X_all

    def update_batch(self, Xb_np: np.ndarray, Yb_np: np.ndarray) -> None:
        self.X_buf.append(np.asarray(Xb_np, dtype=float).copy())
        self.Y_buf.append(np.asarray(Yb_np, dtype=float).reshape(-1).copy())
        self._fit()

    def predict(self, X_new: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X_new = np.asarray(X_new, dtype=float)
        if X_new.ndim == 1:
            X_new = X_new[None, :]
        phi_new = phi_features_np(X_new)
        sim_pred = phi_new @ self.current_theta
        if self._gp_L is None or self._gp_X is None or self._gp_alpha is None:
            return sim_pred, np.full(len(X_new), self.sigma2, dtype=float)
        k_star = self._kernel(X_new, self._gp_X)
        gp_mu = k_star @ self._gp_alpha
        v = np.linalg.solve(self._gp_L, k_star.T)
        gp_var = np.maximum(self.sv + self.sigma2 - np.sum(v ** 2, axis=0), 1e-8)
        return sim_pred + gp_mu, gp_var


class HighDimMovePF:
    def __init__(
        self,
        theta_dim: int,
        num_particles: int,
        theta_lo: float = -2.0,
        theta_hi: float = 2.0,
        sigma_like: float = 0.30,
        move_theta_std: float = 0.15,
        seed: int = 0,
    ):
        self.theta_dim = int(theta_dim)
        self.num_particles = int(num_particles)
        self.theta_lo = float(theta_lo)
        self.theta_hi = float(theta_hi)
        self.sigma_like = float(sigma_like)
        self.move_theta_std = float(move_theta_std)
        self.rng = np.random.RandomState(int(seed))
        self.theta = self.rng.uniform(
            low=self.theta_lo,
            high=self.theta_hi,
            size=(self.num_particles, self.theta_dim),
        ).astype(float)
        self.weights = np.full(self.num_particles, 1.0 / self.num_particles, dtype=float)

    def _systematic_resample(self, weights: np.ndarray) -> np.ndarray:
        N = int(weights.shape[0])
        positions = (self.rng.rand() + np.arange(N)) / N
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

    def predict_batch(self, X_batch_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X_np = np.asarray(X_batch_np, dtype=float)
        pred_particles = additive_ridge_simulator_np(
            np.repeat(X_np[None, :, :], self.num_particles, axis=0).reshape(-1, X_np.shape[1]),
            np.repeat(self.theta, X_np.shape[0], axis=0),
        ).reshape(self.num_particles, X_np.shape[0])
        mu = self.weights @ pred_particles
        second = self.weights @ (pred_particles ** 2)
        var = np.maximum(second - mu ** 2 + self.sigma_like ** 2, 1e-8)
        return mu, var

    def update_batch(self, X_batch_np: np.ndarray, Y_batch_np: np.ndarray) -> None:
        X_np = np.asarray(X_batch_np, dtype=float)
        y_np = np.asarray(Y_batch_np, dtype=float).reshape(-1)
        pred_particles = additive_ridge_simulator_np(
            np.repeat(X_np[None, :, :], self.num_particles, axis=0).reshape(-1, X_np.shape[1]),
            np.repeat(self.theta, X_np.shape[0], axis=0),
        ).reshape(self.num_particles, X_np.shape[0])
        resid = y_np[None, :] - pred_particles
        loglik = -0.5 * np.sum((resid / self.sigma_like) ** 2, axis=1)
        logw = np.log(np.clip(self.weights, 1e-300, None)) + loglik
        logw = logw - np.max(logw)
        w = np.exp(logw)
        w_sum = np.sum(w)
        if not np.isfinite(w_sum) or w_sum <= 0:
            self.weights[:] = 1.0 / self.num_particles
        else:
            self.weights = w / w_sum
        idx = self._systematic_resample(self.weights)
        self.theta = self.theta[idx]
        if self.move_theta_std > 0.0:
            self.theta = self.theta + self.rng.normal(
                loc=0.0,
                scale=self.move_theta_std,
                size=self.theta.shape,
            )
        self.theta = np.clip(self.theta, self.theta_lo, self.theta_hi)
        self.weights[:] = 1.0 / self.num_particles

    def mean_theta(self) -> np.ndarray:
        return np.asarray(self.theta.mean(axis=0), dtype=float)

    def var_theta_diag(self) -> np.ndarray:
        return np.asarray(np.var(self.theta, axis=0), dtype=float)


def make_bocpd_cfg(
    spec: HighDimDiagSpec,
    num_particles: int,
    delta_bpc_lambda: float,
    num_support: int,
    delta_mode: str,
    controller: str,
    hazard_lambda: float = 200.0,
    restart_margin: float = 1.0,
    restart_cooldown: int = 10,
    wcusum_window: int = 4,
    wcusum_threshold: float = 0.25,
    wcusum_kappa: float = 0.25,
    wcusum_sigma_floor: float = 0.25,
) -> CalibrationConfig:
    cfg = CalibrationConfig()
    cfg.model.device = "cpu"
    cfg.model.dtype = torch.float64
    cfg.model.rho = 1.0
    cfg.model.sigma_eps = float(spec.noise_sd)
    cfg.model.use_discrepancy = False
    cfg.model.bocpd_use_discrepancy = True
    cfg.model.delta_update_mode = str(delta_mode)
    cfg.model.delta_bpc_lambda = float(delta_bpc_lambda)
    cfg.model.delta_bpc_obs_noise_mode = "sigma_eps"
    cfg.model.delta_bpc_predict_add_kernel_noise = False
    cfg.model.delta_online_init_max_iter = 30
    cfg.model.delta_inducing_num_points = int(num_support)
    cfg.model.delta_kernel.lengthscale = 0.75
    cfg.model.delta_kernel.variance = 0.10
    cfg.model.delta_kernel.noise = 1e-6
    cfg.pf.num_particles = int(num_particles)
    cfg.pf.random_walk_scale = 0.08
    if controller == "wCUSUM":
        cfg.bocpd.bocpd_mode = "wcusum"
        cfg.bocpd.controller_name = "wcusum"
        cfg.bocpd.controller_stat = "log_surprise_mean"
        cfg.bocpd.controller_wcusum_warmup_batches = 3
        cfg.bocpd.controller_wcusum_window = int(wcusum_window)
        cfg.bocpd.controller_wcusum_threshold = float(wcusum_threshold)
        cfg.bocpd.controller_wcusum_kappa = float(wcusum_kappa)
        cfg.bocpd.controller_wcusum_sigma_floor = float(wcusum_sigma_floor)
    elif controller == "none":
        cfg.bocpd.bocpd_mode = "single_segment"
        cfg.bocpd.controller_name = "none"
        cfg.bocpd.use_restart = False
    elif controller == "standard":
        cfg.bocpd.bocpd_mode = "standard"
        cfg.bocpd.use_restart = False
        cfg.bocpd.hazard_lambda = float(hazard_lambda)
    else:
        cfg.bocpd.bocpd_mode = "restart"
        cfg.bocpd.use_restart = True
        cfg.bocpd.restart_impl = "rolled_cusum_260324"
        cfg.bocpd.hazard_lambda = float(hazard_lambda)
        cfg.bocpd.restart_margin = float(restart_margin)
        cfg.bocpd.restart_cooldown = int(restart_cooldown)
    return cfg


def prior_sampler_factory(theta_dim: int, lo: float = -2.0, hi: float = 2.0):
    def prior_sampler(N: int) -> torch.Tensor:
        return lo + (hi - lo) * torch.rand(N, theta_dim, dtype=torch.float64)
    return prior_sampler


def _finite_mean(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def _theta_crps_diag_mean(theta_mean: np.ndarray, theta_var_diag: np.ndarray, theta_true: np.ndarray) -> float:
    mu = torch.tensor(theta_mean, dtype=torch.float64)
    var = torch.tensor(np.clip(theta_var_diag, 1e-12, None), dtype=torch.float64)
    y = torch.tensor(theta_true, dtype=torch.float64)
    return float(crps_gaussian(mu, var, y).mean().item())


def _match_events_forward(gt: Sequence[int], det: Sequence[int], tol: int = 2) -> Dict[str, float]:
    gt = [int(v) for v in gt]
    det = [int(v) for v in det]
    used = [False] * len(det)
    tp = 0
    delays: List[int] = []
    for cp in gt:
        found = None
        for idx, dd in enumerate(det):
            if used[idx]:
                continue
            if cp <= dd <= cp + tol:
                found = idx
                break
        if found is not None:
            used[found] = True
            tp += 1
            delays.append(det[found] - cp)
    precision = float(tp / max(len(det), 1))
    recall = float(tp / max(len(gt), 1))
    f1 = float(2.0 * precision * recall / max(precision + recall, 1e-12))
    return {
        "precision_at2": precision,
        "recall_at2": recall,
        "f1_at2": f1,
        "mean_delay": _finite_mean(delays),
    }


def _save_theta_tracking(
    out_dir: Path,
    tag: str,
    theta_est: np.ndarray,
    theta_true: np.ndarray,
    theta_var_diag: np.ndarray,
    cp_batches: Sequence[int],
    restart_batches: Sequence[int],
    paper_label: str,
) -> None:
    plot_dir = out_dir / "theta_tracking_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    csv_path = plot_dir / f"{tag}_theta_tracking.csv"
    png_path = plot_dir / f"{tag}_theta_tracking.png"

    rows = []
    B, d = theta_est.shape
    cp_set = set(int(v) for v in cp_batches)
    restart_set = set(int(v) for v in restart_batches)
    for b in range(B):
        for j in range(d):
            rows.append(
                dict(
                    batch_idx=int(b),
                    theta_idx=int(j),
                    theta_est=float(theta_est[b, j]),
                    theta_true=float(theta_true[b, j]),
                    theta_var=float(theta_var_diag[b, j]),
                    is_cp_batch=int(b in cp_set),
                    did_restart=int(b in restart_set),
                )
            )
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    fig, axes = plt.subplots(theta_est.shape[1], 1, figsize=(11, 2.6 * theta_est.shape[1]), sharex=True)
    if theta_est.shape[1] == 1:
        axes = [axes]
    xaxis = np.arange(theta_est.shape[0])
    for j, ax in enumerate(axes):
        ax.plot(xaxis, theta_true[:, j], "k--", lw=2.0, label="Ground Truth")
        ax.plot(xaxis, theta_est[:, j], color="tab:blue", lw=1.8, label=paper_label)
        for cpb in cp_batches:
            ax.axvline(int(cpb), color="black", alpha=0.18, lw=1.0)
        if restart_batches:
            y_level = np.nanmax(theta_true[:, j]) + 0.05 * max(np.nanstd(theta_true[:, j]), 1.0)
            ax.scatter(restart_batches, np.full(len(restart_batches), y_level), color="tab:red", s=14, marker="x", label="Restart" if j == 0 else None)
        ax.set_ylabel(f"theta[{j}]")
    axes[0].legend(loc="best")
    axes[-1].set_xlabel("batch index")
    fig.tight_layout()
    fig.savefig(png_path, dpi=220)
    plt.close(fig)


def save_combined_theta_tracking_from_rows(out_dir: Path, rows: Sequence[Dict[str, float]]) -> None:
    if not rows:
        return
    plot_dir = out_dir / "theta_tracking_plots_combined"
    plot_dir.mkdir(parents=True, exist_ok=True)
    run_df = pd.DataFrame(rows)
    manifest_rows: List[Dict[str, object]] = []
    group_cols = ["scenario", "seed", "x_dim", "theta_dim", "batch_size", "total_batches"]
    if "data_mode" in run_df.columns:
        group_cols.append("data_mode")
    for key, grp in run_df.groupby(group_cols, dropna=False):
        loaded = []
        for _, row in grp.iterrows():
            raw_path = out_dir / str(row["raw_relpath"])
            payload = torch.load(raw_path, map_location="cpu", weights_only=False)
            loaded.append(payload)
        if not loaded:
            continue
        first = loaded[0]
        theta_true = np.asarray(first["theta_star_true"], dtype=float)
        cp_batches = [int(v) for v in first.get("cp_batches", [])]
        if len(key) == 7:
            scenario, seed, x_dim, theta_dim, batch_size, total_batches, data_mode = key
        else:
            scenario, seed, x_dim, theta_dim, batch_size, total_batches = key
            data_mode = "orthogonalized_rff"
        prefix = f"{scenario}_{data_mode}_seed{int(seed)}_dx{int(x_dim)}_dth{int(theta_dim)}_B{int(batch_size)}_Tb{int(total_batches)}"
        csv_path = plot_dir / f"{prefix}_theta_tracking.csv"
        png_path = plot_dir / f"{prefix}_theta_tracking.png"

        rows_out: List[Dict[str, object]] = []
        fig, axes = plt.subplots(int(theta_dim), 1, figsize=(12, 2.6 * int(theta_dim)), sharex=True)
        if int(theta_dim) == 1:
            axes = [axes]
        xaxis = np.arange(theta_true.shape[0])
        for j, ax in enumerate(axes):
            ax.plot(xaxis, theta_true[:, j], "k--", lw=2.0, label="Ground Truth")
            for b, val in enumerate(theta_true[:, j]):
                rows_out.append(dict(batch_idx=int(b), theta_idx=int(j), series="Ground Truth", theta=float(val)))
            for payload in loaded:
                label = str(payload.get("paper_label", payload.get("method", "method")))
                theta_est = np.asarray(payload["theta"], dtype=float)
                ax.plot(xaxis, theta_est[:, j], lw=1.8, label=label)
                for b, val in enumerate(theta_est[:, j]):
                    rows_out.append(dict(batch_idx=int(b), theta_idx=int(j), series=label, theta=float(val)))
            for cpb in cp_batches:
                ax.axvline(int(cpb), color="black", alpha=0.18, lw=1.0)
            ax.set_ylabel(f"theta[{j}]")
        handles, labels = axes[0].get_legend_handles_labels()
        axes[0].legend(handles, labels, loc="best")
        axes[-1].set_xlabel("batch index")
        fig.tight_layout()
        fig.savefig(png_path, dpi=220)
        plt.close(fig)
        pd.DataFrame(rows_out).to_csv(csv_path, index=False)
        manifest_rows.append(
            dict(
                scenario=str(scenario),
                seed=int(seed),
                x_dim=int(x_dim),
                theta_dim=int(theta_dim),
                batch_size=int(batch_size),
                total_batches=int(total_batches),
                data_mode=str(data_mode),
                plot_prefix=prefix,
                num_methods=int(len(loaded)),
            )
        )
    pd.DataFrame(manifest_rows).to_csv(plot_dir / "theta_tracking_manifest.csv", index=False)


def method_specs() -> Dict[str, Dict[str, object]]:
    specs = {
        "DA": dict(kind="paper_pf_vector", paper_label=PAPER_LABELS["DA"]),
        "DA_Analogue": dict(kind="da", paper_label="DA-analogue"),
        "BRPC-P": dict(kind="bocpd", delta_mode="online_bpc_proxy_stablemean", controller="none", paper_label=PAPER_LABELS["BRPC-P"]),
        "BRPC-E": dict(kind="bocpd", delta_mode="online_bpc_exact", controller="none", paper_label=PAPER_LABELS["BRPC-E"]),
        "BRPC-F": dict(kind="bocpd", delta_mode="online_bpc_fixedsupport_exact", controller="none", paper_label=PAPER_LABELS["BRPC-F"]),
        "B-BRPC-F": dict(kind="bocpd", delta_mode="online_bpc_fixedsupport_exact", controller="BOCPD", paper_label=PAPER_LABELS["B-BRPC-F"]),
        "C-BRPC-F": dict(kind="bocpd", delta_mode="online_bpc_fixedsupport_exact", controller="wCUSUM", paper_label=PAPER_LABELS["C-BRPC-F"]),
        "B-BRPC-E": dict(kind="bocpd", delta_mode="online_bpc_exact", controller="BOCPD", paper_label=PAPER_LABELS["B-BRPC-E"]),
        "C-BRPC-E": dict(kind="bocpd", delta_mode="online_bpc_exact", controller="wCUSUM", paper_label=PAPER_LABELS["C-BRPC-E"]),
        "B-BRPC-P": dict(kind="bocpd", delta_mode="online_bpc_proxy_stablemean", controller="BOCPD", paper_label=PAPER_LABELS["B-BRPC-P"]),
        "C-BRPC-P": dict(kind="bocpd", delta_mode="online_bpc_proxy_stablemean", controller="wCUSUM", paper_label=PAPER_LABELS["C-BRPC-P"]),
        "B-BRPC-RRA": dict(kind="bocpd", delta_mode="refit", controller="BOCPD", paper_label=PAPER_LABELS["B-BRPC-RRA"]),
        "BC": dict(kind="bc", paper_label=PAPER_LABELS["BC"]),
    }
    for legacy, paper in method_aliases().items():
        if paper in specs and legacy not in specs:
            specs[legacy] = dict(specs[paper], alias_for=paper)
    return specs


def run_one_method(
    scenario: str,
    seed: int,
    out_dir: Path,
    spec: HighDimDiagSpec,
    num_particles: int,
    delta_bpc_lambda: float,
    num_support: int,
    method_name: str,
    run_name: str | None = None,
    controller_overrides: Dict[str, float] | None = None,
) -> Dict[str, float]:
    spec_map = method_specs()
    if method_name not in spec_map:
        raise ValueError(f"Unknown method: {method_name}")
    mcfg = spec_map[method_name]
    paper_label = str(mcfg["paper_label"])
    run_name = str(run_name or method_name)
    controller_overrides = dict(controller_overrides or {})
    stream = HighDimProjectedDataStream(scenario=scenario, spec=spec, seed=seed)

    theta_hist: List[np.ndarray] = []
    theta_true_hist: List[np.ndarray] = []
    theta_var_diag_hist: List[np.ndarray] = []
    restart_mode_hist: List[str] = []
    y_rmse_hist: List[float] = []
    y_crps_hist: List[float] = []
    theta_crps_hist: List[float] = []
    X_batches: List[np.ndarray] = []
    Y_batches: List[np.ndarray] = []
    y_noiseless_batches: List[np.ndarray] = []
    pred_mu_batches: List[np.ndarray] = []
    pred_var_batches: List[np.ndarray] = []

    t0 = time()
    if mcfg["kind"] == "bocpd":
        controller_name = str(controller_overrides.get("controller_mode", mcfg["controller"]))
        cfg = make_bocpd_cfg(
            spec,
            num_particles=num_particles,
            delta_bpc_lambda=delta_bpc_lambda,
            num_support=num_support,
            delta_mode=str(mcfg["delta_mode"]),
            controller=controller_name,
            hazard_lambda=float(controller_overrides.get("hazard_lambda", 200.0)),
            restart_margin=float(controller_overrides.get("restart_margin", 1.0)),
            restart_cooldown=int(controller_overrides.get("restart_cooldown", 10)),
            wcusum_window=int(controller_overrides.get("wcusum_window", 4)),
            wcusum_threshold=float(controller_overrides.get("wcusum_threshold", 0.25)),
            wcusum_kappa=float(controller_overrides.get("wcusum_kappa", 0.25)),
            wcusum_sigma_floor=float(controller_overrides.get("wcusum_sigma_floor", 0.25)),
        )
        prior_sampler = prior_sampler_factory(spec.theta_dim)
        emulator = DeterministicSimulator(func=additive_ridge_simulator_torch, enable_autograd=True)
        calib = OnlineBayesCalibrator(cfg, emulator, prior_sampler)

        for batch_idx in range(spec.total_batches):
            Xb, Yb = stream.next()
            X_np = Xb.detach().cpu().numpy().copy()
            theta_true = stream.theta_true_hist[-1]
            y0_np = stream.y_noiseless_hist[-1]
            pred_mu_batch = None
            pred_var_batch = None

            if batch_idx > 0:
                pred = calib.predict_batch(Xb)
                pred_mu_batch = pred["mu"].detach().cpu()
                pred_var_batch = pred["var"].detach().cpu()
                y_rmse_hist.append(float(torch.sqrt(((pred["mu"] - Yb) ** 2).mean()).item()))
                y_crps_hist.append(float(crps_gaussian(pred["mu"], pred["var"], Yb).mean().item()))

            X_batches.append(X_np)
            Y_batches.append(Yb.detach().cpu().numpy().reshape(-1).copy())
            y_noiseless_batches.append(np.asarray(y0_np, dtype=float).reshape(-1).copy())
            if pred_mu_batch is None:
                pred_mu_batches.append(np.full(spec.batch_size, np.nan, dtype=float))
                pred_var_batches.append(np.full(spec.batch_size, np.nan, dtype=float))
            else:
                pred_mu_batches.append(np.asarray(pred_mu_batch.numpy(), dtype=float).reshape(-1).copy())
                pred_var_batches.append(np.asarray(pred_var_batch.numpy(), dtype=float).reshape(-1).copy())

            rec = calib.step_batch(Xb, Yb, verbose=False)
            rm = rec.get("restart_mode", None)
            if rm is None:
                rm = "full" if bool(rec.get("did_restart", False)) else "none"
            restart_mode_hist.append(str(rm))
            mean_theta, cov_theta, _, _ = calib._aggregate_particles(0.9)
            mean_np = mean_theta.detach().cpu().numpy().reshape(-1)
            cov_np = cov_theta.detach().cpu().numpy()
            var_diag = np.clip(np.diag(cov_np), 1e-12, None)
            theta_hist.append(mean_np)
            theta_true_hist.append(theta_true.copy())
            theta_var_diag_hist.append(var_diag)
            theta_crps_hist.append(_theta_crps_diag_mean(mean_np, var_diag, theta_true))
    elif mcfg["kind"] == "bc":
        bc_window_points = max(1, int(round(float(spec.bc_window_batches) * int(spec.batch_size))))
        bc = KOHSlidingWindowVector(
            theta_dim=spec.theta_dim,
            window_points=bc_window_points,
            sigma_obs=spec.noise_sd,
            gp_lengthscale=1.5,
            gp_signal_var=spec.discrepancy_amp ** 2,
            theta_lo=-2.0,
            theta_hi=2.0,
        )
        for batch_idx in range(spec.total_batches):
            Xb, Yb = stream.next()
            X_np = Xb.detach().cpu().numpy().copy()
            Y_np = Yb.detach().cpu().numpy().reshape(-1).copy()
            theta_true = stream.theta_true_hist[-1]
            y0_np = stream.y_noiseless_hist[-1]
            pred_mu_batch = None
            pred_var_batch = None

            if batch_idx > 0:
                mu_np, var_np = bc.predict(X_np)
                pred_mu_batch = np.asarray(mu_np, dtype=float).reshape(-1)
                pred_var_batch = np.asarray(var_np, dtype=float).reshape(-1)
                y_rmse_hist.append(float(np.sqrt(np.mean((pred_mu_batch - Y_np) ** 2))))
                y_crps_hist.append(
                    float(
                        crps_gaussian(
                            torch.tensor(pred_mu_batch, dtype=torch.float64),
                            torch.tensor(pred_var_batch, dtype=torch.float64),
                            torch.tensor(Y_np, dtype=torch.float64),
                        ).mean().item()
                    )
                )

            X_batches.append(X_np)
            Y_batches.append(Y_np.copy())
            y_noiseless_batches.append(np.asarray(y0_np, dtype=float).reshape(-1).copy())
            if pred_mu_batch is None:
                pred_mu_batches.append(np.full(spec.batch_size, np.nan, dtype=float))
                pred_var_batches.append(np.full(spec.batch_size, np.nan, dtype=float))
            else:
                pred_mu_batches.append(pred_mu_batch.copy())
                pred_var_batches.append(pred_var_batch.copy())

            bc.update_batch(X_np, Y_np)
            restart_mode_hist.append("none")
            mean_np = np.asarray(bc.current_theta, dtype=float).reshape(-1)
            var_diag = np.full(spec.theta_dim, np.nan, dtype=float)
            theta_hist.append(mean_np)
            theta_true_hist.append(theta_true.copy())
            theta_var_diag_hist.append(var_diag)
            theta_crps_hist.append(float("nan"))
    elif mcfg["kind"] == "paper_pf_vector":
        x_sobol = torch.quasirandom.SobolEngine(dimension=spec.x_dim, scramble=True, seed=seed + 701)
        x_design_np = x_sobol.draw(8).cpu().numpy()
        theta_rng = np.random.RandomState(seed + 702)
        theta_design_np = theta_rng.uniform(low=-2.0, high=2.0, size=(8, spec.theta_dim)).astype(np.float64)
        da_cfg = WardPaperPFVectorConfig(
            num_particles=int(num_particles),
            theta_dim=int(spec.theta_dim),
            theta_lo=-2.0,
            theta_hi=2.0,
            emulator_var=1.0,
            discrepancy_var=float(spec.discrepancy_amp ** 2),
            sigma_obs_var=float(spec.noise_sd ** 2),
            design_x_points=int(x_design_np.shape[0]),
            design_theta_points=int(theta_design_np.shape[0]),
            x_design_np=x_design_np,
            theta_design_np=theta_design_np,
            prior_l_median=2.0,
            prior_l_logsd=0.50,
            l_min=0.20,
            l_max=8.00,
            move_theta_std=0.15,
            move_logl_std=0.10,
            device="cpu",
            dtype=torch.float64,
            seed=seed + 703,
        )
        da = WardPaperParticleFilterVector(additive_ridge_simulator_np, da_cfg)
        for batch_idx in range(spec.total_batches):
            Xb, Yb = stream.next()
            X_np = Xb.detach().cpu().numpy().copy()
            Y_np = Yb.detach().cpu().numpy().reshape(-1).copy()
            theta_true = stream.theta_true_hist[-1]
            y0_np = stream.y_noiseless_hist[-1]
            pred_mu_batch = None
            pred_var_batch = None

            if batch_idx > 0:
                mu_np, var_np = da.predict_batch(X_np)
                pred_mu_batch = np.asarray(mu_np, dtype=float).reshape(-1)
                pred_var_batch = np.asarray(var_np, dtype=float).reshape(-1)
                y_rmse_hist.append(float(np.sqrt(np.mean((pred_mu_batch - Y_np) ** 2))))
                y_crps_hist.append(
                    float(
                        crps_gaussian(
                            torch.tensor(pred_mu_batch, dtype=torch.float64),
                            torch.tensor(pred_var_batch, dtype=torch.float64),
                            torch.tensor(Y_np, dtype=torch.float64),
                        ).mean().item()
                    )
                )

            X_batches.append(X_np)
            Y_batches.append(Y_np.copy())
            y_noiseless_batches.append(np.asarray(y0_np, dtype=float).reshape(-1).copy())
            if pred_mu_batch is None:
                pred_mu_batches.append(np.full(spec.batch_size, np.nan, dtype=float))
                pred_var_batches.append(np.full(spec.batch_size, np.nan, dtype=float))
            else:
                pred_mu_batches.append(pred_mu_batch.copy())
                pred_var_batches.append(pred_var_batch.copy())

            da.step_batch(X_np, Y_np)
            restart_mode_hist.append("none")
            mean_np, var_diag = da.posterior_mean_var_diag()
            var_diag = np.clip(np.asarray(var_diag, dtype=float).reshape(-1), 1e-12, None)
            theta_hist.append(np.asarray(mean_np, dtype=float).reshape(-1))
            theta_true_hist.append(theta_true.copy())
            theta_var_diag_hist.append(var_diag)
            theta_crps_hist.append(_theta_crps_diag_mean(mean_np, var_diag, theta_true))
    elif mcfg["kind"] == "da":
        da = HighDimMovePF(
            theta_dim=spec.theta_dim,
            num_particles=num_particles,
            theta_lo=-2.0,
            theta_hi=2.0,
            sigma_like=float(np.sqrt(spec.noise_sd ** 2 + spec.discrepancy_amp ** 2)),
            move_theta_std=0.15,
            seed=seed + 707,
        )
        for batch_idx in range(spec.total_batches):
            Xb, Yb = stream.next()
            X_np = Xb.detach().cpu().numpy().copy()
            Y_np = Yb.detach().cpu().numpy().reshape(-1).copy()
            theta_true = stream.theta_true_hist[-1]
            y0_np = stream.y_noiseless_hist[-1]
            pred_mu_batch = None
            pred_var_batch = None

            if batch_idx > 0:
                mu_np, var_np = da.predict_batch(X_np)
                pred_mu_batch = np.asarray(mu_np, dtype=float).reshape(-1)
                pred_var_batch = np.asarray(var_np, dtype=float).reshape(-1)
                y_rmse_hist.append(float(np.sqrt(np.mean((pred_mu_batch - Y_np) ** 2))))
                y_crps_hist.append(
                    float(
                        crps_gaussian(
                            torch.tensor(pred_mu_batch, dtype=torch.float64),
                            torch.tensor(pred_var_batch, dtype=torch.float64),
                            torch.tensor(Y_np, dtype=torch.float64),
                        ).mean().item()
                    )
                )

            X_batches.append(X_np)
            Y_batches.append(Y_np.copy())
            y_noiseless_batches.append(np.asarray(y0_np, dtype=float).reshape(-1).copy())
            if pred_mu_batch is None:
                pred_mu_batches.append(np.full(spec.batch_size, np.nan, dtype=float))
                pred_var_batches.append(np.full(spec.batch_size, np.nan, dtype=float))
            else:
                pred_mu_batches.append(pred_mu_batch.copy())
                pred_var_batches.append(pred_var_batch.copy())

            da.update_batch(X_np, Y_np)
            restart_mode_hist.append("none")
            mean_np = da.mean_theta()
            var_diag = np.clip(da.var_theta_diag(), 1e-12, None)
            theta_hist.append(mean_np)
            theta_true_hist.append(theta_true.copy())
            theta_var_diag_hist.append(var_diag)
            theta_crps_hist.append(_theta_crps_diag_mean(mean_np, var_diag, theta_true))
    else:
        raise ValueError(f"Unsupported method kind: {mcfg['kind']}")

    runtime_sec = float(time() - t0)
    theta_arr = np.asarray(theta_hist, dtype=float)
    theta_true_arr = np.asarray(theta_true_hist, dtype=float)
    theta_var_arr = np.asarray(theta_var_diag_hist, dtype=float)
    restart_batches = [idx for idx, mode in enumerate(restart_mode_hist) if mode != "none"]

    tag = f"{scenario}_{spec.data_mode}_seed{seed}_dx{spec.x_dim}_dth{spec.theta_dim}_B{spec.batch_size}_Tb{spec.total_batches}_{run_name}"
    orthog_ratio = float(stream.discrepancy.orthog_ratio) if stream.discrepancy is not None else float("nan")
    raw_dir = out_dir / "raw_runs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"{tag}.pt"
    payload = dict(
        method=method_name,
        run_name=run_name,
        paper_label=paper_label,
        scenario=scenario,
        seed=int(seed),
        x_dim=int(spec.x_dim),
        theta_dim=int(spec.theta_dim),
        batch_size=int(spec.batch_size),
        total_batches=int(spec.total_batches),
        data_mode=str(spec.data_mode),
        projection_ref_size=int(spec.projection_ref_size),
        sudden_jump_scale=float(spec.sudden_jump_scale),
        mixed_jump_scale=float(spec.mixed_jump_scale),
        bc_window_batches=float(spec.bc_window_batches),
        cp_batches=list(stream.cp_batches),
        cp_times=list(stream.cp_times),
        theta=theta_arr,
        theta_oracle=theta_true_arr,
        theta_star_true=theta_true_arr,
        theta_var_diag=theta_var_arr,
        restart_mode_hist=list(restart_mode_hist),
        X_batches=X_batches,
        Y_batches=Y_batches,
        y_noiseless_batches=y_noiseless_batches,
        pred_mu_batches=pred_mu_batches,
        pred_var_batches=pred_var_batches,
        y_rmse_hist=np.asarray(y_rmse_hist, dtype=float),
        y_crps_hist=np.asarray(y_crps_hist, dtype=float),
        theta_crps_hist=np.asarray(theta_crps_hist, dtype=float),
        orthog_ratio=orthog_ratio,
        runtime_sec=runtime_sec,
        controller_overrides=controller_overrides,
    )
    torch.save(payload, raw_path)
    _save_theta_tracking(out_dir, tag, theta_arr, theta_true_arr, theta_var_arr, stream.cp_batches, restart_batches, paper_label)

    theta_rmse = float(np.sqrt(np.mean((theta_arr - theta_true_arr) ** 2)))
    event_stats = _match_events_forward(stream.cp_batches, restart_batches, tol=2) if scenario in {"sudden", "mixed"} else {
        "precision_at2": float("nan"),
        "recall_at2": float("nan"),
        "f1_at2": float("nan"),
        "mean_delay": float("nan"),
    }
    return dict(
        scenario=scenario,
        run_name=run_name,
        method=method_name,
        paper_label=paper_label,
        seed=int(seed),
        x_dim=int(spec.x_dim),
        theta_dim=int(spec.theta_dim),
        batch_size=int(spec.batch_size),
        total_batches=int(spec.total_batches),
        num_particles=int(num_particles),
        num_support=int(num_support),
        delta_bpc_lambda=float(delta_bpc_lambda),
        data_mode=str(spec.data_mode),
        projection_ref_size=int(spec.projection_ref_size),
        sudden_jump_scale=float(spec.sudden_jump_scale),
        mixed_jump_scale=float(spec.mixed_jump_scale),
        bc_window_batches=float(spec.bc_window_batches),
        orthog_ratio=orthog_ratio,
        theta_rmse=theta_rmse,
        theta_crps=_finite_mean(theta_crps_hist),
        y_rmse=_finite_mean(y_rmse_hist),
        y_crps=_finite_mean(y_crps_hist),
        restart_count=float(len(restart_batches)),
        runtime_sec=runtime_sec,
        raw_relpath=str(raw_path.relative_to(out_dir)).replace("\\", "/"),
        hazard_lambda=float(controller_overrides.get("hazard_lambda", float("nan"))),
        restart_margin=float(controller_overrides.get("restart_margin", float("nan"))),
        restart_cooldown=float(controller_overrides.get("restart_cooldown", float("nan"))),
        controller_mode=str(controller_overrides.get("controller_mode", mcfg.get("controller", "none"))),
        wcusum_window=float(controller_overrides.get("wcusum_window", float("nan"))),
        wcusum_threshold=float(controller_overrides.get("wcusum_threshold", float("nan"))),
        wcusum_kappa=float(controller_overrides.get("wcusum_kappa", float("nan"))),
        wcusum_sigma_floor=float(controller_overrides.get("wcusum_sigma_floor", float("nan"))),
        **event_stats,
    )


def write_summaries(out_dir: Path, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    summary_dir = out_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    run_df = pd.DataFrame(rows)
    run_df.to_csv(summary_dir / "run_level.csv", index=False)
    agg_df = (
        run_df.groupby(["scenario", "data_mode", "run_name", "method", "paper_label", "batch_size", "num_particles", "num_support", "delta_bpc_lambda", "sudden_jump_scale", "mixed_jump_scale", "bc_window_batches"], as_index=False)
        .agg(
            theta_rmse_mean=("theta_rmse", "mean"),
            theta_rmse_std=("theta_rmse", "std"),
            theta_crps_mean=("theta_crps", "mean"),
            y_rmse_mean=("y_rmse", "mean"),
            y_rmse_std=("y_rmse", "std"),
            y_crps_mean=("y_crps", "mean"),
            restart_count_mean=("restart_count", "mean"),
            runtime_sec_mean=("runtime_sec", "mean"),
            f1_at2_mean=("f1_at2", "mean"),
            mean_delay_mean=("mean_delay", "mean"),
            orthog_ratio_mean=("orthog_ratio", "mean"),
            controller_mode=("controller_mode", "first"),
            hazard_lambda=("hazard_lambda", "mean"),
            restart_margin=("restart_margin", "mean"),
            restart_cooldown=("restart_cooldown", "mean"),
            wcusum_window=("wcusum_window", "mean"),
            wcusum_threshold=("wcusum_threshold", "mean"),
            wcusum_kappa=("wcusum_kappa", "mean"),
            wcusum_sigma_floor=("wcusum_sigma_floor", "mean"),
        )
    )
    agg_df.to_csv(summary_dir / "scenario_summary.csv", index=False)
    with (summary_dir / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "method": "B-BRPC-F",
                "paper_label": "B-BRPC-F",
                "data_modes": sorted(run_df["data_mode"].unique().tolist()) if "data_mode" in run_df.columns else ["orthogonalized_rff"],
                "scenarios": sorted(run_df["scenario"].unique().tolist()),
                "seeds": sorted(run_df["seed"].unique().tolist()),
                "notes": "Moderate high-dimensional projected diagnostic with 20D inputs, 5D theta, optional orthogonalized RFF or physical_projected L2 target data, and fixed-support online-BPC + R-BOCPD.",
            },
            fh,
            indent=2,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="figs/highdim_projected_diag_brpcf")
    parser.add_argument("--scenarios", nargs="+", choices=["slope", "sudden", "mixed", "all"], default=["all"])
    parser.add_argument("--seed_count", type=int, default=5)
    parser.add_argument("--seed_offset", type=int, default=0)
    parser.add_argument("--num_particles", type=int, default=1024)
    parser.add_argument("--delta_bpc_lambda", type=float, default=2.0)
    parser.add_argument("--num_support", type=int, default=32)
    parser.add_argument("--total_batches", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_rff", type=int, default=10)
    parser.add_argument("--discrepancy_amp", type=float, default=0.3)
    parser.add_argument("--noise_sd", type=float, default=0.05)
    parser.add_argument("--methods", nargs="+", default=["B-BRPC-F"])
    parser.add_argument("--data_mode", choices=["orthogonalized_rff", "physical_projected"], default="orthogonalized_rff")
    parser.add_argument("--projection_ref_size", type=int, default=50000)
    parser.add_argument("--projection_ridge", type=float, default=1e-6)
    parser.add_argument("--physical_h_amp", type=float, default=0.4)
    parser.add_argument("--physical_local_scale", type=float, default=0.12)
    parser.add_argument("--sudden_jump_scale", type=float, default=0.8)
    parser.add_argument("--mixed_jump_scale", type=float, default=0.6)
    parser.add_argument("--bc_window_batches", type=float, default=2.5)
    parser.add_argument("--wcusum_threshold", type=float, default=None)
    parser.add_argument("--wcusum_window", type=int, default=None)
    parser.add_argument("--wcusum_kappa", type=float, default=None)
    parser.add_argument("--wcusum_sigma_floor", type=float, default=None)
    args = parser.parse_args()

    scenarios = ["slope", "sudden", "mixed"] if "all" in args.scenarios else list(args.scenarios)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = HighDimDiagSpec(
        total_batches=int(args.total_batches),
        batch_size=int(args.batch_size),
        noise_sd=float(args.noise_sd),
        num_rff=int(args.num_rff),
        discrepancy_amp=float(args.discrepancy_amp),
        data_mode=str(args.data_mode),
        projection_ref_size=int(args.projection_ref_size),
        projection_ridge=float(args.projection_ridge),
        physical_h_amp=float(args.physical_h_amp),
        physical_local_scale=float(args.physical_local_scale),
        sudden_jump_scale=float(args.sudden_jump_scale),
        mixed_jump_scale=float(args.mixed_jump_scale),
        bc_window_batches=float(args.bc_window_batches),
    )

    valid_methods = method_specs().keys()
    methods = [paper_method_name(m) for m in args.methods]
    for m in methods:
        if m not in valid_methods:
            raise ValueError(f"Unknown method {m}. Valid methods: {sorted(valid_methods)}")

    rows: List[Dict[str, float]] = []
    for scenario in scenarios:
        for method_name in methods:
            for seed in range(int(args.seed_offset), int(args.seed_offset) + int(args.seed_count)):
                print(f"[highdim] scenario={scenario} method={method_name} seed={seed}")
                controller_overrides: Dict[str, float] = {}
                if method_name == "C-BRPC-F":
                    if args.wcusum_threshold is not None:
                        controller_overrides["wcusum_threshold"] = float(args.wcusum_threshold)
                    if args.wcusum_window is not None:
                        controller_overrides["wcusum_window"] = int(args.wcusum_window)
                    if args.wcusum_kappa is not None:
                        controller_overrides["wcusum_kappa"] = float(args.wcusum_kappa)
                    if args.wcusum_sigma_floor is not None:
                        controller_overrides["wcusum_sigma_floor"] = float(args.wcusum_sigma_floor)
                rows.append(
                    run_one_method(
                        scenario=scenario,
                        seed=seed,
                        out_dir=out_dir,
                        spec=spec,
                        num_particles=int(args.num_particles),
                        delta_bpc_lambda=float(args.delta_bpc_lambda),
                        num_support=int(args.num_support),
                        method_name=method_name,
                        run_name=method_name,
                        controller_overrides=controller_overrides,
                    )
                )
                write_summaries(out_dir, rows)
                save_combined_theta_tracking_from_rows(out_dir, rows)


if __name__ == "__main__":
    main()
