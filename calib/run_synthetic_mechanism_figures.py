"""Mechanism-figure runner for synthetic discrepancy update ablations.

Runs a focused BOCPD half-discrepancy method set across sudden, gradual-drift,
and random-walk synthetic regimes. It saves per-batch mechanism diagnostics,
raw run payloads, and figures for prediction-error decomposition, residual
coherence, expert evidence, run-length posterior mass, and restart-centered
behavior.

conda run -n jumpGP python -m calib.run_synthetic_mechanism_figures --method-set shared --seeds 0 1 2 --out_dir figs/mechanism_shared
conda run -n jumpGP python -m py_compile calib\run_synthetic_mechanism_figures.py
conda run -n jumpGP python -m calib.run_synthetic_mechanism_figures --smoke --out_dir figs/mechanism_figures_check2
conda run -n jumpGP python -m calib.run_synthetic_mechanism_figures --scenarios sudden --seeds 0 --methods half_online half_onlineDynamic particleGP_online --batch-size 10 --sudden-seg-len 20 --num-particles 48 --max-experts 3 --oracle-grid-size 80 --phi2-grid-size 80 --delta-online-init-max-iter 2 --event-window 2 --restart-window 2 --out_dir figs/mechanism_figures_check_online

conda run -n jumpGP python -m calib.run_synthetic_mechanism_figures --scenarios sudden slope random_walk --method-set core --seeds 0 1 2 3 4 --plot-all-seeds --plot-all-heatmaps --plot-all-runlengths --out_dir figs/mechanism_figures_full
conda run -n jumpGP python -m calib.run_synthetic_mechanism_figures --scenarios sudden slope random_walk --method-set basis_ablation --seeds 0 1 2 3 4 --plot-all-seeds --out_dir figs/mechanism_figures_basis20_fixedhyper_ablation
conda run -n jumpGP python -m calib.run_synthetic_mechanism_figures --scenarios sudden slope random_walk --method-set inducing_ablation --seeds 0 1 2 3 4 --plot-all-seeds --out_dir figs/mechanism_figures_inducing_ablation
conda run -n jumpGP python -m calib.run_synthetic_mechanism_figures --scenarios sudden slope random_walk --method-set mc_inducing_ablation --seeds 0 1 2 3 4 --plot-all-seeds --out_dir figs/mechanism_figures_mc_inducing_ablation
conda run -n jumpGP python -m calib.run_synthetic_mechanism_figures --scenarios sudden slope random_walk --method-set online_bpc_ablation --seeds 0 1 2 3 4 --plot-all-seeds --out_dir figs/mechanism_figures_online_bpc_ablation
conda run -n jumpGP python -m calib.run_synthetic_mechanism_figures --scenarios sudden slope random_walk --method-set online_bpc_exact_ablation --seeds 0 1 2 3 4 --plot-all-seeds --out_dir figs/mechanism_figures_online_bpc_exact_ablation
conda run -n jumpGP python -m calib.run_synthetic_mechanism_figures --scenarios sudden slope random_walk --method-set online_bpc_exact_hyper_ablation --seeds 0 1 2 3 4 --plot-all-seeds --out_dir figs/mechanism_figures_online_bpc_exact_hyper_ablation
conda run -n jumpGP python -m calib.run_synthetic_mechanism_figures --scenarios sudden slope random_walk --method-set online_bpc_proxy_ablation --seeds 0 1 2 3 4 --plot-all-seeds --out_dir figs/mechanism_figures_online_bpc_proxy_ablation
conda run -n jumpGP python -m calib.run_synthetic_mechanism_figures --scenarios sudden slope random_walk --method-set online_bpc_proxy_stable_ablation --seeds 0 1 2 3 4 --plot-all-seeds --out_dir figs/mechanism_figures_online_bpc_proxy_stable_ablation
conda run -n jumpGP python -m calib.run_synthetic_mechanism_figures --scenarios sudden slope random_walk --method-set online_bpc_proxy_sigmaobs_ablation --seeds 0 1 2 3 4 --plot-all-seeds --out_dir figs/mechanism_figures_online_bpc_proxy_sigmaobs_ablation
conda run -n jumpGP python -m calib.run_synthetic_mechanism_figures --scenarios sudden slope random_walk --method-set online_bpc_sigmaobs_best_ablation --seeds 0 1 2 3 4 --plot-all-seeds --out_dir figs/mechanism_figures_online_bpc_sigmaobs_best_ablation
conda run -n jumpGP python -m calib.run_synthetic_mechanism_figures --scenarios sudden slope random_walk --method-set online_bpc_fixedsupport_ablation --seeds 0 1 2 3 4 --plot-all-seeds --out_dir figs/mechanism_figures_online_bpc_fixedsupport_ablation
conda run -n jumpGP python -m calib.run_synthetic_mechanism_figures --scenarios sudden slope random_walk --method-set online_bpc_controller_ablation --seeds 0 1 2 3 4 --plot-all-seeds --out_dir figs/mechanism_figures_online_bpc_controller_ablation


"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from time import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .configs import CalibrationConfig
from .emulator import DeterministicSimulator
from .online_calibrator import OnlineBayesCalibrator, crps_gaussian
from .restart_bocpd_debug_260115_gpytorch import RollingStats


def simulator_np(x: np.ndarray, theta: float | np.ndarray) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float).reshape(-1)
    theta_val = float(np.asarray(theta, dtype=float).reshape(-1)[0])
    return np.sin(5.0 * theta_val * x_arr) + 5.0 * x_arr


def simulator_torch(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    if x.dim() == 1:
        x = x[:, None]
    if theta.dim() == 1:
        theta = theta[None, :]
    return torch.sin(5.0 * theta[:, 0:1] * x[:, 0:1]) + 5.0 * x[:, 0:1]


def physical_system_np(x: np.ndarray, phi: np.ndarray) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float).reshape(-1)
    a1, a2, a3 = np.asarray(phi, dtype=float).reshape(-1)
    return a1 * x_arr * np.cos(a2 * x_arr) + a3 * x_arr


def oracle_theta(phi: np.ndarray, theta_grid: np.ndarray, x_grid: Optional[np.ndarray] = None) -> float:
    if x_grid is None:
        x_grid = np.linspace(0.0, 1.0, 400).reshape(-1, 1)
    eta = physical_system_np(x_grid, phi)
    errs = [float(np.mean((eta - simulator_np(x_grid, th)) ** 2)) for th in theta_grid]
    return float(theta_grid[int(np.argmin(errs))])


@dataclass
class Phi2ThetaMap:
    theta_grid: np.ndarray
    phi2_grid: np.ndarray

    def phi2_of_theta(self, theta: float) -> float:
        return float(np.interp(float(theta), self.theta_grid, self.phi2_grid))


def build_phi2_theta_map(theta_grid_size: int = 500, phi2_grid_size: int = 400,
                         phi2_min: float = 2.0, phi2_max: float = 12.0) -> Phi2ThetaMap:
    theta_grid = np.linspace(0.0, 3.0, int(theta_grid_size))
    phi2_grid = np.linspace(float(phi2_min), float(phi2_max), int(phi2_grid_size))
    x_grid = np.linspace(0.0, 1.0, 400).reshape(-1, 1)
    theta_star_vals = []
    for phi2 in phi2_grid:
        phi = np.array([5.0, float(phi2), 5.0], dtype=float)
        theta_star_vals.append(oracle_theta(phi, theta_grid, x_grid=x_grid))
    theta_star_vals = np.asarray(theta_star_vals, dtype=float)
    order = np.argsort(theta_star_vals)
    theta_unique, unique_idx = np.unique(theta_star_vals[order], return_index=True)
    phi2_unique = phi2_grid[order][unique_idx]
    return Phi2ThetaMap(theta_grid=theta_unique, phi2_grid=phi2_unique)


def build_phi_segments_centered(delta: float, center: float = 7.5) -> List[np.ndarray]:
    phi2_vals = np.array([center - 1.5 * delta, center - 0.5 * delta,
                          center + 0.5 * delta, center + 1.5 * delta])
    return [np.array([5.0, float(v), 5.0], dtype=float) for v in phi2_vals]


@dataclass
class ScenarioSpec:
    name: str
    theta_path: np.ndarray
    phi_path: np.ndarray
    batch_size: int
    noise_sd: float
    event_batches: List[int]
    event_kind: str
    metadata: Dict[str, Any]

    @property
    def n_batches(self) -> int:
        return int(self.theta_path.shape[0])

    @property
    def total_T(self) -> int:
        return int(self.n_batches * self.batch_size)


class ThetaPathPhysicalStream:
    def __init__(self, spec: ScenarioSpec, seed: int, shuffle_x: bool = True):
        self.spec = spec
        self.rng = np.random.RandomState(int(seed))
        self.shuffle_x = bool(shuffle_x)
        self.batch_idx = 0

    def next(self) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        if self.batch_idx >= self.spec.n_batches:
            raise StopIteration
        bs = int(self.spec.batch_size)
        X = ((np.arange(bs, dtype=float) + self.rng.rand(bs)) / bs)[:, None]
        if self.shuffle_x:
            self.rng.shuffle(X)
        theta_star = float(self.spec.theta_path[self.batch_idx])
        phi = np.asarray(self.spec.phi_path[self.batch_idx], dtype=float)
        y_noiseless = physical_system_np(X, phi)
        noise = self.spec.noise_sd * self.rng.randn(bs)
        y = y_noiseless + noise
        sim_true = simulator_np(X, theta_star)
        delta_star = y_noiseless - sim_true
        info = dict(batch_idx=int(self.batch_idx), t_obs=int(self.batch_idx * bs),
                    theta_star=theta_star, phi=phi.copy(), X=X.copy(), y=y.copy(),
                    y_noiseless=y_noiseless.copy(), sim_true=sim_true.copy(),
                    delta_star=delta_star.copy(), noise=noise.copy())
        self.batch_idx += 1
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), info


def phi_path_from_theta(theta_path: np.ndarray, mapper: Phi2ThetaMap) -> np.ndarray:
    return np.vstack([np.array([5.0, mapper.phi2_of_theta(float(theta)), 5.0], dtype=float)
                      for theta in theta_path])


def make_sudden_spec(args: argparse.Namespace, mapper: Phi2ThetaMap) -> ScenarioSpec:
    del mapper
    if int(args.sudden_seg_len) % int(args.batch_size) != 0:
        raise ValueError("--sudden-seg-len must be divisible by --batch-size")
    seg_len = int(args.sudden_seg_len)
    bs = int(args.batch_size)
    n_batches = int(4 * seg_len // bs)
    cp_batches = [int(seg_len // bs), int(2 * seg_len // bs), int(3 * seg_len // bs)]
    cp_times = [seg_len, 2 * seg_len, 3 * seg_len]
    phi_segments = build_phi_segments_centered(float(args.sudden_mag), float(args.phi_center))
    theta_grid = np.linspace(0.0, 3.0, int(args.oracle_grid_size))
    theta_segments = np.array([oracle_theta(phi, theta_grid) for phi in phi_segments], dtype=float)
    theta_path = np.zeros(n_batches, dtype=float)
    phi_path = np.zeros((n_batches, 3), dtype=float)
    for b in range(n_batches):
        t_obs = b * bs
        seg_id = sum(t_obs >= cp for cp in cp_times)
        theta_path[b] = theta_segments[seg_id]
        phi_path[b] = phi_segments[seg_id]
    return ScenarioSpec("sudden", theta_path, phi_path, bs, float(args.noise_sd), cp_batches, "true_cp",
                        dict(sudden_seg_len=seg_len, sudden_mag=float(args.sudden_mag),
                             phi_center=float(args.phi_center), theta_segments=theta_segments.tolist(),
                             cp_times=cp_times))


def make_slope_spec(args: argparse.Namespace, mapper: Phi2ThetaMap) -> ScenarioSpec:
    total_T = int(args.slope_total_T)
    bs = int(args.batch_size)
    if total_T % bs != 0:
        raise ValueError("--slope-total-T must be divisible by --batch-size")
    n_batches = int(total_T // bs)
    t_obs = np.arange(n_batches, dtype=float) * bs
    theta_path = np.clip(float(args.theta0) + float(args.slope) * t_obs, 0.05, 2.95)
    phi_path = phi_path_from_theta(theta_path, mapper)
    event_batches = sorted(set([n_batches // 3, (2 * n_batches) // 3]))
    return ScenarioSpec("slope", theta_path, phi_path, bs, float(args.noise_sd), event_batches, "drift_window",
                        dict(theta0=float(args.theta0), slope=float(args.slope), total_T=total_T))


def random_walk_theta_path(n_batches: int, theta0: float, step_sd: float, seed: int) -> np.ndarray:
    rng = np.random.RandomState(int(seed) + 10007)
    out = np.zeros(int(n_batches), dtype=float)
    out[0] = float(theta0)
    for b in range(1, int(n_batches)):
        out[b] = np.clip(out[b - 1] + rng.normal(0.0, float(step_sd)), 0.05, 2.95)
    return out


def turning_points(theta_path: np.ndarray, max_points: int = 4) -> List[int]:
    theta = np.asarray(theta_path, dtype=float)
    if theta.size < 4:
        return []
    diff = np.diff(theta)
    sign = np.sign(diff)
    candidates = []
    for i in range(1, sign.size):
        if sign[i - 1] != 0 and sign[i] != 0 and sign[i - 1] != sign[i]:
            candidates.append(i)
    if len(candidates) == 0:
        curvature = np.abs(np.diff(theta, n=2))
        order = np.argsort(curvature)[::-1] if curvature.size else []
        candidates = [int(i + 1) for i in order[:max_points]]
    return sorted(set(int(c) for c in candidates[:max_points] if 0 < c < theta.size))


def make_random_walk_spec(args: argparse.Namespace, mapper: Phi2ThetaMap, seed: int) -> ScenarioSpec:
    total_T = int(args.rw_total_T)
    bs = int(args.batch_size)
    if total_T % bs != 0:
        raise ValueError("--rw-total-T must be divisible by --batch-size")
    n_batches = int(total_T // bs)
    theta_path = random_walk_theta_path(n_batches, float(args.theta0), float(args.rw_step_sd), int(seed))
    phi_path = phi_path_from_theta(theta_path, mapper)
    event_batches = turning_points(theta_path, max_points=4)
    return ScenarioSpec("random_walk", theta_path, phi_path, bs, float(args.noise_sd), event_batches, "turning_point",
                        dict(theta0=float(args.theta0), rw_step_sd=float(args.rw_step_sd), total_T=total_T))


def default_methods() -> Dict[str, Dict[str, Any]]:
    base = dict(type="bocpd", mode="restart", use_discrepancy=False, bocpd_use_discrepancy=True,
                restart_impl="rolled_cusum_260324", hybrid_partial_restart=False,
                use_dual_restart=False, use_cusum=False, particle_delta_mode="shared_gp",
                shared_delta_model="gp", delta_update_mode="refit")
    return {
        "half_refit": dict(base),
        "half_basisRefit": dict(base, shared_delta_model="basis"),
        "half_basisRefitFixedHyper": dict(base, shared_delta_model="basis", delta_basis_fix_hyper=True),
        "half_online": dict(base, delta_update_mode="online"),
        "half_onlineInducing": dict(base, delta_update_mode="online_inducing"),
        "half_onlineInducing_none": dict(base, mode="wcusum", controller_name="none", controller_stat="log_surprise_mean", delta_update_mode="online_inducing"),
        "half_onlineInducing_wCUSUM": dict(base, mode="wcusum", controller_name="wcusum", controller_stat="log_surprise_mean", controller_wcusum_warmup_batches=3, controller_wcusum_window=4, controller_wcusum_threshold=0.25, controller_wcusum_kappa=0.25, controller_wcusum_sigma_floor=0.25, delta_update_mode="online_inducing"),
        "half_onlineDynamic": dict(base, delta_update_mode="online_dynamic"),
        "shared_onlineBPC": dict(base, delta_update_mode="online_bpc"),
        "shared_onlineBPC_exact": dict(base, delta_update_mode="online_bpc_exact"),
        "shared_onlineBPC_exact_sigmaObs": dict(base, delta_update_mode="online_bpc_exact", delta_bpc_obs_noise_mode="sigma_eps", delta_bpc_predict_add_kernel_noise=False),
        "shared_onlineBPC_fixedSupport_sigmaObs": dict(base, delta_update_mode="online_bpc_fixedsupport_exact", delta_bpc_obs_noise_mode="sigma_eps", delta_bpc_predict_add_kernel_noise=False),
        "shared_onlineBPC_fixedSupport_sigmaObs_none": dict(base, mode="wcusum", controller_name="none", controller_stat="log_surprise_mean", delta_update_mode="online_bpc_fixedsupport_exact", delta_bpc_obs_noise_mode="sigma_eps", delta_bpc_predict_add_kernel_noise=False),
        "shared_onlineBPC_fixedSupport_sigmaObs_wCUSUM": dict(base, mode="wcusum", controller_name="wcusum", controller_stat="log_surprise_mean", controller_wcusum_warmup_batches=3, controller_wcusum_window=4, controller_wcusum_threshold=0.25, controller_wcusum_kappa=0.25, controller_wcusum_sigma_floor=0.25, delta_update_mode="online_bpc_fixedsupport_exact", delta_bpc_obs_noise_mode="sigma_eps", delta_bpc_predict_add_kernel_noise=False),
        "shared_onlineBPC_exact_refitHyper": dict(base, delta_update_mode="online_bpc_exact_refithyper"),
        "shared_onlineBPC_proxyRefitHyper": dict(base, delta_update_mode="online_bpc_proxy_refithyper"),
        "shared_onlineBPC_proxyRefitHyper_sigmaObs": dict(base, delta_update_mode="online_bpc_proxy_refithyper", delta_bpc_obs_noise_mode="sigma_eps", delta_bpc_predict_add_kernel_noise=False),
        "shared_onlineBPC_proxyStableMean": dict(base, delta_update_mode="online_bpc_proxy_stablemean"),
        "shared_onlineBPC_proxyStableMean_sigmaObs": dict(base, delta_update_mode="online_bpc_proxy_stablemean", delta_bpc_obs_noise_mode="sigma_eps", delta_bpc_predict_add_kernel_noise=False),
        "shared_onlineBPC_proxyStableMean_sigmaObs_h400": dict(base, delta_update_mode="online_bpc_proxy_stablemean", delta_bpc_obs_noise_mode="sigma_eps", delta_bpc_predict_add_kernel_noise=False, hazard_lambda=400.0),
        "shared_onlineBPC_proxyStableMean_sigmaObs_m2": dict(base, delta_update_mode="online_bpc_proxy_stablemean", delta_bpc_obs_noise_mode="sigma_eps", delta_bpc_predict_add_kernel_noise=False, restart_margin=2.0),
        "shared_onlineBPC_proxyStableMean_sigmaObs_c20": dict(base, delta_update_mode="online_bpc_proxy_stablemean", delta_bpc_obs_noise_mode="sigma_eps", delta_bpc_predict_add_kernel_noise=False, restart_cooldown=20),
        "shared_onlineBPC_proxyStableMean_sigmaObs_h400_m2_c20": dict(base, delta_update_mode="online_bpc_proxy_stablemean", delta_bpc_obs_noise_mode="sigma_eps", delta_bpc_predict_add_kernel_noise=False, hazard_lambda=400.0, restart_margin=2.0, restart_cooldown=20),
        "shared_onlineBPC_proxyStableMean_sigmaObs_h800_m2_c20": dict(base, delta_update_mode="online_bpc_proxy_stablemean", delta_bpc_obs_noise_mode="sigma_eps", delta_bpc_predict_add_kernel_noise=False, hazard_lambda=800.0, restart_margin=2.0, restart_cooldown=20),
        "shared_onlineBPC_proxyStableMean_sigmaObs_h400_m4_c20": dict(base, delta_update_mode="online_bpc_proxy_stablemean", delta_bpc_obs_noise_mode="sigma_eps", delta_bpc_predict_add_kernel_noise=False, hazard_lambda=400.0, restart_margin=4.0, restart_cooldown=20),
        "shared_onlineBPC_proxyStableMean_sigmaObs_backdated_h400_m2_c20": dict(base, delta_update_mode="online_bpc_proxy_stablemean", delta_bpc_obs_noise_mode="sigma_eps", delta_bpc_predict_add_kernel_noise=False, hazard_lambda=400.0, restart_margin=2.0, restart_cooldown=20, use_backdated_restart=True),
        "shared_onlineBPC_exact_sigmaObs_none": dict(base, mode="wcusum", controller_name="none", controller_stat="log_surprise_mean", delta_update_mode="online_bpc_exact", delta_bpc_obs_noise_mode="sigma_eps", delta_bpc_predict_add_kernel_noise=False),
        "shared_onlineBPC_exact_sigmaObs_wCUSUM": dict(base, mode="wcusum", controller_name="wcusum", controller_stat="log_surprise_mean", controller_wcusum_warmup_batches=3, controller_wcusum_window=4, controller_wcusum_threshold=0.25, controller_wcusum_kappa=0.25, controller_wcusum_sigma_floor=0.25, delta_update_mode="online_bpc_exact", delta_bpc_obs_noise_mode="sigma_eps", delta_bpc_predict_add_kernel_noise=False),
        "shared_onlineBPC_exact_sigmaObs_srCS": dict(base, mode="single_segment", controller_name="sr_cs", controller_stat="calibration_gap_mean", controller_cs_clip_low=-1.0, controller_cs_clip_high=4.0, delta_update_mode="online_bpc_exact", delta_bpc_obs_noise_mode="sigma_eps", delta_bpc_predict_add_kernel_noise=False),
        "shared_onlineBPC_proxyStableMean_sigmaObs_none": dict(base, mode="wcusum", controller_name="none", controller_stat="log_surprise_mean", delta_update_mode="online_bpc_proxy_stablemean", delta_bpc_obs_noise_mode="sigma_eps", delta_bpc_predict_add_kernel_noise=False),
        "shared_onlineBPC_proxyStableMean_sigmaObs_wCUSUM": dict(base, mode="wcusum", controller_name="wcusum", controller_stat="log_surprise_mean", controller_wcusum_warmup_batches=3, controller_wcusum_window=4, controller_wcusum_threshold=0.25, controller_wcusum_kappa=0.25, controller_wcusum_sigma_floor=0.25, delta_update_mode="online_bpc_proxy_stablemean", delta_bpc_obs_noise_mode="sigma_eps", delta_bpc_predict_add_kernel_noise=False),
        "shared_onlineBPC_proxyStableMean_sigmaObs_srCS": dict(base, mode="single_segment", controller_name="sr_cs", controller_stat="calibration_gap_mean", controller_cs_clip_low=-1.0, controller_cs_clip_high=4.0, delta_update_mode="online_bpc_proxy_stablemean", delta_bpc_obs_noise_mode="sigma_eps", delta_bpc_predict_add_kernel_noise=False),
        "DOC_SharedMCInducing": dict(base, delta_update_mode="online_shared_mc_inducing"),
        "DOC_SharedMCInducing_Refresh": dict(base, delta_update_mode="online_shared_mc_inducing_refresh", delta_mc_refresh_every=5),
        "DOC_ParticleMCInducing": dict(base, particle_delta_mode="particle_mc_inducing", delta_update_mode="online_particle_mc_inducing", delta_mc_num_inducing_points=8, delta_mc_num_particles=4),
        "DOC_ParticleMCInducing_Refresh": dict(base, particle_delta_mode="particle_mc_inducing", delta_update_mode="online_particle_mc_inducing_refresh", delta_mc_num_inducing_points=8, delta_mc_num_particles=4, delta_mc_refresh_every=5),
        "WardPF_BOCPD": dict(base, particle_delta_mode="particle_mc_inducing", delta_update_mode="online_particle_mc_inducing", delta_mc_num_inducing_points=8, delta_mc_num_particles=4, use_discrepancy=True, bocpd_use_discrepancy=True),
        "particleGP_refit": dict(base, particle_delta_mode="particle_gp_shared_hyper"),
        "particleGP_online": dict(base, particle_delta_mode="particle_gp_online_shared_hyper", delta_update_mode="online"),
        "particleGP_onlineDynamic": dict(base, particle_delta_mode="particle_gp_dynamic_shared_hyper", delta_update_mode="online_dynamic"),
        "particleGP_onlineBPC": dict(base, particle_delta_mode="particle_gp_online_bpc_shared_hyper", delta_update_mode="online_bpc"),
        "particleGP_onlineBPC_exact": dict(base, particle_delta_mode="particle_gp_online_bpc_exact_shared_hyper", delta_update_mode="online_bpc_exact"),
        "particleGP_fixedSupport_onlineBPC_sigmaObs": dict(base, particle_delta_mode="particle_gp_fixedsupport_online_bpc_shared_hyper", delta_update_mode="online_bpc_fixedsupport_exact", delta_bpc_obs_noise_mode="sigma_eps", delta_bpc_predict_add_kernel_noise=False),
    }


def select_methods(args: argparse.Namespace) -> Dict[str, Dict[str, Any]]:
    methods = default_methods()
    if args.method_set == "shared":
        keep = ["half_refit", "half_online", "half_onlineDynamic"]
    elif args.method_set == "particle":
        keep = ["particleGP_refit", "particleGP_online", "particleGP_onlineDynamic"]
    elif args.method_set == "basis_ablation":
        keep = ["half_refit", "half_basisRefitFixedHyper", "half_onlineDynamic"]
    elif args.method_set == "inducing_ablation":
        keep = ["half_refit", "half_onlineInducing", "half_onlineDynamic"]
    elif args.method_set == "mc_inducing_ablation":
        keep = ["half_refit", "half_onlineDynamic", "DOC_SharedMCInducing", "DOC_SharedMCInducing_Refresh", "DOC_ParticleMCInducing", "DOC_ParticleMCInducing_Refresh", "WardPF_BOCPD"]
    elif args.method_set == "online_bpc_ablation":
        keep = ["half_refit", "shared_onlineBPC", "particleGP_onlineBPC"]
    elif args.method_set == "online_bpc_exact_ablation":
        keep = ["half_refit", "shared_onlineBPC_exact", "particleGP_onlineBPC_exact"]
    elif args.method_set == "online_bpc_exact_hyper_ablation":
        keep = ["half_refit", "shared_onlineBPC_exact", "shared_onlineBPC_exact_refitHyper"]
    elif args.method_set == "online_bpc_proxy_ablation":
        keep = ["half_refit", "shared_onlineBPC_exact", "shared_onlineBPC_proxyRefitHyper"]
    elif args.method_set == "online_bpc_proxy_stable_ablation":
        keep = ["half_refit", "shared_onlineBPC_exact", "shared_onlineBPC_proxyRefitHyper", "shared_onlineBPC_proxyStableMean"]
    elif args.method_set == "online_bpc_proxy_sigmaobs_ablation":
        keep = ["half_refit", "shared_onlineBPC_proxyRefitHyper", "shared_onlineBPC_proxyRefitHyper_sigmaObs", "shared_onlineBPC_proxyStableMean", "shared_onlineBPC_proxyStableMean_sigmaObs"]
    elif args.method_set == "online_bpc_sigmaobs_best_ablation":
        keep = ["half_refit", "shared_onlineBPC_exact_sigmaObs", "shared_onlineBPC_proxyStableMean_sigmaObs"]
    elif args.method_set == "online_bpc_fixedsupport_ablation":
        keep = ["half_refit", "shared_onlineBPC_exact_sigmaObs", "shared_onlineBPC_proxyStableMean_sigmaObs", "shared_onlineBPC_fixedSupport_sigmaObs"]
    elif args.method_set == "online_bpc_fixedsupport_particle_ablation":
        keep = ["half_refit", "shared_onlineBPC_fixedSupport_sigmaObs", "particleGP_fixedSupport_onlineBPC_sigmaObs"]
    elif args.method_set == "online_bpc_controller_ablation":
        keep = ["shared_onlineBPC_proxyStableMean_sigmaObs_none", "shared_onlineBPC_proxyStableMean_sigmaObs_srCS", "shared_onlineBPC_proxyStableMean_sigmaObs", "shared_onlineBPC_exact_sigmaObs_none", "shared_onlineBPC_exact_sigmaObs_srCS", "shared_onlineBPC_exact_sigmaObs"]
    elif args.method_set == "proxy_bocpd_tuning":
        keep = ["half_refit", "shared_onlineBPC_proxyStableMean_sigmaObs_none", "shared_onlineBPC_proxyStableMean_sigmaObs", "shared_onlineBPC_proxyStableMean_sigmaObs_h400", "shared_onlineBPC_proxyStableMean_sigmaObs_m2", "shared_onlineBPC_proxyStableMean_sigmaObs_c20", "shared_onlineBPC_proxyStableMean_sigmaObs_h400_m2_c20", "shared_onlineBPC_proxyStableMean_sigmaObs_h800_m2_c20", "shared_onlineBPC_proxyStableMean_sigmaObs_h400_m4_c20", "shared_onlineBPC_proxyStableMean_sigmaObs_backdated_h400_m2_c20"]
    elif args.method_set == "wcusum_ablation":
        keep = ["half_refit", "shared_onlineBPC_proxyStableMean_sigmaObs_none", "shared_onlineBPC_proxyStableMean_sigmaObs_wCUSUM", "shared_onlineBPC_exact_sigmaObs_none", "shared_onlineBPC_exact_sigmaObs_wCUSUM", "shared_onlineBPC_fixedSupport_sigmaObs_none", "shared_onlineBPC_fixedSupport_sigmaObs_wCUSUM"]
    else:
        keep = ["half_refit", "half_online", "half_onlineInducing", "half_onlineDynamic", "shared_onlineBPC", "shared_onlineBPC_exact", "shared_onlineBPC_exact_sigmaObs", "shared_onlineBPC_fixedSupport_sigmaObs", "shared_onlineBPC_exact_refitHyper", "shared_onlineBPC_proxyRefitHyper", "shared_onlineBPC_proxyRefitHyper_sigmaObs", "shared_onlineBPC_proxyStableMean", "shared_onlineBPC_proxyStableMean_sigmaObs", "shared_onlineBPC_exact_sigmaObs_none", "shared_onlineBPC_exact_sigmaObs_srCS", "shared_onlineBPC_proxyStableMean_sigmaObs_none", "shared_onlineBPC_proxyStableMean_sigmaObs_srCS", "DOC_SharedMCInducing", "DOC_SharedMCInducing_Refresh", "DOC_ParticleMCInducing", "DOC_ParticleMCInducing_Refresh", "WardPF_BOCPD", "particleGP_refit", "particleGP_online", "particleGP_onlineDynamic", "particleGP_onlineBPC", "particleGP_onlineBPC_exact", "particleGP_fixedSupport_onlineBPC_sigmaObs"]
    if args.methods:
        keep = list(args.methods)
    missing = [m for m in keep if m not in methods]
    if missing:
        raise ValueError(f"Unknown methods: {missing}. Available: {sorted(methods)}")
    return {m: methods[m] for m in keep}


def build_calibration_config(args: argparse.Namespace, meta: Dict[str, Any]) -> CalibrationConfig:
    cfg = CalibrationConfig()
    cfg.bocpd.bocpd_mode = str(meta.get("mode", "restart"))
    cfg.bocpd.use_restart = True
    cfg.bocpd.restart_impl = str(meta.get("restart_impl", "rolled_cusum_260324"))
    cfg.bocpd.hybrid_partial_restart = bool(meta.get("hybrid_partial_restart", False))
    cfg.bocpd.use_dual_restart = bool(meta.get("use_dual_restart", False))
    cfg.bocpd.use_cusum = bool(meta.get("use_cusum", False))
    cfg.bocpd.particle_delta_mode = str(meta.get("particle_delta_mode", "shared_gp"))
    cfg.bocpd.particle_gp_hyper_candidates = meta.get("particle_gp_hyper_candidates", None)
    cfg.bocpd.hazard_lambda = float(meta.get("hazard_lambda", args.hazard_lambda))
    cfg.bocpd.max_experts = int(meta.get("max_experts", args.max_experts))
    cfg.bocpd.restart_cooldown = int(meta.get("restart_cooldown", args.restart_cooldown))
    cfg.bocpd.restart_margin = float(meta.get("restart_margin", args.restart_margin))
    cfg.bocpd.use_backdated_restart = bool(meta.get("use_backdated_restart", getattr(cfg.bocpd, "use_backdated_restart", False)))
    cfg.bocpd.controller_name = str(meta.get("controller_name", "none"))
    cfg.bocpd.controller_stat = str(meta.get("controller_stat", "surprise_mean"))
    cfg.bocpd.controller_cs_alpha = float(meta.get("controller_cs_alpha", getattr(args, "controller_sr_alpha", 0.01)))
    cfg.bocpd.controller_cs_min_len = int(meta.get("controller_cs_min_len", getattr(args, "controller_sr_min_len", 2)))
    cfg.bocpd.controller_cs_warmup_batches = int(meta.get("controller_cs_warmup_batches", getattr(args, "controller_sr_warmup_batches", 2)))
    cfg.bocpd.controller_cs_max_active = int(meta.get("controller_cs_max_active", getattr(args, "controller_sr_max_active", 64)))
    cfg.bocpd.controller_cs_clip_low = float(meta.get("controller_cs_clip_low", getattr(args, "controller_sr_clip_low", 0.0)))
    cfg.bocpd.controller_cs_clip_high = float(meta.get("controller_cs_clip_high", getattr(args, "controller_sr_clip_high", 20.0)))
    cfg.bocpd.controller_wcusum_warmup_batches = int(meta.get("controller_wcusum_warmup_batches", getattr(args, "controller_wcusum_warmup_batches", 3)))
    cfg.bocpd.controller_wcusum_window = int(meta.get("controller_wcusum_window", getattr(args, "controller_wcusum_window", 4)))
    cfg.bocpd.controller_wcusum_threshold = float(meta.get("controller_wcusum_threshold", getattr(args, "controller_wcusum_threshold", 3.0)))
    cfg.bocpd.controller_wcusum_kappa = float(meta.get("controller_wcusum_kappa", getattr(args, "controller_wcusum_kappa", 0.5)))
    cfg.bocpd.controller_wcusum_sigma_floor = float(meta.get("controller_wcusum_sigma_floor", getattr(args, "controller_wcusum_sigma_floor", 0.5)))
    cfg.bocpd.hybrid_tau_delta = 0.05
    cfg.bocpd.hybrid_tau_theta = 0.05
    cfg.bocpd.hybrid_tau_full = 0.05
    cfg.bocpd.hybrid_delta_share_rho = 0.75
    cfg.bocpd.hybrid_pf_sigma_mode = "fixed"
    cfg.bocpd.hybrid_sigma_delta_alpha = 1.0
    cfg.bocpd.hybrid_sigma_ema_beta = 0.98
    cfg.bocpd.hybrid_sigma_min = 1e-4
    cfg.bocpd.hybrid_sigma_max = 10.0

    cfg.pf.num_particles = int(args.num_particles)
    cfg.pf.random_walk_scale = float(args.pf_random_walk_scale)
    cfg.pf.resample_ess_ratio = float(args.resample_ess_ratio)

    cfg.model.device = str(args.device)
    cfg.model.dtype = torch.float64
    cfg.model.rho = float(args.rho)
    cfg.model.sigma_eps = float(args.sigma_eps)
    cfg.model.use_discrepancy = bool(meta.get("use_discrepancy", False))
    cfg.model.bocpd_use_discrepancy = bool(meta.get("bocpd_use_discrepancy", True))
    cfg.model.shared_delta_model = str(meta.get("shared_delta_model", "gp"))
    cfg.model.delta_update_mode = str(meta.get("delta_update_mode", "refit"))
    cfg.model.delta_online_min_points = int(args.delta_online_min_points)
    cfg.model.delta_online_init_max_iter = int(args.delta_online_init_max_iter)
    cfg.model.delta_basis_num_features = int(getattr(args, "delta_basis_num_features", args.delta_dynamic_num_features))
    cfg.model.delta_basis_prior_var_scale = float(getattr(args, "delta_basis_prior_var_scale", args.delta_dynamic_prior_var_scale))
    cfg.model.delta_basis_fix_hyper = bool(meta.get("delta_basis_fix_hyper", False))
    cfg.model.delta_dynamic_num_features = int(args.delta_dynamic_num_features)
    cfg.model.delta_dynamic_forgetting = float(args.delta_dynamic_forgetting)
    cfg.model.delta_dynamic_process_noise_scale = float(args.delta_dynamic_process_noise_scale)
    cfg.model.delta_dynamic_prior_var_scale = float(args.delta_dynamic_prior_var_scale)
    cfg.model.delta_dynamic_buffer_max_points = int(args.delta_dynamic_buffer_max_points)
    cfg.model.delta_bpc_lambda = float(args.delta_bpc_lambda)
    cfg.model.delta_bpc_obs_noise_mode = str(meta.get("delta_bpc_obs_noise_mode", "kernel"))
    cfg.model.delta_bpc_predict_add_kernel_noise = bool(meta.get("delta_bpc_predict_add_kernel_noise", True))
    cfg.model.delta_inducing_num_points = int(args.delta_inducing_num_points)
    cfg.model.delta_inducing_init_steps = int(args.delta_inducing_init_steps)
    cfg.model.delta_inducing_update_steps = int(args.delta_inducing_update_steps)
    cfg.model.delta_inducing_lr = float(args.delta_inducing_lr)
    cfg.model.delta_inducing_buffer_max_points = int(args.delta_inducing_buffer_max_points)
    cfg.model.delta_inducing_learn_locations = bool(args.delta_inducing_learn_locations)
    cfg.model.delta_mc_num_inducing_points = int(meta.get("delta_mc_num_inducing_points", args.delta_mc_num_inducing_points))
    cfg.model.delta_mc_num_particles = int(meta.get("delta_mc_num_particles", args.delta_mc_num_particles))
    cfg.model.delta_mc_resample_ess_ratio = float(meta.get("delta_mc_resample_ess_ratio", args.delta_mc_resample_ess_ratio))
    cfg.model.delta_mc_refresh_every = int(meta.get("delta_mc_refresh_every", args.delta_mc_refresh_every))
    cfg.model.delta_mc_include_conditional_var = bool(meta.get("delta_mc_include_conditional_var", args.delta_mc_include_conditional_var))
    cfg.model.delta_kernel.lengthscale = float(args.delta_lengthscale)
    cfg.model.delta_kernel.variance = float(args.delta_variance)
    cfg.model.delta_kernel.noise = float(args.delta_noise)
    return cfg


def safe_float(value: Any) -> float:
    try:
        if value is None:
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def tensor_to_np(value: Any) -> np.ndarray:
    return torch.as_tensor(value).detach().cpu().numpy()


def aggregate_theta(calib: OnlineBayesCalibrator) -> Tuple[float, float, float, float]:
    try:
        if len(calib.bocpd.experts) == 0:
            return float("nan"), float("nan"), float("nan"), float("nan")
        mean_theta, var_theta, lo_theta, hi_theta = calib._aggregate_particles(0.9)
        mean0 = safe_float(mean_theta[0])
        var_arr = torch.as_tensor(var_theta).detach().cpu().numpy()
        var0 = float(var_arr.reshape(-1)[0]) if var_arr.size > 0 else float("nan")
        return mean0, var0, safe_float(lo_theta[0]), safe_float(hi_theta[0])
    except Exception:
        return float("nan"), float("nan"), float("nan"), float("nan")


def discrepancy_summary(calib: OnlineBayesCalibrator) -> Dict[str, float]:
    ess_vals = []
    num_resamples = 0.0
    num_refreshes = 0.0
    for e in getattr(calib.bocpd, "experts", []):
        delta_state = getattr(e, "delta_state", None)
        if delta_state is None:
            continue
        last_ess = getattr(delta_state, "last_ess", None)
        if last_ess is not None:
            arr = np.asarray(torch.as_tensor(last_ess).detach().cpu().numpy(), dtype=float).reshape(-1)
            if arr.size > 0 and np.isfinite(arr).any():
                ess_vals.append(float(np.nanmean(arr)))
        num_resamples += float(getattr(delta_state, "num_resamples", 0.0) or 0.0)
        num_refreshes += float(getattr(delta_state, "num_refreshes", 0.0) or 0.0)
    return dict(delta_ess_mean=float(np.nanmean(ess_vals)) if ess_vals else float("nan"),
                delta_resamples_total=num_resamples,
                delta_refreshes_total=num_refreshes)


def expert_mass_rows(calib: OnlineBayesCalibrator, scenario: str, seed: int, method: str,
                     batch_idx: int) -> List[Dict[str, Any]]:
    rows = []
    for expert_idx, e in enumerate(calib.bocpd.experts):
        try:
            w = e.pf.particles.weights()
            theta = e.pf.particles.theta
            theta_mean = (w[:, None] * theta).sum(dim=0)
            theta_val = safe_float(theta_mean[0])
        except Exception:
            theta_val = float("nan")
        log_mass = safe_float(getattr(e, "log_mass", float("nan")))
        rows.append(dict(scenario=scenario, seed=int(seed), method=method, batch_idx=int(batch_idx),
                         expert_idx=int(expert_idx), run_length=int(getattr(e, "run_length", -1)),
                         log_mass=log_mass, mass=float(math.exp(log_mass)) if np.isfinite(log_mass) else 0.0,
                         theta_mean=theta_val))
    return rows


def evidence_summary(pred_complete: Optional[Dict[str, Any]], experts: Sequence[Any]) -> Dict[str, float]:
    out = dict(evidence_anchor_logp=float("nan"), evidence_best_logp=float("nan"),
               evidence_best_competitor_logp=float("nan"), evidence_margin=float("nan"),
               evidence_best_minus_anchor=float("nan"))
    if not pred_complete:
        return out
    items = list(pred_complete.get("experts_logpred", []))
    if len(items) == 0:
        return out
    logp = np.asarray([safe_float(item.get("logp", float("nan"))) for item in items], dtype=float)
    log_mass = np.asarray([safe_float(item.get("log_mass", float("nan"))) for item in items], dtype=float)
    if not np.isfinite(log_mass).any() and len(experts) == len(items):
        log_mass = np.asarray([safe_float(getattr(e, "log_mass", float("nan"))) for e in experts], dtype=float)
    if not np.isfinite(logp).any():
        return out
    anchor_idx = int(np.nanargmax(log_mass)) if np.isfinite(log_mass).any() else 0
    best_idx = int(np.nanargmax(logp))
    anchor_logp = float(logp[anchor_idx]) if anchor_idx < logp.size else float("nan")
    best_logp = float(logp[best_idx])
    if logp.size > 1 and anchor_idx < logp.size:
        mask = np.ones(logp.size, dtype=bool)
        mask[anchor_idx] = False
        comp_logp = float(np.nanmax(logp[mask]))
    else:
        comp_logp = float("nan")
    out.update(evidence_anchor_logp=anchor_logp, evidence_best_logp=best_logp,
               evidence_best_competitor_logp=comp_logp,
               evidence_margin=comp_logp - anchor_logp if np.isfinite(comp_logp) and np.isfinite(anchor_logp) else float("nan"),
               evidence_best_minus_anchor=best_logp - anchor_logp if np.isfinite(best_logp) and np.isfinite(anchor_logp) else float("nan"))
    return out


def current_segment_start(restart_flags: Sequence[bool], end_idx: int) -> int:
    start = 0
    for i in range(0, int(end_idx) + 1):
        if i < len(restart_flags) and bool(restart_flags[i]):
            start = i
    return int(start)


def residual_matrices(X_batches: Sequence[np.ndarray], Y_batches: Sequence[np.ndarray],
                      theta_post: Sequence[float], restart_flags: Sequence[bool],
                      end_idx: int) -> Dict[str, Any]:
    if len(X_batches) == 0 or end_idx < 0:
        return dict(start_idx=0, end_idx=end_idx, fixed_endpoint=None, frozen_anchor=None)
    end_idx = min(int(end_idx), len(X_batches) - 1)
    start_idx = current_segment_start(restart_flags, end_idx)
    theta_fixed = safe_float(theta_post[end_idx]) if end_idx < len(theta_post) else float("nan")
    fixed_rows = []
    frozen_rows = []
    for u in range(start_idx, end_idx + 1):
        X_u = np.asarray(X_batches[u], dtype=float)
        Y_u = np.asarray(Y_batches[u], dtype=float).reshape(-1)
        fixed_rows.append(Y_u - simulator_np(X_u, theta_fixed) if np.isfinite(theta_fixed) else np.full_like(Y_u, np.nan, dtype=float))
        theta_u = safe_float(theta_post[u]) if u < len(theta_post) else float("nan")
        frozen_rows.append(Y_u - simulator_np(X_u, theta_u) if np.isfinite(theta_u) else np.full_like(Y_u, np.nan, dtype=float))
    return dict(start_idx=start_idx, end_idx=end_idx,
                fixed_endpoint=np.vstack(fixed_rows) if fixed_rows else None,
                frozen_anchor=np.vstack(frozen_rows) if frozen_rows else None)


def heterogeneity(residual_matrix: Optional[np.ndarray]) -> float:
    if residual_matrix is None:
        return float("nan")
    R = np.asarray(residual_matrix, dtype=float)
    if R.ndim != 2 or R.shape[0] < 2 or not np.isfinite(R).any():
        return float("nan")
    center = np.nanmean(R, axis=0, keepdims=True)
    return float(np.nanmean((R - center) ** 2))


def slug(text: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(text))


def run_single_method(args: argparse.Namespace, spec: ScenarioSpec, seed: int, method: str,
                      meta: Dict[str, Any], raw_dir: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    print(f"  -> {spec.name} seed={seed} method={method}")
    method_offset = sum(ord(c) for c in method) % 10000
    torch.manual_seed(int(seed) + method_offset)
    np.random.seed(int(seed) + method_offset)

    cfg = build_calibration_config(args, meta)
    emulator = DeterministicSimulator(func=simulator_torch, enable_autograd=True)

    def prior_sampler(N: int, **_: Any) -> torch.Tensor:
        return torch.rand(int(N), 1) * 3.0

    calib = OnlineBayesCalibrator(cfg, emulator, prior_sampler, notify_on_restart=False)
    roll = RollingStats(window=50)
    stream = ThetaPathPhysicalStream(spec, seed=seed, shuffle_x=True)

    batch_rows: List[Dict[str, Any]] = []
    evidence_rows: List[Dict[str, Any]] = []
    runlength_rows: List[Dict[str, Any]] = []
    X_batches: List[np.ndarray] = []
    Y_batches: List[np.ndarray] = []
    y_noiseless_batches: List[np.ndarray] = []
    sim_true_batches: List[np.ndarray] = []
    delta_star_batches: List[np.ndarray] = []
    pred_mu_batches: List[np.ndarray] = []
    pred_mu_sim_batches: List[np.ndarray] = []
    pred_var_batches: List[np.ndarray] = []
    theta_pred_hist: List[float] = []
    theta_post_hist: List[float] = []
    theta_var_hist: List[float] = []
    restart_flags: List[bool] = []
    restart_modes: List[str] = []
    raw_recs: List[Dict[str, Any]] = []

    t0 = time()
    for batch_idx in range(spec.n_batches):
        Xb, Yb, info = stream.next()
        X_np = np.asarray(info["X"], dtype=float)
        Y_np = np.asarray(info["y"], dtype=float).reshape(-1)
        y_noiseless = np.asarray(info["y_noiseless"], dtype=float).reshape(-1)
        sim_true = np.asarray(info["sim_true"], dtype=float).reshape(-1)
        delta_star = np.asarray(info["delta_star"], dtype=float).reshape(-1)

        theta_pred, theta_var_pred, _, _ = aggregate_theta(calib)
        pred_available = len(calib.bocpd.experts) > 0
        evidence = evidence_summary(None, [])
        pred_mse = pred_noiseless_mse = theta_mismatch = delta_mismatch = crps = float("nan")
        pred_mu_np = np.full_like(Y_np, np.nan, dtype=float)
        pred_mu_sim_np = np.full_like(Y_np, np.nan, dtype=float)
        pred_var_np = np.full_like(Y_np, np.nan, dtype=float)
        if pred_available:
            pred = calib.predict_batch(Xb)
            experts_pre = list(calib.bocpd.experts)
            pred_complete = calib.predict_complete(Xb, Yb)
            evidence = evidence_summary(pred_complete, experts_pre)
            pred_mu_np = tensor_to_np(pred["mu"]).reshape(-1)
            pred_mu_sim_np = tensor_to_np(pred["mu_sim"]).reshape(-1)
            pred_var_np = tensor_to_np(pred["var"]).reshape(-1)
            delta_hat_np = pred_mu_np - pred_mu_sim_np
            pred_mse = float(np.mean((Y_np - pred_mu_np) ** 2))
            pred_noiseless_mse = float(np.mean((y_noiseless - pred_mu_np) ** 2))
            theta_mismatch = float(np.mean((sim_true - pred_mu_sim_np) ** 2))
            delta_mismatch = float(np.mean((delta_star - delta_hat_np) ** 2))
            crps = float(crps_gaussian(torch.tensor(pred_mu_np, dtype=torch.float64),
                                       torch.tensor(np.clip(pred_var_np, 1e-12, None), dtype=torch.float64),
                                       torch.tensor(Y_np, dtype=torch.float64)).mean().item())
            evidence_rows.append(dict(scenario=spec.name, seed=int(seed), method=method,
                                      batch_idx=int(batch_idx), t_obs=int(info["t_obs"]), **evidence))

        rec = calib.step_batch(Xb, Yb, verbose=False)
        did_restart = bool(rec.get("did_restart", False))
        restart_mode = rec.get("restart_mode", None)
        if restart_mode is None:
            restart_mode = "full" if did_restart else "none"
        restart_mode = str(restart_mode)
        restart_flags.append(did_restart)
        restart_modes.append(restart_mode)

        dll = rec.get("delta_ll_pair", None)
        if dll is not None and np.isfinite(safe_float(dll)):
            roll.update(safe_float(dll))

        theta_post, theta_var_post, lo_theta, hi_theta = aggregate_theta(calib)
        theta_pred_hist.append(theta_pred)
        theta_post_hist.append(theta_post)
        theta_var_hist.append(theta_var_post)
        anchor_movement = (abs(theta_post_hist[-1] - theta_post_hist[-2])
                           if len(theta_post_hist) >= 2 and np.isfinite(theta_post_hist[-1]) and np.isfinite(theta_post_hist[-2])
                           else float("nan"))

        X_batches.append(X_np.copy())
        Y_batches.append(Y_np.copy())
        y_noiseless_batches.append(y_noiseless.copy())
        sim_true_batches.append(sim_true.copy())
        delta_star_batches.append(delta_star.copy())
        pred_mu_batches.append(pred_mu_np.copy())
        pred_mu_sim_batches.append(pred_mu_sim_np.copy())
        pred_var_batches.append(pred_var_np.copy())

        residual_state = residual_matrices(X_batches, Y_batches, theta_post_hist, restart_flags, batch_idx)
        H_fixed = heterogeneity(residual_state.get("fixed_endpoint"))
        H_frozen = heterogeneity(residual_state.get("frozen_anchor"))
        delta_diag = discrepancy_summary(calib)
        runlength_rows.extend(expert_mass_rows(calib, spec.name, seed, method, batch_idx))
        raw_recs.append(dict(rec={k: v for k, v in rec.items() if k != "pf_diags"}, pf_diags=rec.get("pf_diags", None)))

        batch_rows.append(dict(scenario=spec.name, seed=int(seed), method=method, batch_idx=int(batch_idx),
                               t_obs=int(info["t_obs"]), theta_star=float(info["theta_star"]),
                               theta_hat_pred=theta_pred, theta_hat_post=theta_post,
                               theta_var_pred=theta_var_pred, theta_var_post=theta_var_post,
                               theta_lo=lo_theta, theta_hi=hi_theta, pred_mse=pred_mse,
                               pred_noiseless_mse=pred_noiseless_mse, theta_mismatch=theta_mismatch,
                               delta_mismatch=delta_mismatch, y_crps=crps, did_restart=did_restart,
                               restart_mode=restart_mode, anchor_movement=anchor_movement,
                               resid_heterogeneity_fixed_endpoint=H_fixed,
                               resid_heterogeneity_frozen_anchor=H_frozen,
                               active_segment_start=int(residual_state.get("start_idx", 0)),
                               n_active_segment_batches=int(batch_idx - int(residual_state.get("start_idx", 0)) + 1),
                               n_experts=int(len(calib.bocpd.experts)), delta_ll_pair=safe_float(dll),
                               delta_ll_roll_mean=roll.mean(), delta_ll_roll_std=roll.std(),
                               h_log=safe_float(rec.get("h_log", float("nan"))),
                               log_odds_mass=safe_float(rec.get("log_odds_mass", float("nan"))),
                               anchor_rl=safe_float(rec.get("anchor_rl", float("nan"))),
                               cand_rl=safe_float(rec.get("cand_rl", float("nan"))),
                               evidence_margin=evidence["evidence_margin"],
                               evidence_best_minus_anchor=evidence["evidence_best_minus_anchor"],
                               delta_ess_mean=delta_diag["delta_ess_mean"],
                               delta_resamples_total=delta_diag["delta_resamples_total"],
                               delta_refreshes_total=delta_diag["delta_refreshes_total"],
                               controller_name=str(rec.get("controller_name", meta.get("controller_name", "bocpd"))),
                               controller_stat=str(rec.get("controller_stat", meta.get("controller_stat", "none"))),
                               controller_pre_update_score=safe_float(rec.get("controller_pre_update_score", float("nan"))),
                               controller_pre_update_score_clipped=safe_float(rec.get("controller_pre_update_score_clipped", float("nan"))),
                               controller_pre_update_surprise=safe_float(rec.get("controller_pre_update_surprise", float("nan"))),
                               controller_pre_update_surprise_clipped=safe_float(rec.get("controller_pre_update_surprise_clipped", float("nan"))),
                               controller_pre_update_mean_logpred=safe_float(rec.get("controller_pre_update_mean_logpred", float("nan"))),
                               controller_pre_update_calibration_gap=safe_float(rec.get("controller_pre_update_calibration_gap", float("nan"))),
                               controller_wcusum_stat=safe_float(rec.get("controller_wcusum_stat", float("nan"))),
                               controller_wcusum_baseline_mean=safe_float(rec.get("controller_wcusum_baseline_mean", float("nan"))),
                               controller_wcusum_baseline_sigma=safe_float(rec.get("controller_wcusum_baseline_sigma", float("nan"))),
                               controller_cs_lower=safe_float(rec.get("controller_cs_lower", float("nan"))),
                               controller_cs_upper=safe_float(rec.get("controller_cs_upper", float("nan"))),
                               controller_num_active=safe_float(rec.get("controller_num_active", float("nan"))),
                               delta_update_mode=str(meta.get("delta_update_mode", "refit")),
                               particle_delta_mode=str(meta.get("particle_delta_mode", "shared_gp"))))

    snapshot_batches = sorted(set([spec.n_batches - 1] + [min(spec.n_batches - 1, b + 3) for b in spec.event_batches]))
    residual_snapshots: Dict[str, Any] = {}
    for end_idx in snapshot_batches:
        residual_snapshots[f"batch_{int(end_idx)}"] = residual_matrices(X_batches, Y_batches, theta_post_hist, restart_flags, int(end_idx))
    residual_snapshots["final"] = residual_matrices(X_batches, Y_batches, theta_post_hist, restart_flags, spec.n_batches - 1)

    raw = dict(scenario=asdict(spec), seed=int(seed), method=method, method_meta=dict(meta),
               elapsed_sec=float(time() - t0), X_batches=np.asarray(X_batches, dtype=object),
               Y_batches=np.asarray(Y_batches, dtype=object),
               y_noiseless_batches=np.asarray(y_noiseless_batches, dtype=object),
               sim_true_batches=np.asarray(sim_true_batches, dtype=object),
               delta_star_batches=np.asarray(delta_star_batches, dtype=object),
               pred_mu_batches=np.asarray(pred_mu_batches, dtype=object),
               pred_mu_sim_batches=np.asarray(pred_mu_sim_batches, dtype=object),
               pred_var_batches=np.asarray(pred_var_batches, dtype=object),
               theta_pred=np.asarray(theta_pred_hist, dtype=float),
               theta_post=np.asarray(theta_post_hist, dtype=float),
               theta_var=np.asarray(theta_var_hist, dtype=float),
               restart_flags=np.asarray(restart_flags, dtype=bool),
               restart_modes=np.asarray(restart_modes, dtype=object),
               batch_rows=batch_rows, evidence_rows=evidence_rows,
               runlength_rows=runlength_rows, residual_snapshots=residual_snapshots,
               raw_update_records=raw_recs)
    raw_path = raw_dir / f"{slug(spec.name)}_seed{seed}_{slug(method)}.pt"
    torch.save(raw, raw_path)
    return batch_rows, evidence_rows, runlength_rows, raw


def save_summary_tables(batch_df: pd.DataFrame, out_dir: Path) -> None:
    if batch_df.empty:
        return
    metric_cols = ["pred_mse", "pred_noiseless_mse", "theta_mismatch", "delta_mismatch", "y_crps",
                   "anchor_movement", "resid_heterogeneity_fixed_endpoint",
                   "resid_heterogeneity_frozen_anchor", "evidence_margin",
                   "delta_ess_mean", "delta_resamples_total", "delta_refreshes_total"]
    summary = batch_df.groupby(["scenario", "method"], as_index=False)[metric_cols].mean(numeric_only=True)
    restarts = batch_df.groupby(["scenario", "seed", "method"], as_index=False)["did_restart"].sum()
    restart_summary = restarts.groupby(["scenario", "method"], as_index=False)["did_restart"].agg(["mean", "std"]).reset_index()
    summary.to_csv(out_dir / "mechanism_metric_summary.csv", index=False)
    restarts.to_csv(out_dir / "mechanism_restart_counts_by_run.csv", index=False)
    restart_summary.to_csv(out_dir / "mechanism_restart_count_summary.csv", index=False)


def run_all(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[Tuple[str, int, str], Dict[str, Any]], pd.DataFrame]:
    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw_runs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    methods = select_methods(args)
    mapper = build_phi2_theta_map(theta_grid_size=int(args.oracle_grid_size),
                                  phi2_grid_size=int(args.phi2_grid_size),
                                  phi2_min=float(args.phi2_min), phi2_max=float(args.phi2_max))
    all_batch_rows: List[Dict[str, Any]] = []
    all_evidence_rows: List[Dict[str, Any]] = []
    all_runlength_rows: List[Dict[str, Any]] = []
    event_rows: List[Dict[str, Any]] = []
    raw_results: Dict[Tuple[str, int, str], Dict[str, Any]] = {}

    for seed in args.seeds:
        scenario_specs: List[ScenarioSpec] = []
        if "sudden" in args.scenarios:
            scenario_specs.append(make_sudden_spec(args, mapper))
        if "slope" in args.scenarios:
            scenario_specs.append(make_slope_spec(args, mapper))
        if "random_walk" in args.scenarios:
            scenario_specs.append(make_random_walk_spec(args, mapper, int(seed)))
        for spec in scenario_specs:
            for event_batch in spec.event_batches:
                event_rows.append(dict(scenario=spec.name, seed=int(seed), event_batch=int(event_batch), event_kind=spec.event_kind))
            for method, meta in methods.items():
                batch_rows, evidence_rows, runlength_rows, raw = run_single_method(args, spec, int(seed), method, meta, raw_dir)
                all_batch_rows.extend(batch_rows)
                all_evidence_rows.extend(evidence_rows)
                all_runlength_rows.extend(runlength_rows)
                raw_results[(spec.name, int(seed), method)] = raw

    batch_df = pd.DataFrame(all_batch_rows)
    evidence_df = pd.DataFrame(all_evidence_rows)
    runlength_df = pd.DataFrame(all_runlength_rows)
    event_df = pd.DataFrame(event_rows)
    batch_df.to_csv(out_dir / "mechanism_batch_records.csv", index=False)
    evidence_df.to_csv(out_dir / "mechanism_evidence_records.csv", index=False)
    runlength_df.to_csv(out_dir / "mechanism_runlength_records.csv", index=False)
    event_df.to_csv(out_dir / "mechanism_event_records.csv", index=False)
    torch.save(raw_results, out_dir / "mechanism_all_raw_results.pt")
    save_summary_tables(batch_df, out_dir)
    return batch_df, evidence_df, runlength_df, raw_results, event_df


def method_order(methods: Sequence[str]) -> List[str]:
    preferred = ["half_refit", "half_basisRefit", "half_basisRefitFixedHyper", "half_online", "half_onlineInducing", "half_onlineDynamic", "shared_onlineBPC", "shared_onlineBPC_exact", "shared_onlineBPC_exact_sigmaObs", "shared_onlineBPC_fixedSupport_sigmaObs", "shared_onlineBPC_exact_refitHyper", "shared_onlineBPC_proxyRefitHyper", "shared_onlineBPC_proxyRefitHyper_sigmaObs", "shared_onlineBPC_proxyStableMean", "shared_onlineBPC_proxyStableMean_sigmaObs", "DOC_SharedMCInducing", "DOC_SharedMCInducing_Refresh", "DOC_ParticleMCInducing", "DOC_ParticleMCInducing_Refresh", "WardPF_BOCPD", "particleGP_refit", "particleGP_online", "particleGP_onlineDynamic", "particleGP_onlineBPC", "particleGP_onlineBPC_exact", "particleGP_fixedSupport_onlineBPC_sigmaObs"]
    return [m for m in preferred if m in set(methods)] + [m for m in methods if m not in preferred]


def add_event_lines(ax: plt.Axes, event_batches: Sequence[int], color: str = "black") -> None:
    for eb in event_batches:
        ax.axvline(int(eb), color=color, ls="--", lw=1.0, alpha=0.45)


def add_restart_lines(ax: plt.Axes, sub: pd.DataFrame) -> None:
    if "did_restart" not in sub:
        return
    for _, row in sub[sub["did_restart"].astype(bool)].iterrows():
        ax.axvline(int(row["batch_idx"]), color="tab:red", lw=0.8, alpha=0.35)


def plot_temporal_decomposition(batch_df: pd.DataFrame, event_df: pd.DataFrame, out_dir: Path, scenario: str, seed: int) -> None:
    sub_all = batch_df[(batch_df["scenario"] == scenario) & (batch_df["seed"] == seed)].copy()
    if sub_all.empty:
        return
    methods = method_order(list(sub_all["method"].unique()))
    events = event_df[(event_df["scenario"] == scenario) & (event_df["seed"] == seed)]["event_batch"].tolist()
    fig, axes = plt.subplots(len(methods), 1, figsize=(11, max(2.2, 2.2 * len(methods))), sharex=True)
    if len(methods) == 1:
        axes = [axes]
    metrics = [("pred_mse", "prediction error"), ("theta_mismatch", "simulator mismatch"),
               ("delta_mismatch", "discrepancy mismatch"), ("anchor_movement", "anchor movement")]
    for ax, method in zip(axes, methods):
        sub = sub_all[sub_all["method"] == method].sort_values("batch_idx")
        x = sub["batch_idx"].to_numpy()
        for col, label in metrics:
            y = sub[col].to_numpy(dtype=float)
            y = np.where(np.isfinite(y), np.maximum(y, 1e-12), np.nan)
            ax.plot(x, y, marker="o", ms=2.5, lw=1.0, label=label)
        ax.set_yscale("log")
        ax.set_ylabel(method)
        add_event_lines(ax, events)
        add_restart_lines(ax, sub)
        ax.grid(True, alpha=0.2)
        ax.legend(loc="upper right", fontsize=7, ncol=2)
    axes[-1].set_xlabel("batch index")
    fig.suptitle(f"Event-aligned temporal mechanism diagnostics: {scenario}, seed {seed}")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_dir / f"temporal_decomposition_{scenario}_seed{seed}.png", dpi=250)
    plt.close(fig)


def plot_evidence_margin(batch_df: pd.DataFrame, event_df: pd.DataFrame, out_dir: Path, scenario: str, seed: int) -> None:
    sub_all = batch_df[(batch_df["scenario"] == scenario) & (batch_df["seed"] == seed)].copy()
    if sub_all.empty:
        return
    methods = method_order(list(sub_all["method"].unique()))
    events = event_df[(event_df["scenario"] == scenario) & (event_df["seed"] == seed)]["event_batch"].tolist()
    fig, axes = plt.subplots(len(methods), 1, figsize=(11, max(2.0, 2.0 * len(methods))), sharex=True)
    if len(methods) == 1:
        axes = [axes]
    for ax, method in zip(axes, methods):
        sub = sub_all[sub_all["method"] == method].sort_values("batch_idx")
        ax.plot(sub["batch_idx"], sub["evidence_margin"], lw=1.0, marker="o", ms=2.5, label="best competitor - anchor")
        ax.plot(sub["batch_idx"], sub["evidence_best_minus_anchor"], lw=1.0, alpha=0.65, label="best - anchor")
        ax.axhline(0.0, color="black", lw=0.8, alpha=0.5)
        add_event_lines(ax, events)
        add_restart_lines(ax, sub)
        ax.set_ylabel(method)
        ax.grid(True, alpha=0.2)
        ax.legend(loc="upper right", fontsize=7)
    axes[-1].set_xlabel("batch index")
    fig.suptitle(f"Expert evidence margin: {scenario}, seed {seed}")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_dir / f"evidence_margin_{scenario}_seed{seed}.png", dpi=250)
    plt.close(fig)


def plot_residual_heterogeneity(batch_df: pd.DataFrame, event_df: pd.DataFrame, out_dir: Path, scenario: str, seed: int) -> None:
    sub_all = batch_df[(batch_df["scenario"] == scenario) & (batch_df["seed"] == seed)].copy()
    if sub_all.empty:
        return
    methods = method_order(list(sub_all["method"].unique()))
    events = event_df[(event_df["scenario"] == scenario) & (event_df["seed"] == seed)]["event_batch"].tolist()
    fig, axes = plt.subplots(len(methods), 1, figsize=(11, max(2.0, 2.0 * len(methods))), sharex=True)
    if len(methods) == 1:
        axes = [axes]
    for ax, method in zip(axes, methods):
        sub = sub_all[sub_all["method"] == method].sort_values("batch_idx")
        ax.plot(sub["batch_idx"], sub["resid_heterogeneity_fixed_endpoint"], lw=1.1, label="fixed-endpoint target")
        ax.plot(sub["batch_idx"], sub["resid_heterogeneity_frozen_anchor"], lw=1.1, label="frozen-anchor target")
        add_event_lines(ax, events)
        add_restart_lines(ax, sub)
        ax.set_ylabel(method)
        ax.grid(True, alpha=0.2)
        ax.legend(loc="upper right", fontsize=7)
    axes[-1].set_xlabel("batch index")
    fig.suptitle(f"Residual-target heterogeneity proxy: {scenario}, seed {seed}")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_dir / f"residual_heterogeneity_{scenario}_seed{seed}.png", dpi=250)
    plt.close(fig)


def plot_theta_tracking(batch_df: pd.DataFrame, event_df: pd.DataFrame, out_dir: Path, scenario: str, seed: int) -> None:
    sub_all = batch_df[(batch_df["scenario"] == scenario) & (batch_df["seed"] == seed)].copy()
    if sub_all.empty:
        return
    methods = method_order(list(sub_all["method"].unique()))
    events = event_df[(event_df["scenario"] == scenario) & (event_df["seed"] == seed)]["event_batch"].tolist()
    fig, axes = plt.subplots(len(methods), 1, figsize=(11, max(2.3, 2.4 * len(methods))), sharex=True)
    if len(methods) == 1:
        axes = [axes]
    for ax, method in zip(axes, methods):
        sub = sub_all[sub_all["method"] == method].sort_values("batch_idx")
        x = sub["batch_idx"].to_numpy(dtype=int)
        theta_star = sub["theta_star"].to_numpy(dtype=float)
        theta_hat = sub["theta_hat_post"].to_numpy(dtype=float)
        theta_lo = sub["theta_lo"].to_numpy(dtype=float)
        theta_hi = sub["theta_hi"].to_numpy(dtype=float)
        ax.plot(x, theta_star, color="black", lw=1.4, label="theta ground truth")
        ax.plot(x, theta_hat, color="tab:blue", lw=1.4, marker="o", ms=2.5, label="theta estimate")
        if np.isfinite(theta_lo).any() and np.isfinite(theta_hi).any():
            ax.fill_between(x, theta_lo, theta_hi, color="tab:blue", alpha=0.15, label="90% band")
        add_event_lines(ax, events)
        add_restart_lines(ax, sub)
        ax.set_ylabel(method)
        ax.grid(True, alpha=0.2)
        ax.legend(loc="best", fontsize=7)
    axes[-1].set_xlabel("batch index")
    fig.suptitle(f"Theta tracking: {scenario}, seed {seed}")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_dir / f"theta_tracking_{scenario}_seed{seed}.png", dpi=250)
    plt.close(fig)


def _batch_mean_series(items: Sequence[Any]) -> np.ndarray:
    vals = []
    for item in items:
        arr = np.asarray(item, dtype=float).reshape(-1)
        finite = arr[np.isfinite(arr)]
        vals.append(float(finite.mean()) if finite.size > 0 else float("nan"))
    return np.asarray(vals, dtype=float)


def plot_y_tracking(raw_results: Dict[Tuple[str, int, str], Dict[str, Any]], event_df: pd.DataFrame, out_dir: Path, scenario: str, seed: int) -> None:
    keys = [key for key in raw_results.keys() if key[0] == scenario and int(key[1]) == int(seed)]
    if not keys:
        return
    methods = method_order([str(key[2]) for key in keys])
    events = event_df[(event_df["scenario"] == scenario) & (event_df["seed"] == seed)]["event_batch"].tolist()
    fig, axes = plt.subplots(len(methods), 1, figsize=(11, max(2.3, 2.4 * len(methods))), sharex=True)
    if len(methods) == 1:
        axes = [axes]
    for ax, method in zip(axes, methods):
        raw = raw_results.get((scenario, int(seed), method))
        if raw is None:
            ax.set_visible(False)
            continue
        y_mean = _batch_mean_series(raw.get("Y_batches", []))
        y_noiseless_mean = _batch_mean_series(raw.get("y_noiseless_batches", []))
        pred_mean = _batch_mean_series(raw.get("pred_mu_batches", []))
        x = np.arange(len(y_mean), dtype=int)
        ax.plot(x, y_mean, color="black", lw=1.4, label="y batch mean")
        if np.isfinite(y_noiseless_mean).any():
            ax.plot(x, y_noiseless_mean, color="black", lw=1.0, ls="--", alpha=0.7, label="y noiseless mean")
        ax.plot(x, pred_mean, color="tab:green", lw=1.4, marker="o", ms=2.5, label="predicted batch mean")
        batch_df_method = pd.DataFrame(raw.get("batch_rows", []))
        add_event_lines(ax, events)
        if not batch_df_method.empty:
            add_restart_lines(ax, batch_df_method)
        ax.set_ylabel(method)
        ax.grid(True, alpha=0.2)
        ax.legend(loc="best", fontsize=7)
    axes[-1].set_xlabel("batch index")
    fig.suptitle(f"Y tracking: {scenario}, seed {seed}")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_dir / f"y_tracking_{scenario}_seed{seed}.png", dpi=250)
    plt.close(fig)


def plot_runlength_heatmap(runlength_df: pd.DataFrame, out_dir: Path, scenario: str, seed: int, method: str, max_run_length: int) -> None:
    sub = runlength_df[(runlength_df["scenario"] == scenario) & (runlength_df["seed"] == seed) & (runlength_df["method"] == method)].copy()
    if sub.empty:
        return
    max_batch = int(sub["batch_idx"].max())
    max_rl = int(min(max_run_length, max(1, sub["run_length"].max())))
    mat = np.zeros((max_rl + 1, max_batch + 1), dtype=float)
    for _, row in sub.iterrows():
        rl = int(row["run_length"])
        b = int(row["batch_idx"])
        if 0 <= rl <= max_rl and 0 <= b <= max_batch:
            mat[rl, b] += float(row["mass"])
    fig, ax = plt.subplots(figsize=(10, 4))
    cmap = "mako" if "mako" in plt.colormaps() else "viridis"
    im = ax.imshow(mat, origin="lower", aspect="auto", interpolation="nearest", cmap=cmap)
    ax.set_title(f"Run-length posterior heatmap: {scenario}, seed {seed}, {method}")
    ax.set_xlabel("batch index")
    ax.set_ylabel("run length")
    fig.colorbar(im, ax=ax, label="posterior mass")
    fig.tight_layout()
    fig.savefig(out_dir / f"runlength_heatmap_{scenario}_seed{seed}_{slug(method)}.png", dpi=250)
    plt.close(fig)


def plot_residual_coherence(raw: Dict[str, Any], out_dir: Path) -> None:
    scenario = raw["scenario"]["name"]
    seed = int(raw["seed"])
    method = str(raw["method"])
    snap = raw.get("residual_snapshots", {}).get("final")
    if not snap:
        return
    fixed = snap.get("fixed_endpoint")
    frozen = snap.get("frozen_anchor")
    if fixed is None or frozen is None:
        return
    fixed = np.asarray(fixed, dtype=float)
    frozen = np.asarray(frozen, dtype=float)
    if fixed.size == 0 or frozen.size == 0:
        return
    vmax = np.nanpercentile(np.abs(np.concatenate([fixed.reshape(-1), frozen.reshape(-1)])), 97.5)
    vmax = max(float(vmax), 1e-6)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, mat, title in [(axes[0], fixed, "fixed-endpoint residual target"),
                           (axes[1], frozen, "frozen-anchor residual target")]:
        im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("within-batch x order")
    axes[0].set_ylabel("batch in active segment")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, label="residual value")
    fig.suptitle(f"Residual coherence proxy: {scenario}, seed {seed}, {method}")
    fig.savefig(out_dir / f"residual_coherence_{scenario}_seed{seed}_{slug(method)}.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def collect_aligned_values(df: pd.DataFrame, events: pd.DataFrame, method: str, metric: str, window: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rel = np.arange(-int(window), int(window) + 1)
    rows = []
    sub_method = df[df["method"] == method]
    for _, event in events.iterrows():
        sub = sub_method[(sub_method["scenario"] == event["scenario"]) & (sub_method["seed"] == event["seed"])]
        if sub.empty:
            continue
        by_batch = sub.set_index("batch_idx")[metric]
        vals = [float(by_batch.loc[int(event["event_batch"] + r)]) if int(event["event_batch"] + r) in by_batch.index else float("nan") for r in rel]
        rows.append(vals)
    if not rows:
        return rel, np.full_like(rel, np.nan, dtype=float), np.full_like(rel, np.nan, dtype=float)
    arr = np.asarray(rows, dtype=float)
    mean = np.full(arr.shape[1], np.nan, dtype=float)
    std = np.full(arr.shape[1], np.nan, dtype=float)
    for j in range(arr.shape[1]):
        col = arr[:, j]
        finite = np.isfinite(col)
        if finite.any():
            mean[j] = float(col[finite].mean())
            std[j] = float(col[finite].std())
    return rel, mean, std


def plot_event_aligned(batch_df: pd.DataFrame, event_df: pd.DataFrame, out_dir: Path, scenario: str, window: int) -> None:
    events = event_df[event_df["scenario"] == scenario].copy()
    df = batch_df[batch_df["scenario"] == scenario].copy()
    if events.empty or df.empty:
        return
    methods = method_order(list(df["method"].unique()))
    metrics = [("pred_mse", "prediction error"), ("theta_mismatch", "simulator mismatch"), ("delta_mismatch", "discrepancy mismatch")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4), sharex=True)
    if len(metrics) == 1:
        axes = [axes]
    for ax, (metric, title) in zip(axes, metrics):
        for method in methods:
            rel, mean, std = collect_aligned_values(df, events, method, metric, window)
            ax.plot(rel, mean, lw=1.4, marker="o", ms=2.5, label=method)
            ax.fill_between(rel, mean - std, mean + std, alpha=0.12)
        ax.axvline(0, color="black", ls="--", lw=1.0)
        ax.set_title(title)
        ax.set_xlabel("batches relative to event")
        ax.grid(True, alpha=0.2)
    axes[0].set_ylabel("mean over events/seeds")
    axes[-1].legend(fontsize=7, loc="best")
    fig.suptitle(f"Event-aligned decomposition: {scenario}")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_dir / f"event_aligned_decomposition_{scenario}.png", dpi=250)
    plt.close(fig)


def plot_restart_centered_stack(batch_df: pd.DataFrame, out_dir: Path, scenario: str, window: int) -> None:
    df = batch_df[batch_df["scenario"] == scenario].copy()
    if df.empty:
        return
    restart_events = df[df["did_restart"].astype(bool)][["scenario", "seed", "method", "batch_idx"]].rename(columns={"batch_idx": "event_batch"})
    if restart_events.empty:
        return
    methods = method_order(list(df["method"].unique()))
    metrics = [("pred_mse", "prediction error"), ("theta_mismatch", "simulator mismatch"),
               ("delta_mismatch", "discrepancy mismatch"), ("evidence_margin", "evidence margin")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4), sharex=True)
    if len(metrics) == 1:
        axes = [axes]
    for ax, (metric, title) in zip(axes, metrics):
        for method in methods:
            events_m = restart_events[restart_events["method"] == method]
            rel, mean, std = collect_aligned_values(df, events_m, method, metric, window)
            if np.isfinite(mean).any():
                ax.plot(rel, mean, lw=1.4, marker="o", ms=2.5, label=method)
                ax.fill_between(rel, mean - std, mean + std, alpha=0.12)
        ax.axvline(0, color="tab:red", ls="--", lw=1.0)
        ax.set_title(title)
        ax.set_xlabel("batches relative to restart")
        ax.grid(True, alpha=0.2)
    axes[0].set_ylabel("mean over restarts")
    axes[-1].legend(fontsize=7, loc="best")
    fig.suptitle(f"Restart-centered mechanism stack: {scenario}")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_dir / f"restart_centered_stack_{scenario}.png", dpi=250)
    plt.close(fig)


def make_plots(args: argparse.Namespace, batch_df: pd.DataFrame, evidence_df: pd.DataFrame,
               runlength_df: pd.DataFrame, raw_results: Dict[Tuple[str, int, str], Dict[str, Any]],
               event_df: pd.DataFrame) -> None:
    del evidence_df
    out_dir = Path(args.out_dir)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    seeds_to_plot = list(args.seeds) if args.plot_all_seeds else [int(args.seeds[0])]
    for scenario in sorted(batch_df["scenario"].unique()):
        plot_event_aligned(batch_df, event_df, plot_dir, scenario, int(args.event_window))
        plot_restart_centered_stack(batch_df, plot_dir, scenario, int(args.restart_window))
        for seed in seeds_to_plot:
            plot_temporal_decomposition(batch_df, event_df, plot_dir, scenario, int(seed))
            plot_evidence_margin(batch_df, event_df, plot_dir, scenario, int(seed))
            plot_residual_heterogeneity(batch_df, event_df, plot_dir, scenario, int(seed))
            plot_theta_tracking(batch_df, event_df, plot_dir, scenario, int(seed))
            plot_y_tracking(raw_results, event_df, plot_dir, scenario, int(seed))
            methods = method_order(list(batch_df[(batch_df["scenario"] == scenario) & (batch_df["seed"] == seed)]["method"].unique()))
            for method in methods:
                if args.plot_all_runlengths or seed == int(args.seeds[0]):
                    plot_runlength_heatmap(runlength_df, plot_dir, scenario, int(seed), method, int(args.max_run_length_plot))
                raw = raw_results.get((scenario, int(seed), method))
                if raw is not None and (args.plot_all_heatmaps or seed == int(args.seeds[0])):
                    plot_residual_coherence(raw, plot_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run mechanism diagnostics for DOC half-discrepancy, online-dynamic, and MC-inducing discrepancy ablations.")
    parser.add_argument("--out_dir", type=str, default="figs/mechanism_figures")
    parser.add_argument("--scenarios", nargs="+", default=["sudden", "slope", "random_walk"], choices=["sudden", "slope", "random_walk"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--method-set", choices=["core", "shared", "particle", "basis_ablation", "inducing_ablation", "mc_inducing_ablation", "online_bpc_ablation", "online_bpc_exact_ablation", "online_bpc_exact_hyper_ablation", "online_bpc_proxy_ablation", "online_bpc_proxy_stable_ablation", "online_bpc_proxy_sigmaobs_ablation", "online_bpc_sigmaobs_best_ablation", "online_bpc_fixedsupport_ablation", "online_bpc_fixedsupport_particle_ablation", "online_bpc_controller_ablation", "proxy_bocpd_tuning", "wcusum_ablation"], default="core")
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--noise-sd", type=float, default=0.2)
    parser.add_argument("--theta0", type=float, default=1.5)
    parser.add_argument("--sudden-mag", type=float, default=0.40)
    parser.add_argument("--sudden-seg-len", type=int, default=120)
    parser.add_argument("--phi-center", type=float, default=7.5)
    parser.add_argument("--slope", type=float, default=0.0010)
    parser.add_argument("--slope-total-T", type=int, default=480)
    parser.add_argument("--rw-step-sd", type=float, default=0.06)
    parser.add_argument("--rw-total-T", type=int, default=480)
    parser.add_argument("--oracle-grid-size", type=int, default=500)
    parser.add_argument("--phi2-grid-size", type=int, default=400)
    parser.add_argument("--phi2-min", type=float, default=2.0)
    parser.add_argument("--phi2-max", type=float, default=12.0)
    parser.add_argument("--num-particles", type=int, default=512)
    parser.add_argument("--pf-random-walk-scale", type=float, default=0.10)
    parser.add_argument("--resample-ess-ratio", type=float, default=0.5)
    parser.add_argument("--max-experts", type=int, default=5)
    parser.add_argument("--hazard-lambda", type=float, default=200.0)
    parser.add_argument("--restart-cooldown", type=int, default=10)
    parser.add_argument("--restart-margin", type=float, default=1.0)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--sigma-eps", type=float, default=0.05)
    parser.add_argument("--delta-lengthscale", type=float, default=1.0)
    parser.add_argument("--delta-variance", type=float, default=0.01)
    parser.add_argument("--delta-noise", type=float, default=1e-6)
    parser.add_argument("--delta-online-min-points", type=int, default=3)
    parser.add_argument("--delta-online-init-max-iter", type=int, default=40)
    parser.add_argument("--delta-basis-num-features", type=int, default=20)
    parser.add_argument("--delta-basis-prior-var-scale", type=float, default=1.0)
    parser.add_argument("--delta-dynamic-num-features", type=int, default=20)
    parser.add_argument("--delta-dynamic-forgetting", type=float, default=0.98)
    parser.add_argument("--delta-dynamic-process-noise-scale", type=float, default=1e-3)
    parser.add_argument("--delta-dynamic-prior-var-scale", type=float, default=1.0)
    parser.add_argument("--delta-dynamic-buffer-max-points", type=int, default=256)
    parser.add_argument("--delta-bpc-lambda", type=float, default=1.0)
    parser.add_argument("--controller-sr-alpha", type=float, default=0.01)
    parser.add_argument("--controller-sr-min-len", type=int, default=2)
    parser.add_argument("--controller-sr-warmup-batches", type=int, default=2)
    parser.add_argument("--controller-sr-max-active", type=int, default=64)
    parser.add_argument("--controller-sr-clip-low", type=float, default=0.0)
    parser.add_argument("--controller-sr-clip-high", type=float, default=20.0)
    parser.add_argument("--controller-wcusum-warmup-batches", type=int, default=3)
    parser.add_argument("--controller-wcusum-window", type=int, default=4)
    parser.add_argument("--controller-wcusum-threshold", type=float, default=0.25)
    parser.add_argument("--controller-wcusum-kappa", type=float, default=0.25)
    parser.add_argument("--controller-wcusum-sigma-floor", type=float, default=0.25)
    parser.add_argument("--delta-inducing-num-points", type=int, default=20)
    parser.add_argument("--delta-inducing-init-steps", type=int, default=40)
    parser.add_argument("--delta-inducing-update-steps", type=int, default=6)
    parser.add_argument("--delta-inducing-lr", type=float, default=0.03)
    parser.add_argument("--delta-inducing-buffer-max-points", type=int, default=256)
    parser.add_argument("--delta-inducing-learn-locations", action="store_true")
    parser.add_argument("--delta-mc-num-inducing-points", type=int, default=16)
    parser.add_argument("--delta-mc-num-particles", type=int, default=8)
    parser.add_argument("--delta-mc-resample-ess-ratio", type=float, default=0.5)
    parser.add_argument("--delta-mc-refresh-every", type=int, default=0)
    parser.add_argument("--delta-mc-include-conditional-var", action="store_true")
    parser.add_argument("--no-delta-mc-include-conditional-var", action="store_false", dest="delta_mc_include_conditional_var")
    parser.set_defaults(delta_mc_include_conditional_var=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--event-window", type=int, default=6)
    parser.add_argument("--restart-window", type=int, default=6)
    parser.add_argument("--max-run-length-plot", type=int, default=120)
    parser.add_argument("--plot-all-seeds", action="store_true")
    parser.add_argument("--plot-all-heatmaps", action="store_true")
    parser.add_argument("--plot-all-runlengths", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Run a tiny single-method smoke test.")
    return parser.parse_args()


def apply_smoke_overrides(args: argparse.Namespace) -> argparse.Namespace:
    if not args.smoke:
        return args
    args.scenarios = [args.scenarios[0] if args.scenarios else "sudden"]
    args.seeds = [0]
    args.methods = ["half_refit"]
    args.batch_size = 10
    args.sudden_seg_len = 20
    args.slope_total_T = 40
    args.rw_total_T = 40
    args.num_particles = 48
    args.max_experts = 3
    args.oracle_grid_size = 80
    args.phi2_grid_size = 80
    args.delta_online_init_max_iter = 2
    args.event_window = 2
    args.restart_window = 2
    args.out_dir = str(Path(args.out_dir) / "smoke")
    return args


def json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, torch.dtype):
        return str(obj)
    return str(obj)


def main() -> None:
    args = apply_smoke_overrides(parse_args())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "mechanism_runner_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=json_default)
    print(f"Writing outputs to {out_dir}")
    batch_df, evidence_df, runlength_df, raw_results, event_df = run_all(args)
    make_plots(args, batch_df, evidence_df, runlength_df, raw_results, event_df)
    print("Mechanism experiment finished.")
    print(f"Saved batch records: {out_dir / 'mechanism_batch_records.csv'}")
    print(f"Saved plots under: {out_dir / 'plots'}")


if __name__ == "__main__":
    main()
