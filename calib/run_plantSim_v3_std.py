"""
run_plantSim_v3_std.py - Plant Simulation 校准实验 (标准化版本)

用法:
    # 方式1: 从目录读取Excel文件 (默认)
    python -m calib.run_plantSim_v3_std --data_dir "C:/Users/yxu59/files/winter2026/park/simulation/PhysicalData_v3"
    
    # 方式2: 从CSV文件读取
    python -m calib.run_plantSim_v3_std --csv "C:/Users/yxu59/files/autumn2025/park/DynamicCalibration/physical_data.csv"
    
    # 其他参数
    python -m calib.run_plantSim_v3_std --csv physical_data.csv --out_dir figs/plantSim/v3_stdSingle --modes 0 1 2
"""

import math
import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List
from time import time
from tqdm import tqdm

# -------------------------------------------------------------
# Your existing modules (keep same as before)
# -------------------------------------------------------------
from .configs import CalibrationConfig, BOCPDConfig, ModelConfig
from .emulator import DeterministicSimulator
from .online_calibrator import OnlineBayesCalibrator, crps_gaussian
from .bpc import BayesianProjectedCalibration
from .bpc_bocpd import *
from .restart_bocpd_ogp_gpytorch import (
    BOCPD_OGP, OGPPFConfig, OGPParticleFilter,
    RollingStats as OGPRollingStats,
    make_fast_batched_grad_func,
)
from .paper_pf_digital_twin import WardPaperPFConfig, WardPaperParticleFilter
from .run_synthetic_suddenCmp_tryThm import PFWithGPPrediction, KOHSlidingWindow

from .Deal_data import *
from tqdm import tqdm

import warnings
from gpytorch.utils.warnings import GPInputWarning

warnings.filterwarnings("ignore")

from calib.v3_utils import *

import dataclasses
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from tqdm import tqdm


def _finite_mean(values) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size > 0 else float("nan")


def _gaussian_crps_mean(mu, var, y) -> float:
    mu_t = torch.as_tensor(mu, dtype=torch.float64).detach().cpu()
    var_t = torch.clamp(torch.as_tensor(var, dtype=torch.float64).detach().cpu(), min=1e-12)
    y_t = torch.as_tensor(y, dtype=torch.float64).detach().cpu()
    return float(crps_gaussian(mu_t, var_t, y_t).mean().item())


def _gaussian_crps_mean_raw(gt, mu_s, var_s, y_raw) -> float:
    mu_np = torch.as_tensor(mu_s, dtype=torch.float64).detach().cpu().numpy().reshape(-1)
    var_np = torch.as_tensor(var_s, dtype=torch.float64).detach().cpu().numpy().reshape(-1)
    y_np = np.asarray(y_raw, dtype=float).reshape(-1)
    y_scale = float(gt.y_scaler.scale_[0])
    mu_raw = np.asarray(gt.y_s_to_raw(mu_np), dtype=float).reshape(-1)
    var_raw = np.clip(var_np, 1e-12, None) * (y_scale ** 2)
    return _gaussian_crps_mean(mu_raw, var_raw, y_np)


def _raw_pred_from_standardized(gt, mu_s, var_s):
    mu_np = torch.as_tensor(mu_s, dtype=torch.float64).detach().cpu().numpy().reshape(-1)
    var_np = torch.as_tensor(var_s, dtype=torch.float64).detach().cpu().numpy().reshape(-1)
    y_scale = float(gt.y_scaler.scale_[0])
    mu_raw = np.asarray(gt.y_s_to_raw(mu_np), dtype=float).reshape(-1)
    var_raw = np.clip(var_np, 1e-12, None) * (y_scale ** 2)
    return mu_raw, var_raw


def _relative_error_denom(y_true: np.ndarray) -> np.ndarray:
    y_abs = np.abs(np.asarray(y_true, dtype=float).reshape(-1))
    if y_abs.size == 0:
        return y_abs
    floor = float(np.quantile(y_abs, 0.05))
    floor = max(floor, 1e-8)
    return np.maximum(y_abs, floor)


def _default_sigma_eps_s() -> float:
    return float(CalibrationConfig().model.sigma_eps)


def _set_global_seed(seed: int) -> None:
    seed_i = int(seed)
    random.seed(seed_i)
    np.random.seed(seed_i)
    torch.manual_seed(seed_i)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_i)


def build_methods(profile: str) -> Dict[str, Dict]:
    profile_name = str(profile).strip().lower()
    base = dict(
        type="bocpd",
        restart_impl="rolled_cusum_260324",
        use_discrepancy=False,
        bocpd_use_discrepancy=True,
    )
    if profile_name == "cpd_ablation":
        return {
            "HalfRefit_BOCPD": dict(base, mode="restart"),
            "Proxy_BOCPD": dict(base, mode="restart",
                                 delta_update_mode="online_bpc_proxy_stablemean",
                                 delta_bpc_obs_noise_mode="sigma_eps",
                                 delta_bpc_predict_add_kernel_noise=False),
            "Proxy_wCUSUM": dict(base, mode="wcusum", controller_name="wcusum",
                                  controller_stat="log_surprise_mean",
                                  controller_wcusum_warmup_batches=3,
                                  controller_wcusum_window=4,
                                  controller_wcusum_threshold=0.25,
                                  controller_wcusum_kappa=0.25,
                                  controller_wcusum_sigma_floor=0.25,
                                  delta_update_mode="online_bpc_proxy_stablemean",
                                  delta_bpc_obs_noise_mode="sigma_eps",
                                  delta_bpc_predict_add_kernel_noise=False),
            "Exact_BOCPD": dict(base, mode="restart",
                                 delta_update_mode="online_bpc_exact",
                                 delta_bpc_obs_noise_mode="sigma_eps",
                                 delta_bpc_predict_add_kernel_noise=False),
            "Exact_wCUSUM": dict(base, mode="wcusum", controller_name="wcusum",
                                  controller_stat="log_surprise_mean",
                                  controller_wcusum_warmup_batches=3,
                                  controller_wcusum_window=4,
                                  controller_wcusum_threshold=0.25,
                                  controller_wcusum_kappa=0.25,
                                  controller_wcusum_sigma_floor=0.25,
                                  delta_update_mode="online_bpc_exact",
                                  delta_bpc_obs_noise_mode="sigma_eps",
                                  delta_bpc_predict_add_kernel_noise=False),
            "FixedSupport_BOCPD": dict(base, mode="restart",
                                        delta_update_mode="online_bpc_fixedsupport_exact",
                                        delta_bpc_obs_noise_mode="sigma_eps",
                                        delta_bpc_predict_add_kernel_noise=False),
            "FixedSupport_wCUSUM": dict(base, mode="wcusum", controller_name="wcusum",
                                         controller_stat="log_surprise_mean",
                                         controller_wcusum_warmup_batches=3,
                                         controller_wcusum_window=4,
                                         controller_wcusum_threshold=0.25,
                                         controller_wcusum_kappa=0.25,
                                         controller_wcusum_sigma_floor=0.25,
                                         delta_update_mode="online_bpc_fixedsupport_exact",
                                         delta_bpc_obs_noise_mode="sigma_eps",
                                         delta_bpc_predict_add_kernel_noise=False),
        }
    if profile_name == "half_exact_ogp":
        return {
            "HalfRefit_BOCPD": dict(base, mode="restart"),
            "Exact_BOCPD": dict(base, mode="restart",
                                 delta_update_mode="online_bpc_exact",
                                 delta_bpc_obs_noise_mode="sigma_eps",
                                 delta_bpc_predict_add_kernel_noise=False),
            "Exact_wCUSUM": dict(base, mode="wcusum", controller_name="wcusum",
                                  controller_stat="log_surprise_mean",
                                  controller_wcusum_warmup_batches=3,
                                  controller_wcusum_window=4,
                                  controller_wcusum_threshold=0.25,
                                  controller_wcusum_kappa=0.25,
                                  controller_wcusum_sigma_floor=0.25,
                                  delta_update_mode="online_bpc_exact",
                                  delta_bpc_obs_noise_mode="sigma_eps",
                                  delta_bpc_predict_add_kernel_noise=False),
            "R-BOCPD-PF-OGP": dict(type="ogp"),
        }
    if profile_name == "half_exact_da_bc":
        return {
            "HalfRefit_BOCPD": dict(base, mode="restart"),
            "Exact_BOCPD": dict(base, mode="restart",
                                 delta_update_mode="online_bpc_exact",
                                 delta_bpc_obs_noise_mode="sigma_eps",
                                 delta_bpc_predict_add_kernel_noise=False),
            "Exact_wCUSUM": dict(base, mode="wcusum", controller_name="wcusum",
                                  controller_stat="log_surprise_mean",
                                  controller_wcusum_warmup_batches=3,
                                  controller_wcusum_window=4,
                                  controller_wcusum_threshold=0.25,
                                  controller_wcusum_kappa=0.25,
                                  controller_wcusum_sigma_floor=0.25,
                                  delta_update_mode="online_bpc_exact",
                                  delta_bpc_obs_noise_mode="sigma_eps",
                                  delta_bpc_predict_add_kernel_noise=False),
            "DA": dict(type="da"),
            "BC": dict(type="bc"),
        }
    if profile_name == "half_exact_damove_bc":
        return {
            "HalfRefit_BOCPD": dict(base, mode="restart"),
            "Exact_BOCPD": dict(base, mode="restart",
                                 delta_update_mode="online_bpc_exact",
                                 delta_bpc_obs_noise_mode="sigma_eps",
                                 delta_bpc_predict_add_kernel_noise=False),
            "Exact_wCUSUM": dict(base, mode="wcusum", controller_name="wcusum",
                                  controller_stat="log_surprise_mean",
                                  controller_wcusum_warmup_batches=3,
                                  controller_wcusum_window=4,
                                  controller_wcusum_threshold=0.25,
                                  controller_wcusum_kappa=0.25,
                                  controller_wcusum_sigma_floor=0.25,
                                  delta_update_mode="online_bpc_exact",
                                  delta_bpc_obs_noise_mode="sigma_eps",
                                  delta_bpc_predict_add_kernel_noise=False),
            "DA": dict(
                type="paper_pf",
                paper_pf_design_x_points=32,
                paper_pf_design_theta_points=7,
                paper_pf_move_logl_std=0.10,
            ),
            "BC": dict(type="bc"),
        }
    if profile_name == "cpd_ablation_plus":
        return {
            "Proxy_None": dict(base, mode="single_segment", controller_name="none",
                                delta_update_mode="online_bpc_proxy_stablemean",
                                delta_bpc_obs_noise_mode="sigma_eps",
                                delta_bpc_predict_add_kernel_noise=False),
            "Proxy_BOCPD": dict(base, mode="restart",
                                 delta_update_mode="online_bpc_proxy_stablemean",
                                 delta_bpc_obs_noise_mode="sigma_eps",
                                 delta_bpc_predict_add_kernel_noise=False),
            "Proxy_wCUSUM": dict(base, mode="wcusum", controller_name="wcusum",
                                  controller_stat="log_surprise_mean",
                                  controller_wcusum_warmup_batches=3,
                                  controller_wcusum_window=4,
                                  controller_wcusum_threshold=0.25,
                                  controller_wcusum_kappa=0.25,
                                  controller_wcusum_sigma_floor=0.25,
                                  delta_update_mode="online_bpc_proxy_stablemean",
                                  delta_bpc_obs_noise_mode="sigma_eps",
                                  delta_bpc_predict_add_kernel_noise=False),
            "Exact_None": dict(base, mode="single_segment", controller_name="none",
                                delta_update_mode="online_bpc_exact",
                                delta_bpc_obs_noise_mode="sigma_eps",
                                delta_bpc_predict_add_kernel_noise=False),
            "Exact_BOCPD": dict(base, mode="restart",
                                 delta_update_mode="online_bpc_exact",
                                 delta_bpc_obs_noise_mode="sigma_eps",
                                 delta_bpc_predict_add_kernel_noise=False),
            "Exact_wCUSUM": dict(base, mode="wcusum", controller_name="wcusum",
                                  controller_stat="log_surprise_mean",
                                  controller_wcusum_warmup_batches=3,
                                  controller_wcusum_window=4,
                                  controller_wcusum_threshold=0.25,
                                  controller_wcusum_kappa=0.25,
                                  controller_wcusum_sigma_floor=0.25,
                                  delta_update_mode="online_bpc_exact",
                                  delta_bpc_obs_noise_mode="sigma_eps",
                                  delta_bpc_predict_add_kernel_noise=False),
            "FixedSupport_None": dict(base, mode="single_segment", controller_name="none",
                                       delta_update_mode="online_bpc_fixedsupport_exact",
                                       delta_bpc_obs_noise_mode="sigma_eps",
                                       delta_bpc_predict_add_kernel_noise=False),
            "FixedSupport_BOCPD": dict(base, mode="restart",
                                        delta_update_mode="online_bpc_fixedsupport_exact",
                                        delta_bpc_obs_noise_mode="sigma_eps",
                                        delta_bpc_predict_add_kernel_noise=False),
            "FixedSupport_wCUSUM": dict(base, mode="wcusum", controller_name="wcusum",
                                         controller_stat="log_surprise_mean",
                                         controller_wcusum_warmup_batches=3,
                                         controller_wcusum_window=4,
                                         controller_wcusum_threshold=0.25,
                                         controller_wcusum_kappa=0.25,
                                         controller_wcusum_sigma_floor=0.25,
                                         delta_update_mode="online_bpc_fixedsupport_exact",
                                         delta_bpc_obs_noise_mode="sigma_eps",
                                         delta_bpc_predict_add_kernel_noise=False),
            "DA": dict(type="da"),
            "BC": dict(type="bc"),
        }
    return {
        "R-BOCPD-PF-halfdiscrepancy": dict(base, mode="restart"),
    }


# =========================
# 1) y transform: signed log1p
# =========================
@dataclasses.dataclass
class SignedLog1pTransformer:
    c: float = None

    def fit(self, y: np.ndarray):
        y = np.asarray(y).reshape(-1)
        abs_y = np.abs(y)
        c = np.median(abs_y[abs_y > 0]) if np.any(abs_y > 0) else 1.0
        self.c = float(c)
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        if self.c is None:
            raise ValueError("SignedLog1pTransformer not fitted")
        y = np.asarray(y).reshape(-1)
        return np.sign(y) * np.log1p(np.abs(y) / self.c)

    def inverse_transform(self, z: np.ndarray) -> np.ndarray:
        if self.c is None:
            raise ValueError("SignedLog1pTransformer not fitted")
        z = np.asarray(z).reshape(-1)
        return np.sign(z) * self.c * np.expm1(np.abs(z))


# =========================
# 2) GlobalTransformSep
#    - X_base (5) and theta (1) are standardized SEPARATELY
#    - y_raw -> y_t(signedlog) -> y_s(zscore)
# =========================
@dataclasses.dataclass
class GlobalTransformSep:
    x_base_scaler: StandardScaler = dataclasses.field(default_factory=StandardScaler)   # 5-d
    theta_scaler: StandardScaler = dataclasses.field(default_factory=StandardScaler)   # 1-d
    y_scaler: StandardScaler = dataclasses.field(default_factory=StandardScaler)       # 1-d (on y_t)
    y_transform: SignedLog1pTransformer = dataclasses.field(default_factory=SignedLog1pTransformer)
    fitted: bool = False

    def fit(self, X_base: np.ndarray, theta_raw: np.ndarray, y_raw: np.ndarray):
        X_base = np.asarray(X_base)
        theta_raw = np.asarray(theta_raw).reshape(-1, 1)  # minutes
        y_raw = np.asarray(y_raw).reshape(-1)

        if X_base.shape[0] != theta_raw.shape[0] or X_base.shape[0] != y_raw.shape[0]:
            raise ValueError("fit: length mismatch")

        # fit x scalers
        self.x_base_scaler.fit(X_base)
        self.theta_scaler.fit(theta_raw)

        # fit y transform + y scaler
        self.y_transform.fit(y_raw)
        y_t = self.y_transform.transform(y_raw)
        self.y_scaler.fit(y_t.reshape(-1, 1))

        self.fitted = True
        return self

    # ---- X_base ----
    def X_base_to_s(self, X_base: np.ndarray) -> np.ndarray:
        if not self.fitted: raise ValueError("GlobalTransformSep not fitted")
        return self.x_base_scaler.transform(np.asarray(X_base)).astype(np.float32)

    # ---- theta ----
    def theta_raw_to_s(self, theta_raw: np.ndarray) -> np.ndarray:
        if not self.fitted: raise ValueError("GlobalTransformSep not fitted")
        th = np.asarray(theta_raw).reshape(-1, 1)
        return self.theta_scaler.transform(th).ravel().astype(np.float32)

    def theta_s_to_raw(self, theta_s: np.ndarray) -> np.ndarray:
        if not self.fitted: raise ValueError("GlobalTransformSep not fitted")
        ths = np.asarray(theta_s).reshape(-1, 1)
        return self.theta_scaler.inverse_transform(ths).ravel()

    @property
    def theta_mu(self) -> float:
        return float(self.theta_scaler.mean_[0])

    @property
    def theta_sd(self) -> float:
        return float(self.theta_scaler.scale_[0])

    # ---- y ----
    def y_raw_to_s(self, y_raw: np.ndarray) -> np.ndarray:
        if not self.fitted: raise ValueError("GlobalTransformSep not fitted")
        y_raw = np.asarray(y_raw).reshape(-1)
        y_t = self.y_transform.transform(y_raw)
        y_s = self.y_scaler.transform(y_t.reshape(-1, 1)).ravel()
        return y_s.astype(np.float32)

    def y_s_to_raw(self, y_s: np.ndarray) -> np.ndarray:
        if not self.fitted: raise ValueError("GlobalTransformSep not fitted")
        y_s = np.asarray(y_s).reshape(-1)
        y_t = self.y_scaler.inverse_transform(y_s.reshape(-1, 1)).ravel()
        y_raw = self.y_transform.inverse_transform(y_t)
        return y_raw


# =========================
# 3) MLP (input_dim=6: [x_base_s(5), theta_s(1)])
# =========================
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden=(128, 128, 64), dropout=0.0):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# =========================
# 4) NNModelTorchStd: train/predict in standardized space only
#    - inputs: X_full_s (B,6)
#    - outputs: y_s (B,)
# =========================
@dataclasses.dataclass
class NNModelTorchStd:
    input_dim: int = 6
    device: str = None
    model: nn.Module = None

    def _get_device(self):
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(
        self,
        X_full_s: np.ndarray,    # (N,6) standardized
        y_s: np.ndarray,         # (N,) standardized
        val_frac: float = 0.10,
        batch_size: int = 128,
        lr: float = 1e-3,
        epochs: int = 200,
        hidden=(128, 64, 32),
        dropout: float = 0.0,
        weight_decay: float = 1e-6,
        seed: int = 0,
        verbose_every: int = 20,
    ):
        dev = self._get_device()

        X_full_s = np.asarray(X_full_s).astype(np.float32)
        y_s = np.asarray(y_s).astype(np.float32).reshape(-1)

        X_tr, X_va, y_tr, y_va = train_test_split(
            X_full_s, y_s, test_size=val_frac, random_state=seed, shuffle=True
        )

        X_tr_t = torch.from_numpy(X_tr).to(dev)
        y_tr_t = torch.from_numpy(y_tr).to(dev)
        X_va_t = torch.from_numpy(X_va).to(dev)
        y_va_t = torch.from_numpy(y_va).to(dev)

        train_loader = DataLoader(
            TensorDataset(X_tr_t, y_tr_t),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False
        )

        self.model = MLP(self.input_dim, hidden=hidden, dropout=dropout).to(dev)
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.MSELoss()

        best_val = float("inf")
        best_state = None
        patience = 30
        bad = 0

        for ep in range(1, epochs + 1):
            self.model.train()
            for xb, yb in train_loader:
                opt.zero_grad()
                pred = self.model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()

            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(X_va_t)
                val_loss = loss_fn(val_pred, y_va_t).item()

            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                bad = 0
            else:
                bad += 1

            if verbose_every and ep % verbose_every == 0:
                print(f"epoch {ep:4d} | val_mse(y_s)={val_loss:.6f} | best={best_val:.6f}")

            if bad >= patience:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self

    def predict_y_s_from_Xfull_s(self, X_full_s: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("NN model not fitted/loaded.")
        dev = self._get_device()
        Xs = np.asarray(X_full_s).astype(np.float32)
        Xt = torch.from_numpy(Xs).to(dev)
        self.model.eval()
        with torch.no_grad():
            y_s = self.model(Xt).detach().cpu().numpy()
        return y_s

    def save(self, path: str):
        if self.model is None:
            raise ValueError("Nothing to save.")
        bundle = {"state_dict": self.model.state_dict()}
        joblib.dump(bundle, path)

    @classmethod
    def load(cls, path: str, device: str = None, input_dim: int = 6, hidden=(128,128,64)):
        bundle = joblib.load(path)
        obj = cls(input_dim=input_dim, device=device)
        obj.model = MLP(input_dim, hidden=tuple(hidden), dropout=0.0).to(obj._get_device())
        obj.model.load_state_dict(bundle["state_dict"])
        obj.model.eval()
        return obj


# =========================
# 5) Emulator in standardized space
#    predict(x_base_s (B,5), theta_s (N,1)) -> mu_s (N,B), var_s (N,B)
# =========================
class PlantEmulatorNNStd:
    def __init__(self, nn_std: NNModelTorchStd):
        self.nn = nn_std

    def predict(self, x, theta):
        """
        x: torch.Tensor
           - either (B,5) standardized X_base_s
        theta: torch.Tensor
           - either (N,1) theta_s particles
           - or (B,1) theta_s per-sample
        Returns:
           mu_s, var_s  (both torch.float64)
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        if not isinstance(theta, torch.Tensor):
            theta = torch.tensor(theta)

        x = x.to(torch.float64)
        theta = theta.to(torch.float64)

        # Case 1: theta is particle set (N,1), x is batch (B,5) -> output (N,B)
        if theta.ndim == 2 and theta.shape[1] == 1 and x.ndim == 2 and x.shape[1] == 5:
            N = theta.shape[0]
            B = x.shape[0]

            # Build X_full_s for all (particle, batch)
            x_rep = x.unsqueeze(0).repeat(N, 1, 1)              # (N,B,5)
            th_rep = theta.unsqueeze(1).repeat(1, B, 1)         # (N,B,1)
            X_full = torch.cat([x_rep, th_rep], dim=-1)         # (N,B,6)

            X_full_np = X_full.reshape(N*B, 6).cpu().numpy()
            mu_np = self.nn.predict_y_s_from_Xfull_s(X_full_np).reshape(N, B)
            mu = torch.tensor(mu_np, dtype=torch.float64, device=x.device).T
            var = torch.zeros_like(mu)
            return mu, var

        # Case 2: theta is per-sample (B,1) -> output (B,)
        if theta.ndim == 2 and theta.shape[1] == 1 and x.ndim == 2 and x.shape[1] == 5 and theta.shape[0] == x.shape[0]:
            X_full = torch.cat([x, theta], dim=1)               # (B,6)
            mu_np = self.nn.predict_y_s_from_Xfull_s(X_full.cpu().numpy())
            mu = torch.tensor(mu_np, dtype=torch.float64, device=x.device)
            var = torch.zeros_like(mu)
            return mu, var

        raise ValueError(f"Unsupported shapes: x={tuple(x.shape)}, theta={tuple(theta.shape)}")


def batch_X_base_to_s(gt: GlobalTransformSep, Xb: np.ndarray) -> torch.Tensor:
    Xs = gt.X_base_to_s(Xb).astype(np.float64)      # (B,5)
    return torch.tensor(Xs, dtype=torch.float64)

def batch_y_to_s(gt: GlobalTransformSep, yb: np.ndarray) -> torch.Tensor:
    ys = gt.y_raw_to_s(yb).astype(np.float64)       # (B,)
    return torch.tensor(ys, dtype=torch.float64)


# =========================
# PlantEmulatorNNStdTorch: Pure-torch differentiable for OGP
# =========================
from calib.emulator import Emulator


class PlantEmulatorNNStdTorch(Emulator):
    """
    Pure-torch differentiable wrapper for standardized NN.
    Works in standardized space: x_base_s [5], theta_s [1] -> y_s.
    """
    _NN_CHUNK = 8192

    def __init__(self, nn_std: NNModelTorchStd, gt: GlobalTransformSep,
                 device: str = "cuda", dtype= torch.float64):
        self.device = device
        self.dtype = dtype
        self.gt = gt
        self.nn = nn_std.model.to(device)
        self.nn.eval()
        for p in self.nn.parameters():
            p.requires_grad_(False)

    def _forward_y_s(self, x_full_s: torch.Tensor) -> torch.Tensor:
        """x_full_s [M, 6] standardized -> y_s [M]. Differentiable."""
        return self.nn(x_full_s.float()).to(self.dtype)

    def predict(self, x: torch.Tensor, theta: torch.Tensor):
        """x [B, 5] x_base_s, theta [N, 1] theta_s -> (mu [B,N], var [B,N])"""
        B, N = x.shape[0], theta.shape[0]
        x_dev = x.to(device=self.device, dtype=self.dtype)
        th_dev = theta.to(device=self.device, dtype=self.dtype)
        x_rep = x_dev.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1)
        th_rep = th_dev.unsqueeze(0).expand(B, N, -1).reshape(B * N, -1)
        x_full = torch.cat([x_rep, th_rep], dim=1)
        total = B * N
        with torch.no_grad():
            if total <= self._NN_CHUNK:
                y = self._forward_y_s(x_full)
            else:
                y = torch.empty(total, device=self.device, dtype=self.dtype)
                for i in range(0, total, self._NN_CHUNK):
                    j = min(i + self._NN_CHUNK, total)
                    y[i:j] = self._forward_y_s(x_full[i:j])
        mu = y.reshape(B, N)
        return mu, torch.zeros_like(mu)

    def sim_func(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """One-to-one: x [M,5], theta [M,1] -> y [M]. Differentiable."""
        x_full = torch.cat([x.to(self.dtype), theta.to(self.dtype)], dim=1)
        return self._forward_y_s(x_full)

    def x_domain_from_scaler(self, dx: int = 5):
        """Auto-compute x_domain in STANDARDIZED space: mean ± 3*std."""
        x_mean = self.gt.x_base_scaler.mean_
        x_std = self.gt.x_base_scaler.scale_
        lo = x_mean[:dx] - 3 * x_std[:dx]
        hi = x_mean[:dx] + 3 * x_std[:dx]
        return [(float(lo[i]), float(hi[i])) for i in range(dx)]


def _aggregate_ogp_particles(bocpd, ci=0.9):
    all_theta, all_w = [], []
    for e in bocpd.experts:
        w_e = math.exp(e.log_mass)
        all_theta.append(e.pf.theta)
        all_w.append(e.pf.weights() * w_e)
    theta_cat = torch.cat(all_theta, dim=0)
    w_cat = torch.cat(all_w, dim=0)
    w_cat = w_cat / w_cat.sum()
    mean_th = (w_cat.unsqueeze(1) * theta_cat).sum(dim=0)
    var_th = (w_cat.unsqueeze(1) * (theta_cat - mean_th).pow(2)).sum(dim=0)
    alpha = (1 - ci) / 2
    sorted_th, sorted_idx = torch.sort(theta_cat, dim=0)
    sorted_w = w_cat[sorted_idx.squeeze()]
    cum_w = torch.cumsum(sorted_w, dim=0)
    lo_idx = (cum_w >= alpha).nonzero(as_tuple=True)[0][0]
    hi_idx = (cum_w >= 1 - alpha).nonzero(as_tuple=True)[0][0]
    return mean_th, var_th, sorted_th[lo_idx], sorted_th[hi_idx]


# =========================
# 6) Helper: build standardized batches
# =========================
def batch_X_base_to_s(gt: GlobalTransformSep, Xb: np.ndarray) -> torch.Tensor:
    Xs = gt.X_base_to_s(Xb).astype(np.float64)      # (B,5)
    return torch.tensor(Xs, dtype=torch.float64)

def batch_y_to_s(gt: GlobalTransformSep, yb: np.ndarray) -> torch.Tensor:
    ys = gt.y_raw_to_s(yb).astype(np.float64)       # (B,)
    return torch.tensor(ys, dtype=torch.float64)


# =========================
# 7) Pipeline initialisation & helpers
# =========================
# _DEFAULT_NPZ = r"C:/Users/yxu59/files/winter2026/park/simulation/ComputerData_v3/factory_aggregated.npz"
_DEFAULT_NPZ = r"factory_aggregated.npz"

# Module-level globals — populated by init_pipeline()
gt  = None   # type: GlobalTransformSep
nn_std = None   # type: NNModelTorchStd
emu = None   # type: PlantEmulatorNNStd
a_s = 0.0
b_s = 0.0


def init_pipeline(
    npz_path: str = None,
    model_save_path: str = "nn_std.bundle.joblib",
    epochs: int = 200,
    force_retrain: bool = False,
):
    """Load computer-sim data, fit transforms, train / load NN emulator.

    Sets module-level globals (gt, nn_std, emu, a_s, b_s).
    Returns (gt, nn_std, emu, a_s, b_s) for convenience.
    """
    global gt, nn_std, emu, a_s, b_s
    if gt is not None and not force_retrain:
        return gt, nn_std, emu, a_s, b_s

    if npz_path is None:
        npz_path = _DEFAULT_NPZ

    print(f"[init] Loading computer data from {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    X_base = data["X"]       # (N,5)
    y_raw  = data["y"]       # (N,)
    theta_raw = data["theta"]  # (N,) minutes

    X_tr, X_te, y_tr, y_te, th_tr, th_te = train_test_split(
        X_base, y_raw, theta_raw, test_size=0.2, random_state=0, shuffle=True
    )

    gt = GlobalTransformSep().fit(X_tr, th_tr, y_tr)

    if os.path.exists(model_save_path) and not force_retrain:
        print(f"[init] Loading pre-trained NN from {model_save_path}")
        nn_std = NNModelTorchStd.load(model_save_path, hidden=(128, 64, 32))
    else:
        print(f"[init] Training NN emulator ({epochs} epochs) ...")
        X_tr_s = gt.X_base_to_s(X_tr)
        th_tr_s = gt.theta_raw_to_s(th_tr).reshape(-1, 1)
        X_full_tr_s = np.concatenate([X_tr_s, th_tr_s], axis=1)
        y_tr_s = gt.y_raw_to_s(y_tr)
        nn_std = NNModelTorchStd(input_dim=6).fit(X_full_tr_s, y_tr_s, epochs=epochs)
        nn_std.save(model_save_path)

    emu = PlantEmulatorNNStd(nn_std)

    a_raw, b_raw = 3.0, 21.0
    a_s = (a_raw - gt.theta_mu) / gt.theta_sd
    b_s = (b_raw - gt.theta_mu) / gt.theta_sd

    print("[init] Pipeline ready.\n")
    return gt, nn_std, emu, a_s, b_s


def prior_sampler(N):
    return torch.rand(N, 1, dtype=torch.float64) * (b_s - a_s) + a_s   # theta_s

def batches(stream: StreamClass, batch_size: int, max_batches: int = None):
    emitted_batches = 0
    while True:
        if max_batches is not None and emitted_batches >= int(max_batches):
            break
        try:
            batch = stream.next(batch_size)
        except StopIteration:
            break
        emitted_batches += 1
        yield batch


def _parse_max_batches_by_mode(specs):
    mode_to_cap = {}
    if not specs:
        return mode_to_cap
    for item in specs:
        text = str(item).strip()
        if not text:
            continue
        if ":" not in text:
            raise ValueError(f"Invalid --max-batches-by-mode entry '{text}'. Expected MODE:COUNT.")
        mode_s, cap_s = text.split(":", 1)
        mode_i = int(mode_s)
        cap_i = int(cap_s)
        if cap_i <= 0:
            raise ValueError(f"Invalid max batch count for mode {mode_i}: {cap_i}")
        mode_to_cap[mode_i] = cap_i
    return mode_to_cap

def summarize_metrics(result: dict):
    """Compute theta/y metrics from one method result dict."""
    theta = np.asarray(result.get("theta_hist", []), dtype=float)
    theta_var = np.asarray(result.get("theta_var_hist", []), dtype=float)
    gt_theta = np.asarray(result.get("gt_theta_hist", []), dtype=float)
    y_rmse_hist = np.asarray(result.get("rmse_hist", []), dtype=float)
    y_crps_hist = np.asarray(result.get("y_crps_hist", []), dtype=float)
    y_true = np.asarray(result.get("y_true_hist", []), dtype=float)
    y_pred = np.asarray(result.get("y_pred_hist", []), dtype=float)
    y_var = np.asarray(result.get("y_var_hist", []), dtype=float)
    restart_hist = np.asarray(result.get("restart_hist", []), dtype=bool)

    n_theta = min(len(theta), len(gt_theta), len(theta_var))
    if n_theta == 0:
        theta_rmse = float("nan")
        theta_crps = float("nan")
    else:
        theta_rmse = float(np.sqrt(np.mean((theta[:n_theta] - gt_theta[:n_theta]) ** 2)))
        theta_var_clip = np.clip(theta_var[:n_theta], 1e-12, None)
        theta_crps = float(
            crps_gaussian(
                torch.tensor(theta[:n_theta], dtype=torch.float64),
                torch.tensor(theta_var_clip, dtype=torch.float64),
                torch.tensor(gt_theta[:n_theta], dtype=torch.float64),
            ).mean().item()
        )

    # rmse_hist has an initial placeholder 0 in this script
    y_rmse = _finite_mean(y_rmse_hist[1:]) if len(y_rmse_hist) > 1 else float("nan")
    y_crps = _finite_mean(y_crps_hist) if len(y_crps_hist) > 0 else float("nan")

    n_y = min(len(y_true), len(y_pred), len(y_var))
    if n_y == 0:
        y_rel_rmse = float("nan")
        y_rel_crps = float("nan")
    else:
        denom = _relative_error_denom(y_true[:n_y])
        rel_sq = ((y_pred[:n_y] - y_true[:n_y]) / denom) ** 2
        y_rel_rmse = float(np.sqrt(np.mean(rel_sq)))
        crps_raw = crps_gaussian(
            torch.tensor(y_pred[:n_y], dtype=torch.float64),
            torch.tensor(np.clip(y_var[:n_y], 1e-12, None), dtype=torch.float64),
            torch.tensor(y_true[:n_y], dtype=torch.float64),
        ).detach().cpu().numpy()
        y_rel_crps = float(np.mean(crps_raw / denom))

    return dict(
        theta_rmse=theta_rmse,
        theta_crps=theta_crps,
        y_rmse=y_rmse,
        y_crps=y_crps,
        y_rel_rmse=y_rel_rmse,
        y_rel_crps=y_rel_crps,
        restart_count=float(restart_hist.sum()) if restart_hist.size > 0 else 0.0,
        runtime_sec=float(result.get("runtime_sec", float("nan"))),
    )

def run_plantSim(mode, methods, batch_size, data_dir=None, csv_path=None, seed=None, max_batches=None):
    if data_dir is None and csv_path is None:
        data_dir = "C:/Users/yxu59/files/winter2026/park/simulation/PhysicalData_v3"
    if seed is not None:
        _set_global_seed(int(seed))

    # emulator = PlantEmulatorNN()
    emulator = emu
    theta_lo_s = float((3.0 - gt.theta_mu) / gt.theta_sd)
    theta_hi_s = float((21.0 - gt.theta_mu) / gt.theta_sd)

    def _sim_func_std_np(X_np: np.ndarray, theta_np: np.ndarray) -> np.ndarray:
        X_np = np.asarray(X_np, dtype=np.float64)
        theta_np = np.asarray(theta_np, dtype=np.float64).reshape(-1, 1)
        if X_np.shape[0] != theta_np.shape[0]:
            raise ValueError(f"Expected one-to-one standardized simulation inputs, got X={X_np.shape}, theta={theta_np.shape}")
        X_full = np.concatenate([X_np, theta_np], axis=1)
        return np.asarray(nn_std.predict_y_s_from_Xfull_s(X_full), dtype=float).reshape(-1)

    results = {}

    for name, meta in methods.items():
        t_start = time()
        theta_hist, theta_var_hist, gt_theta_hist = [], [], []
        rmse_hist, comp_rmse_hist = [0], [0]
        y_crps_hist = []
        restart_hist = []
        y_true_hist, y_pred_hist, y_var_hist = [], [], []
        idx = 0
        batch_size = batch_size

        if mode == 2:
            jp = JumpPlan(
                max_jumps=5,           # ~4-5 jumps for ~1200 pts
                min_gap_theta=500.0,   # seconds; tune
                min_interval=180,
                max_interval=320,
                min_jump_span=40,
                seed=int(seed) if seed is not None else 7
            )
            stream = StreamClass(0, folder=data_dir, csv_path=csv_path, jump_plan=jp)
        else:
            stream = StreamClass(mode, folder=data_dir, csv_path=csv_path)

        # ---------- R-BOCPD-PF-OGP ----------
        if name == "R-BOCPD-PF-OGP":
            ogp_dev = "cuda"
            emulator_nn_std = PlantEmulatorNNStdTorch(nn_std, gt, device=ogp_dev)
            grad_func = make_fast_batched_grad_func(
                emulator_nn_std.sim_func, device=ogp_dev, dtype=torch.float64,
            )
            x_domain = emulator_nn_std.x_domain_from_scaler(dx=5)
            ogp_cfg = OGPPFConfig(
                num_particles=1024,
                x_domain=x_domain,
                theta_lo=torch.tensor([theta_lo_s]),
                theta_hi=torch.tensor([theta_hi_s]),
                theta_move_std=0.5 / gt.theta_sd,
                ogp_quad_n=3,
                particle_chunk_size=64,
                max_hist=200,
            )
            bocpd_cfg = BOCPDConfig()
            bocpd_cfg.use_restart = True
            model_cfg = ModelConfig(rho=1.0, sigma_eps=float(meta.get("sigma_eps_s", _default_sigma_eps_s())))
            roll = OGPRollingStats(window=50)

            bocpd = BOCPD_OGP(
                config=bocpd_cfg,
                ogp_pf_cfg=ogp_cfg,
                batched_grad_func=grad_func,
                device=ogp_dev,
            )

            for Xb, yb, thb in tqdm(batches(stream, batch_size, max_batches=max_batches), desc=f"Running {name}"):
                newX = batch_X_base_to_s(gt, Xb).to(device=ogp_dev)
                newY = batch_y_to_s(gt, yb).to(device=ogp_dev)
                gt_theta = torch.tensor(thb)

                if idx > 0 and len(bocpd.experts) > 0:
                    mix_mu = torch.zeros(newX.shape[0], device=ogp_dev, dtype=torch.float64)
                    mix_var = torch.zeros(newX.shape[0], device=ogp_dev, dtype=torch.float64)
                    Z = 0.0
                    for e in bocpd.experts:
                        w_e = math.exp(e.log_mass)
                        e_Xh = e.X_hist if e.X_hist.numel() > 0 else None
                        e_yh = e.y_hist if e.y_hist.numel() > 0 else None
                        mu_e, var_e = e.pf.predict_batch(
                            newX, e_Xh, e_yh,
                            emulator_nn_std, model_cfg.rho, model_cfg.sigma_eps,
                        )
                        mix_mu += w_e * mu_e
                        mix_var += w_e * var_e
                        Z += w_e
                    mix_mu /= max(Z, 1e-12)
                    mix_var /= max(Z, 1e-12)
                    mu_raw = gt.y_s_to_raw(mix_mu.cpu().numpy())
                    _, var_raw = _raw_pred_from_standardized(gt, mix_mu, mix_var)
                    rmse = float(np.sqrt(np.mean((mu_raw - np.asarray(yb))**2)))
                    rmse_hist.append(rmse)
                    y_crps_hist.append(_gaussian_crps_mean_raw(gt, mix_mu, mix_var, yb))
                    y_true_hist.extend(np.asarray(yb, dtype=float).reshape(-1).tolist())
                    y_pred_hist.extend(np.asarray(mu_raw, dtype=float).reshape(-1).tolist())
                    y_var_hist.extend(np.asarray(var_raw, dtype=float).reshape(-1).tolist())
                idx += 1

                rec = bocpd.update_batch(
                    newX, newY, emulator_nn_std, model_cfg, None, prior_sampler,
                    verbose=False,
                )

                dll = rec.get("delta_ll_pair", None)
                if dll is not None and np.isfinite(dll):
                    roll.update(dll)

                mean_theta_s, var_theta_s, lo_s, hi_s = _aggregate_ogp_particles(
                    bocpd, 0.9,
                )
                mean_theta_raw = gt.theta_s_to_raw(float(mean_theta_s[0]))
                var_theta_raw = float(var_theta_s[0]) * (gt.theta_sd ** 2)
                gt_theta_hist.append(float(np.mean(thb)))
                # print(mean_theta_raw, var_theta_raw)
                theta_hist.append(mean_theta_raw)
                theta_var_hist.append(var_theta_raw)
                restart_hist.append(rec["did_restart"])

        # ---------- Standalone PF-OGP (no BOCPD) ----------
        elif name == "PF-OGP":
            ogp_dev = "cuda"
            emulator_nn_std = PlantEmulatorNNStdTorch(nn_std, gt, device=ogp_dev)
            pf_grad_func = make_fast_batched_grad_func(
                emulator_nn_std.sim_func, device=ogp_dev, dtype=torch.float64,
            )
            x_domain = emulator_nn_std.x_domain_from_scaler(dx=5)
            pf_ogp_cfg = OGPPFConfig(
                num_particles=1024,
                x_domain=x_domain,
                theta_lo=torch.tensor([theta_lo_s]),
                theta_hi=torch.tensor([theta_hi_s]),
                theta_move_std=0.5 / gt.theta_sd,
                ogp_quad_n=3,
                particle_chunk_size=64,
                max_hist=200,
            )
            pf_model_cfg = ModelConfig(rho=1.0, sigma_eps=float(meta.get("sigma_eps_s", _default_sigma_eps_s())))

            pf = OGPParticleFilter(
                ogp_cfg=pf_ogp_cfg,
                prior_sampler=prior_sampler,
                batched_grad_func=pf_grad_func,
                device=ogp_dev,
                dtype=torch.float64,
            )

            pf_X_hist = torch.empty(0, 5, dtype=torch.float64, device=ogp_dev)
            pf_y_hist = torch.empty(0, dtype=torch.float64, device=ogp_dev)
            pf_ogp_max_hist = 200

            for Xb, yb, thb in tqdm(batches(stream, batch_size, max_batches=max_batches), desc=f"Running {name}"):
                newX = batch_X_base_to_s(gt, Xb).to(device=ogp_dev)
                newY = batch_y_to_s(gt, yb).to(device=ogp_dev)
                gt_theta = torch.tensor(thb)

                if idx > 0:
                    pf_Xh = pf_X_hist if pf_X_hist.numel() > 0 else None
                    pf_yh = pf_y_hist if pf_y_hist.numel() > 0 else None
                    mu_mix, var_mix = pf.predict_batch(
                        newX, pf_Xh, pf_yh,
                        emulator_nn_std, pf_model_cfg.rho, pf_model_cfg.sigma_eps,
                    )
                    mu_raw = gt.y_s_to_raw(mu_mix.cpu().numpy())
                    _, var_raw = _raw_pred_from_standardized(gt, mu_mix, var_mix)
                    rmse = float(np.sqrt(np.mean((mu_raw - np.asarray(yb))**2)))
                    rmse_hist.append(rmse)
                    y_crps_hist.append(_gaussian_crps_mean_raw(gt, mu_mix, var_mix, yb))
                    y_true_hist.extend(np.asarray(yb, dtype=float).reshape(-1).tolist())
                    y_pred_hist.extend(np.asarray(mu_raw, dtype=float).reshape(-1).tolist())
                    y_var_hist.extend(np.asarray(var_raw, dtype=float).reshape(-1).tolist())
                idx += 1

                pf.step_batch(
                    newX, newY,
                    pf_X_hist if pf_X_hist.numel() > 0 else None,
                    pf_y_hist if pf_y_hist.numel() > 0 else None,
                    emulator_nn_std,
                    pf_model_cfg.rho,
                    pf_model_cfg.sigma_eps,
                )

                if pf_X_hist.numel() == 0:
                    pf_X_hist = newX.clone()
                    pf_y_hist = newY.clone()
                else:
                    pf_X_hist = torch.cat([pf_X_hist, newX], dim=0)
                    pf_y_hist = torch.cat([pf_y_hist, newY], dim=0)
                if pf_X_hist.shape[0] > pf_ogp_max_hist:
                    pf_X_hist = pf_X_hist[-pf_ogp_max_hist:]
                    pf_y_hist = pf_y_hist[-pf_ogp_max_hist:]

                w = pf.weights().view(-1, 1)
                mean_theta_s = (w * pf.theta).sum(dim=0)
                mean_theta_raw = gt.theta_s_to_raw(float(mean_theta_s[0]))
                var_theta_raw = float(
                    (w * (pf.theta - mean_theta_s).pow(2)).sum(dim=0)[0]
                ) * (gt.theta_sd ** 2)
                gt_theta_hist.append(float(np.mean(thb)))
                theta_hist.append(mean_theta_raw)
                theta_var_hist.append(var_theta_raw)
                restart_hist.append(False)

        elif meta.get("type") == "da":
            da = PFWithGPPrediction(
                sim_func_np=_sim_func_std_np,
                n_particles=int(meta.get("num_particles", 1024)),
                theta_lo=theta_lo_s,
                theta_hi=theta_hi_s,
                sigma_obs=float(meta.get("sigma_eps_s", _default_sigma_eps_s())),
                resample_ess_ratio=float(meta.get("resample_ess_ratio", 0.5)),
                theta_move_std=float(meta.get("theta_move_std_s", 0.5 / gt.theta_sd)),
                window_size=int(meta.get("window_size", 80)),
                gp_lengthscale=float(meta.get("gp_lengthscale", 0.3)),
                gp_signal_var=float(meta.get("gp_signal_var", 1.0)),
                seed=int(meta.get("seed", seed if seed is not None else 42 + int(mode))),
            )
            for Xb, yb, thb in batches(stream, batch_size, max_batches=max_batches):
                Xb_s = batch_X_base_to_s(gt, Xb).cpu().numpy()
                yb_s = batch_y_to_s(gt, yb).cpu().numpy()
                if idx > 0:
                    mu_s_np, var_s_np = da.predict(Xb_s)
                    mu_raw, var_raw = _raw_pred_from_standardized(gt, mu_s_np, var_s_np)
                    rmse = float(np.sqrt(np.mean((mu_raw - np.asarray(yb, dtype=float)) ** 2)))
                    rmse_hist.append(rmse)
                    y_crps_hist.append(_gaussian_crps_mean(mu_raw, var_raw, yb))
                    y_true_hist.extend(np.asarray(yb, dtype=float).reshape(-1).tolist())
                    y_pred_hist.extend(np.asarray(mu_raw, dtype=float).reshape(-1).tolist())
                    y_var_hist.extend(np.asarray(var_raw, dtype=float).reshape(-1).tolist())
                idx += 1
                da.update_batch(Xb_s, yb_s)
                theta_mean_s = da.mean_theta()
                theta_w = np.exp(da.logw.copy())
                theta_var_s = float(np.sum(theta_w * (da.theta - theta_mean_s) ** 2))
                gt_theta_hist.append(float(np.mean(thb)))
                theta_hist.append(float(np.asarray(gt.theta_s_to_raw(theta_mean_s)).reshape(-1)[0]))
                theta_var_hist.append(float(theta_var_s * (gt.theta_sd ** 2)))
                restart_hist.append(False)

        elif meta.get("type") == "paper_pf":
            sigma_eps_s = float(meta.get("sigma_eps_s", _default_sigma_eps_s()))
            pf_cfg = WardPaperPFConfig(
                num_particles=int(meta.get("num_particles", 1024)),
                theta_lo=float(theta_lo_s),
                theta_hi=float(theta_hi_s),
                sigma_obs_var=float(meta.get("paper_pf_sigma_obs_var", sigma_eps_s ** 2)),
                design_x_points=int(meta.get("paper_pf_design_x_points", 32)),
                design_theta_points=int(meta.get("paper_pf_design_theta_points", 7)),
                x_domain=[(-3.0, 3.0)] * 5,
                move_theta_std=float(meta.get("paper_pf_move_theta_std_s", 0.15 / gt.theta_sd)),
                move_logl_std=float(meta.get("paper_pf_move_logl_std", 0.10)),
                seed=int(meta.get("seed", seed if seed is not None else 42 + int(mode))),
            )
            da = WardPaperParticleFilter(sim_func_np=_sim_func_std_np, cfg=pf_cfg)
            for Xb, yb, thb in batches(stream, batch_size, max_batches=max_batches):
                Xb_s = batch_X_base_to_s(gt, Xb).cpu().numpy()
                yb_s = batch_y_to_s(gt, yb).cpu().numpy()
                if idx > 0:
                    mu_s_np, var_s_np = da.predict_batch(Xb_s)
                    mu_raw, var_raw = _raw_pred_from_standardized(gt, mu_s_np, var_s_np)
                    rmse = float(np.sqrt(np.mean((mu_raw - np.asarray(yb, dtype=float)) ** 2)))
                    rmse_hist.append(rmse)
                    y_crps_hist.append(_gaussian_crps_mean(mu_raw, var_raw, yb))
                    y_true_hist.extend(np.asarray(yb, dtype=float).reshape(-1).tolist())
                    y_pred_hist.extend(np.asarray(mu_raw, dtype=float).reshape(-1).tolist())
                    y_var_hist.extend(np.asarray(var_raw, dtype=float).reshape(-1).tolist())
                idx += 1
                da.step_batch(Xb_s, yb_s)
                theta_mean_s, theta_var_s = da.posterior_mean_var()
                gt_theta_hist.append(float(np.mean(thb)))
                theta_hist.append(float(np.asarray(gt.theta_s_to_raw(theta_mean_s)).reshape(-1)[0]))
                theta_var_hist.append(float(theta_var_s * (gt.theta_sd ** 2)))
                restart_hist.append(False)

        elif meta.get("type") == "bc":
            bc = KOHSlidingWindow(
                sim_func_np=_sim_func_std_np,
                theta_grid=np.linspace(theta_lo_s, theta_hi_s, int(meta.get("theta_grid_n", 200))),
                window_size=int(meta.get("window_size", 80)),
                sigma_obs=float(meta.get("sigma_eps_s", _default_sigma_eps_s())),
                gp_lengthscale=float(meta.get("gp_lengthscale", 0.3)),
                gp_signal_var=float(meta.get("gp_signal_var", 1.0)),
            )
            for Xb, yb, thb in batches(stream, batch_size, max_batches=max_batches):
                Xb_s = batch_X_base_to_s(gt, Xb).cpu().numpy()
                yb_s = batch_y_to_s(gt, yb).cpu().numpy()
                if idx > 0:
                    mu_s_np, var_s_np = bc.predict(Xb_s)
                    mu_raw, var_raw = _raw_pred_from_standardized(gt, mu_s_np, var_s_np)
                    rmse = float(np.sqrt(np.mean((mu_raw - np.asarray(yb, dtype=float)) ** 2)))
                    rmse_hist.append(rmse)
                    y_crps_hist.append(_gaussian_crps_mean(mu_raw, var_raw, yb))
                    y_true_hist.extend(np.asarray(yb, dtype=float).reshape(-1).tolist())
                    y_pred_hist.extend(np.asarray(mu_raw, dtype=float).reshape(-1).tolist())
                    y_var_hist.extend(np.asarray(var_raw, dtype=float).reshape(-1).tolist())
                idx += 1
                bc.update_batch(Xb_s, yb_s)
                gt_theta_hist.append(float(np.mean(thb)))
                theta_hist.append(float(np.asarray(gt.theta_s_to_raw(bc.mean_theta())).reshape(-1)[0]))
                theta_var_hist.append(float("nan"))
                restart_hist.append(False)

        # ---------- Existing BOCPD ----------
        else:
            cfg = CalibrationConfig()
            cfg.bocpd.bocpd_mode = meta["mode"]
            cfg.bocpd.use_restart = True
            cfg.bocpd.restart_impl = meta.get("restart_impl", "debug_260115")
            cfg.bocpd.controller_name = str(meta.get("controller_name", getattr(cfg.bocpd, "controller_name", "none")))
            cfg.bocpd.controller_stat = str(meta.get("controller_stat", getattr(cfg.bocpd, "controller_stat", "surprise_mean")))
            cfg.bocpd.controller_wcusum_warmup_batches = int(meta.get("controller_wcusum_warmup_batches", getattr(cfg.bocpd, "controller_wcusum_warmup_batches", 3)))
            cfg.bocpd.controller_wcusum_window = int(meta.get("controller_wcusum_window", getattr(cfg.bocpd, "controller_wcusum_window", 4)))
            cfg.bocpd.controller_wcusum_threshold = float(meta.get("controller_wcusum_threshold", getattr(cfg.bocpd, "controller_wcusum_threshold", 0.25)))
            cfg.bocpd.controller_wcusum_kappa = float(meta.get("controller_wcusum_kappa", getattr(cfg.bocpd, "controller_wcusum_kappa", 0.25)))
            cfg.bocpd.controller_wcusum_sigma_floor = float(meta.get("controller_wcusum_sigma_floor", getattr(cfg.bocpd, "controller_wcusum_sigma_floor", 0.25)))
            cfg.bocpd.hybrid_partial_restart = bool(meta.get("use_dual_restart", meta.get("hybrid_partial_restart", True)))
            cfg.bocpd.hybrid_tau_delta = float(meta.get("hybrid_tau_delta", 0.05))
            cfg.bocpd.hybrid_tau_theta = float(meta.get("hybrid_tau_theta", 0.05))
            cfg.bocpd.hybrid_tau_full = float(meta.get("hybrid_tau_full", 0.05))
            cfg.bocpd.hybrid_delta_share_rho = float(meta.get("hybrid_delta_share_rho", 0.75))
            cfg.bocpd.hybrid_pf_sigma_mode = str(meta.get("hybrid_pf_sigma_mode", "fixed"))
            cfg.bocpd.hybrid_sigma_delta_alpha = float(meta.get("hybrid_sigma_delta_alpha", 1.0))
            cfg.bocpd.hybrid_sigma_ema_beta = float(meta.get("hybrid_sigma_ema_beta", 0.98))
            cfg.bocpd.hybrid_sigma_min = float(meta.get("hybrid_sigma_min", 1e-4))
            cfg.bocpd.hybrid_sigma_max = float(meta.get("hybrid_sigma_max", 10.0))
            cfg.bocpd.use_cusum = bool(meta.get("use_cusum", False))
            cfg.bocpd.cusum_threshold = float(meta.get("cusum_threshold", 10.0))
            cfg.bocpd.cusum_recent_obs = int(meta.get("cusum_recent_obs", 20))
            cfg.bocpd.cusum_cov_eps = float(meta.get("cusum_cov_eps", 1e-6))
            cfg.bocpd.cusum_mode = str(meta.get("cusum_mode", "cumulative"))
            cfg.bocpd.standardized_gate_threshold = float(meta.get("standardized_gate_threshold", 3.0))
            cfg.bocpd.standardized_gate_consecutive = int(meta.get("standardized_gate_consecutive", 1))
            cfg.model.sigma_eps = float(meta.get("sigma_eps_s", cfg.model.sigma_eps))
            cfg.bocpd.particle_delta_mode = str(meta.get("particle_delta_mode", "shared_gp"))
            cfg.bocpd.particle_gp_hyper_candidates = meta.get("particle_gp_hyper_candidates", None)
            cfg.bocpd.particle_basis_kind = str(meta.get("particle_basis_kind", "rbf"))
            cfg.bocpd.particle_basis_num_features = int(meta.get("particle_basis_num_features", 8))
            cfg.bocpd.particle_basis_lengthscale = float(meta.get("particle_basis_lengthscale", 0.25))
            cfg.bocpd.particle_basis_ridge = float(meta.get("particle_basis_ridge", 1e-2))
            cfg.bocpd.particle_basis_noise = float(meta.get("particle_basis_noise", cfg.model.delta_kernel.noise))
            cfg.pf.num_particles = int(meta.get("num_particles", 1024))
            cfg.model.use_discrepancy = bool(meta.get("use_discrepancy", False))
            cfg.model.bocpd_use_discrepancy = bool(meta.get("bocpd_use_discrepancy", cfg.model.use_discrepancy))
            cfg.model.delta_update_mode = str(meta.get("delta_update_mode", getattr(cfg.model, "delta_update_mode", "refit")))
            cfg.model.delta_bpc_obs_noise_mode = str(meta.get("delta_bpc_obs_noise_mode", getattr(cfg.model, "delta_bpc_obs_noise_mode", "kernel")))
            cfg.model.delta_bpc_predict_add_kernel_noise = bool(meta.get("delta_bpc_predict_add_kernel_noise", getattr(cfg.model, "delta_bpc_predict_add_kernel_noise", True)))

            calib = OnlineBayesCalibrator(cfg, emulator, prior_sampler)
            for Xb, yb, thb in tqdm(batches(stream, batch_size, max_batches=max_batches), desc=f"Running {name}"):
                newX = batch_X_base_to_s(gt, Xb)    # (B,5) standardized; DO NOT include thb
                newY = batch_y_to_s(gt, yb)         # (B,) standardized

                if idx > 0:
                    pred = calib.predict_batch(newX)           # pred["mu"] is y_s
                    pred_comp = calib.predict_complete(newX, newY)
                    mu_raw, var_raw = _raw_pred_from_standardized(gt, pred["mu"], pred["var"])
                    rmse = float(np.sqrt(np.mean((mu_raw - np.asarray(yb))**2)))
                    rmse_hist.append(rmse)
                    y_crps_hist.append(_gaussian_crps_mean_raw(gt, pred["mu"], pred["var"], yb))
                    y_true_hist.extend(np.asarray(yb, dtype=float).reshape(-1).tolist())
                    y_pred_hist.extend(np.asarray(mu_raw, dtype=float).reshape(-1).tolist())
                    y_var_hist.extend(np.asarray(var_raw, dtype=float).reshape(-1).tolist())
                    report_sub_hist = (pred_comp["crps_sim"].item(),pred_comp["experts_logpred"],pred_comp["var_sim"])
                    comp_rmse_hist.append(report_sub_hist)

                idx += 1

                rec = calib.step_batch(newX, newY, verbose=False)

                mean_theta_s, var_theta_s, lo_s, hi_s = calib._aggregate_particles(0.9)
                mean_theta_raw = gt.theta_s_to_raw(mean_theta_s.item())
                var_theta_raw = var_theta_s * (gt.theta_sd ** 2)

                gt_theta_hist.append(float(np.mean(thb)))      # raw minutes
                theta_hist.append(mean_theta_raw.item())       # raw minutes
                theta_var_hist.append(float(var_theta_raw))    # raw^2
                restart_hist.append(rec["did_restart"])

            
            # newX = torch.tensor(Xb)
            # theta = torch.tensor(thb)
            # newY = torch.tensor(yb)
            # # X_torch, Y_torch = X_torch[:batch_size,:], Y_torch[:batch_size]

            # if idx > 0: 
            #     pred = calib.predict_batch(newX)
            #     rmse_hist.append(torch.sqrt(((pred["mu"] - newY)**2).mean()))
            #     pred_comp = calib.predict_complete(newX, newY)
            #     report_sub_hist = (pred_comp["crps_sim"].item(),pred_comp["experts_logpred"],pred_comp["var_sim"])
            #     comp_rmse_hist.append(report_sub_hist)
            # idx += 1
            
            # rec = calib.step_batch(newX, newY, verbose=False)
            # mean_theta, var_theta, lo_theta, hi_theta = calib._aggregate_particles(0.9)
            # gt_theta_hist.append(theta.mean().item())
            # theta_hist.append(mean_theta.item())
            # theta_var_hist.append(var_theta)

            # restart_hist.append(rec["did_restart"])

        results[name] = dict(
            theta_hist=theta_hist,
            theta_var_hist=theta_var_hist,
            gt_theta_hist=gt_theta_hist,
            rmse_hist=rmse_hist,
            y_crps_hist=y_crps_hist,
            comp_rmse_hist=comp_rmse_hist,
            restart_hist=restart_hist,
            restart_count=int(np.asarray(restart_hist, dtype=bool).sum()) if len(restart_hist) > 0 else 0,
            runtime_sec=float(time() - t_start),
            y_true_hist=y_true_hist,
            y_pred_hist=y_pred_hist,
            y_var_hist=y_var_hist,
        )
    return results

if __name__ == "__main__":
    import argparse
    import csv

    parser = argparse.ArgumentParser(description="Plant Simulation Calibration Experiment (Standardized)")
    parser.add_argument("--out_dir", type=str, default="figs/plantSim/v3_std", help="Output directory for figures")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to PhysicalData directory (Excel files)")
    parser.add_argument("--csv", type=str, default=None, help="Path to aggregated CSV file")
    parser.add_argument("--modes", type=int, nargs="+", default=[1, 2, 0], help="Modes to run (e.g., --modes 0 1 2)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Optional seed sweep; if omitted, runs a single pass")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_particles", type=int, default=1024, help="Number of PF particles for BOCPD/wCUSUM methods")
    parser.add_argument("--profile", type=str, default="core", choices=["core", "cpd_ablation", "cpd_ablation_plus", "half_exact_ogp", "half_exact_da_bc", "half_exact_damove_bc"], help="Method bundle to run")
    parser.add_argument("--max-batches-by-mode", type=str, nargs="*", default=None, help="Optional per-mode batch caps like 1:250 2:100")
    args = parser.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    init_pipeline()

    methods = build_methods(args.profile)
    max_batches_by_mode = _parse_max_batches_by_mode(args.max_batches_by_mode)
    for meta in methods.values():
        meta.setdefault("num_particles", int(args.num_particles))

    all_results = {}
    summary_rows = []
    seed_values = [None] if not args.seeds else [int(s) for s in args.seeds]
    multi_seed = len(seed_values) > 1
    if multi_seed:
        os.makedirs(os.path.join(out_dir, "seed_runs"), exist_ok=True)

    for seed in seed_values:
        for mode in args.modes:
            for bs in [args.batch_size]:
                mode_max_batches = max_batches_by_mode.get(int(mode))
                results = run_plantSim(
                    mode=mode,
                    methods=methods,
                    batch_size=bs,
                    data_dir=args.data_dir,
                    csv_path=args.csv,
                    seed=seed,
                    max_batches=mode_max_batches,
                )
                run_key = f"mode{mode}_bs{bs}" if seed is None else f"seed{int(seed)}_mode{mode}_bs{bs}"
                all_results[run_key] = results

                print("\n" + "=" * 70)
                header = f"Mode={mode}, batch_size={bs} metrics" if seed is None else f"Seed={seed}, mode={mode}, batch_size={bs} metrics"
                print(header)
                print("=" * 70)
                for name, result in results.items():
                    metrics = summarize_metrics(result)
                    summary_rows.append(dict(
                        seed=(None if seed is None else int(seed)),
                        mode=int(mode),
                        batch_size=int(bs),
                        num_particles=int(args.num_particles),
                        profile=str(args.profile),
                        max_batches=(None if mode_max_batches is None else int(mode_max_batches)),
                        method=str(name),
                        theta_rmse=float(metrics["theta_rmse"]),
                        theta_crps=float(metrics["theta_crps"]),
                        y_rmse=float(metrics["y_rmse"]),
                        y_crps=float(metrics["y_crps"]),
                        y_rel_rmse=float(metrics["y_rel_rmse"]),
                        y_rel_crps=float(metrics["y_rel_crps"]),
                        restart_count=float(metrics["restart_count"]),
                        runtime_sec=float(metrics["runtime_sec"]),
                    ))
                    print(
                        f"{name}: "
                        f"theta_rmse={metrics['theta_rmse']:.6f}, "
                        f"theta_crps={metrics['theta_crps']:.6f}, "
                        f"y_rmse={metrics['y_rmse']:.6f}, "
                        f"y_crps={metrics['y_crps']:.6f}, "
                        f"y_rel_rmse={metrics['y_rel_rmse']:.6f}, "
                        f"y_rel_crps={metrics['y_rel_crps']:.6f}, "
                        f"restart_count={metrics['restart_count']:.0f}, "
                        f"runtime_sec={metrics['runtime_sec']:.2f}"
                    )

                if not multi_seed:
                    plt.figure(figsize=(10, 5))
                    for name, result in results.items():
                        plt.plot(result["theta_hist"], label=name)
                    plt.plot(result["gt_theta_hist"], "k--", lw=2, label="oracle ?*")
                    plot_title = (
                        f"Theta tracking (seed={seed}, mode={mode}, batch size={bs})"
                        if seed is not None else
                        f"Theta tracking (mode={mode}, batch size={bs})"
                    )
                    plot_name = (
                        f"seed{int(seed):02d}_mode{mode}_bs{bs}_theta.png"
                        if seed is not None else
                        f"mode{mode}_bs{bs}_theta.png"
                    )
                    plt.title(plot_title)
                    plt.xlabel("batch index")
                    plt.ylabel("theta")
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(os.path.join(out_dir, plot_name), dpi=300)
                    plt.close()

                if seed is None:
                    torch.save(results, f"{out_dir}/plantSim_results_mode{mode}.pt")
                else:
                    seed_dir = os.path.join(out_dir, "seed_runs", f"seed{int(seed):02d}")
                    os.makedirs(seed_dir, exist_ok=True)
                    torch.save(results, os.path.join(seed_dir, f"plantSim_results_mode{mode}.pt"))

    if len(summary_rows) > 0:
        summary_seed_csv = os.path.join(out_dir, "plant_method_mode_seed_summary.csv")
        with open(summary_seed_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\nSaved per-seed summary CSV to {summary_seed_csv}")

        if multi_seed:
            import pandas as pd

            df_summary = pd.DataFrame(summary_rows)
            group_cols = ["mode", "batch_size", "num_particles", "profile", "max_batches", "method"]
            agg_cols = ["theta_rmse", "theta_crps", "y_rmse", "y_crps", "y_rel_rmse", "y_rel_crps", "restart_count", "runtime_sec"]
            df_mean = df_summary.groupby(group_cols, as_index=False)[agg_cols].mean()
            summary_csv = os.path.join(out_dir, "plant_method_mode_summary.csv")
            df_mean.to_csv(summary_csv, index=False)
            print(f"Saved aggregated summary CSV to {summary_csv}")
        else:
            summary_csv = os.path.join(out_dir, "plant_method_mode_summary.csv")
            with open(summary_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
                writer.writeheader()
                writer.writerows(summary_rows)
            print(f"Saved summary CSV to {summary_csv}")
