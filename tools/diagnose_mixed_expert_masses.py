from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from calib.configs import CalibrationConfig
from calib.emulator import DeterministicSimulator
from calib.online_calibrator import OnlineBayesCalibrator
from calib.run_synthetic_mixed_thetaCmp import MixedThetaDataStream
from calib.run_synthetic_slope_deltaCmp import build_phi2_from_theta_star, computer_model_config2_torch


def build_methods() -> Dict[str, Dict[str, Any]]:
    base = dict(
        type="bocpd",
        use_discrepancy=False,
        bocpd_use_discrepancy=True,
        delta_bpc_obs_noise_mode="sigma_eps",
        delta_bpc_predict_add_kernel_noise=False,
    )
    return {
        "Proxy_BOCPD_restart": dict(base, mode="restart", delta_update_mode="online_bpc_proxy_stablemean"),
        "Proxy_BOCPD_standard": dict(base, mode="standard", delta_update_mode="online_bpc_proxy_stablemean"),
        "Exact_BOCPD_restart": dict(base, mode="restart", delta_update_mode="online_bpc_exact"),
        "Exact_BOCPD_standard": dict(base, mode="standard", delta_update_mode="online_bpc_exact"),
    }


def build_config(meta: Dict[str, Any], num_particles: int) -> CalibrationConfig:
    cfg = CalibrationConfig()
    cfg.bocpd.bocpd_mode = str(meta.get("mode", "restart"))
    cfg.bocpd.use_restart = True
    cfg.bocpd.restart_impl = str(meta.get("restart_impl", "debug_260115"))
    cfg.bocpd.controller_name = str(meta.get("controller_name", "none"))
    cfg.bocpd.controller_stat = str(meta.get("controller_stat", "surprise_mean"))
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
    cfg.bocpd.particle_delta_mode = str(meta.get("particle_delta_mode", "shared_gp"))
    cfg.bocpd.particle_gp_hyper_candidates = meta.get("particle_gp_hyper_candidates", None)
    cfg.bocpd.particle_basis_kind = str(meta.get("particle_basis_kind", "rbf"))
    cfg.bocpd.particle_basis_num_features = int(meta.get("particle_basis_num_features", 8))
    cfg.bocpd.particle_basis_lengthscale = float(meta.get("particle_basis_lengthscale", 0.25))
    cfg.bocpd.particle_basis_ridge = float(meta.get("particle_basis_ridge", 1e-2))

    cfg.pf.num_particles = int(num_particles)
    cfg.model.use_discrepancy = bool(meta.get("use_discrepancy", False))
    cfg.model.bocpd_use_discrepancy = bool(meta.get("bocpd_use_discrepancy", cfg.model.use_discrepancy))
    cfg.model.delta_update_mode = str(meta.get("delta_update_mode", "refit"))
    cfg.model.delta_bpc_obs_noise_mode = str(meta.get("delta_bpc_obs_noise_mode", "kernel"))
    cfg.model.delta_bpc_predict_add_kernel_noise = bool(meta.get("delta_bpc_predict_add_kernel_noise", True))
    return cfg


def prior_sampler(N: int, **_: Any) -> torch.Tensor:
    return torch.rand(int(N), 1) * 3.0


def safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def expert_rows_from_state(calib: OnlineBayesCalibrator, method: str, batch_idx: int, phase: str,
                           pre_logpred: List[Dict[str, Any]] | None = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    experts = list(getattr(calib.bocpd, "experts", []))
    for expert_idx, e in enumerate(experts):
        try:
            w = e.pf.particles.weights()
            theta = e.pf.particles.theta
            theta_mean = float((w[:, None] * theta).sum(dim=0)[0].detach().cpu().item())
        except Exception:
            theta_mean = float("nan")
        row = dict(
            method=method,
            batch_idx=int(batch_idx),
            phase=phase,
            expert_idx=int(expert_idx),
            run_length=int(getattr(e, "run_length", -1)),
            log_mass=safe_float(getattr(e, "log_mass", float("nan"))),
            mass=float(math.exp(safe_float(getattr(e, "log_mass", float("-inf"))))) if np.isfinite(safe_float(getattr(e, "log_mass", float("nan")))) else 0.0,
            theta_mean=theta_mean,
            logp=float("nan"),
        )
        if pre_logpred is not None and expert_idx < len(pre_logpred):
            row["logp"] = safe_float(pre_logpred[expert_idx].get("logp", float("nan")))
        rows.append(row)
    return rows


def decision_rows_from_restart(rec: Dict[str, Any], method: str, batch_idx: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for info in rec.get("experts_debug", []):
        theta_mean = info.get("theta_mean", None)
        if isinstance(theta_mean, list) and len(theta_mean) > 0:
            theta_mean_val = safe_float(theta_mean[0])
        else:
            theta_mean_val = float("nan")
        rows.append(
            dict(
                method=method,
                batch_idx=int(batch_idx),
                phase="decision",
                expert_idx=int(info.get("index", -1)),
                run_length=int(info.get("run_length", -1)),
                log_mass=safe_float(info.get("log_mass", float("nan"))),
                mass=safe_float(info.get("mass", float("nan"))),
                theta_mean=theta_mean_val,
                logp=float("nan"),
            )
        )
    return rows


def closest_row(rows: List[Dict[str, Any]], target_rl: float) -> Dict[str, Any] | None:
    valid = [r for r in rows if np.isfinite(r.get("run_length", np.nan))]
    if len(valid) == 0 or not np.isfinite(target_rl):
        return None
    return min(valid, key=lambda r: abs(float(r["run_length"]) - float(target_rl)))


def batch_summary_from_rows(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    if len(rows) == 0:
        return dict(top_mass=float("nan"), second_mass=float("nan"),
                    dominant_rl=float("nan"), dominant_theta=float("nan"),
                    num_experts=0)
    rows_sorted = sorted(rows, key=lambda r: safe_float(r.get("mass", float("nan"))), reverse=True)
    top = rows_sorted[0]
    second = rows_sorted[1] if len(rows_sorted) > 1 else None
    return dict(
        top_mass=safe_float(top.get("mass", float("nan"))),
        second_mass=safe_float(second.get("mass", float("nan"))) if second is not None else float("nan"),
        dominant_rl=safe_float(top.get("run_length", float("nan"))),
        dominant_theta=safe_float(top.get("theta_mean", float("nan"))),
        num_experts=float(len(rows)),
    )


def plot_method(batch_df: pd.DataFrame, expert_df: pd.DataFrame, out_dir: Path, method: str) -> None:
    sub = batch_df[batch_df["method"] == method].copy()
    exp_sub = expert_df[expert_df["method"] == method].copy()
    if sub.empty:
        return

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(sub["batch_idx"], sub["theta_true"], color="black", linewidth=2, label="true theta*")
    if "anchor_theta_decision" in sub.columns and sub["anchor_theta_decision"].notna().any():
        axes[0].plot(sub["batch_idx"], sub["anchor_theta_decision"], color="tab:blue", label="anchor theta")
    if "cand_theta_decision" in sub.columns and sub["cand_theta_decision"].notna().any():
        axes[0].plot(sub["batch_idx"], sub["cand_theta_decision"], color="tab:red", linestyle="--", label="candidate theta")
    axes[0].plot(sub["batch_idx"], sub["dominant_theta_post"], color="tab:green", alpha=0.8, label="dominant theta (post)")
    restart_idx = sub.loc[sub["did_restart"] > 0.5, "batch_idx"].to_numpy()
    for b in restart_idx:
        axes[0].axvline(b, color="red", alpha=0.15)
    axes[0].set_ylabel("theta")
    axes[0].legend(loc="best")
    axes[0].set_title(method)

    axes[1].plot(sub["batch_idx"], sub["top_mass_decision"], label="decision top mass", color="tab:blue")
    axes[1].plot(sub["batch_idx"], sub["second_mass_decision"], label="decision 2nd mass", color="tab:orange")
    if "anchor_mass_decision" in sub.columns and sub["anchor_mass_decision"].notna().any():
        axes[1].plot(sub["batch_idx"], sub["anchor_mass_decision"], label="anchor mass", color="tab:green")
    if "cand_mass_decision" in sub.columns and sub["cand_mass_decision"].notna().any():
        axes[1].plot(sub["batch_idx"], sub["cand_mass_decision"], label="candidate mass", color="tab:red", linestyle="--")
    axes[1].set_ylabel("mass")
    axes[1].legend(loc="best")

    axes[2].plot(sub["batch_idx"], sub["dominant_rl_post"], label="dominant RL (post)", color="tab:green")
    if "anchor_rl" in sub.columns and sub["anchor_rl"].notna().any():
        axes[2].plot(sub["batch_idx"], sub["anchor_rl"], label="anchor RL", color="tab:blue")
    if "cand_rl" in sub.columns and sub["cand_rl"].notna().any():
        axes[2].plot(sub["batch_idx"], sub["cand_rl"], label="candidate RL", color="tab:red", linestyle="--")
    if "log_odds_mass" in sub.columns and sub["log_odds_mass"].notna().any():
        ax2b = axes[2].twinx()
        ax2b.plot(sub["batch_idx"], sub["log_odds_mass"], color="tab:purple", alpha=0.7, label="log_odds_mass")
        ax2b.set_ylabel("log odds")
    axes[2].set_ylabel("run length")
    axes[2].set_xlabel("batch index")
    axes[2].legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(out_dir / f"{method}_diagnostic.png", dpi=220)
    plt.close(fig)

    phase = "decision" if (exp_sub["phase"] == "decision").any() else "post"
    heat = exp_sub[exp_sub["phase"] == phase].copy()
    if not heat.empty:
        pivot = heat.pivot_table(index="run_length", columns="batch_idx", values="mass", aggfunc="sum", fill_value=0.0)
        fig2, ax = plt.subplots(figsize=(12, 5))
        im = ax.imshow(pivot.to_numpy(), aspect="auto", origin="lower", interpolation="nearest")
        ax.set_title(f"{method} mass heatmap ({phase})")
        ax.set_xlabel("batch index")
        ax.set_ylabel("run length")
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels([str(int(v)) for v in pivot.index])
        fig2.colorbar(im, ax=ax, label="mass")
        fig2.tight_layout()
        fig2.savefig(out_dir / f"{method}_mass_heatmap_{phase}.png", dpi=220)
        plt.close(fig2)


def run_method(method: str, meta: Dict[str, Any], args: argparse.Namespace, phi2_of_theta) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cfg = build_config(meta, args.num_particles)
    emulator = DeterministicSimulator(func=computer_model_config2_torch, enable_autograd=True)
    calib = OnlineBayesCalibrator(cfg, emulator, prior_sampler)
    stream = MixedThetaDataStream(
        total_T=args.total_T,
        batch_size=args.batch_size,
        noise_sd=args.noise_sd,
        phi2_of_theta=phi2_of_theta,
        drift_scale=args.drift_scale,
        jump_scale=args.jump_scale,
        theta_noise_sd=args.theta_noise_sd,
        seed=args.seed,
    )

    batch_rows: List[Dict[str, Any]] = []
    expert_rows: List[Dict[str, Any]] = []
    total_obs = 0
    batch_idx = 0
    while total_obs < args.total_T:
        Xb, Yb = stream.next()
        theta_true = float(stream.theta_star_history[-1])
        cp_flag = int(batch_idx in set(stream.cp_batches))

        pred_complete = calib.predict_complete(Xb, Yb) if total_obs > 0 else {"experts_logpred": []}
        pre_rows = expert_rows_from_state(calib, method, batch_idx, "pre", pre_logpred=pred_complete.get("experts_logpred", []))
        expert_rows.extend(pre_rows)
        pre_stats = batch_summary_from_rows(pre_rows)

        rec = calib.step_batch(Xb, Yb, verbose=False)

        decision_rows = decision_rows_from_restart(rec, method, batch_idx) if meta["mode"] == "restart" else []
        expert_rows.extend(decision_rows)
        post_rows = expert_rows_from_state(calib, method, batch_idx, "post")
        expert_rows.extend(post_rows)
        post_stats = batch_summary_from_rows(post_rows)
        decision_stats = batch_summary_from_rows(decision_rows if len(decision_rows) > 0 else post_rows)

        anchor_row = closest_row(decision_rows, safe_float(rec.get("anchor_rl", float("nan"))))
        cand_row = closest_row(decision_rows, safe_float(rec.get("cand_rl", float("nan"))))

        batch_rows.append(
            dict(
                method=method,
                batch_idx=batch_idx,
                obs_t=total_obs,
                theta_true=theta_true,
                seg_id=int(stream.seg_history[-1]),
                cp_flag=cp_flag,
                did_restart=float(bool(rec.get("did_restart", False))) if meta["mode"] == "restart" else 0.0,
                restart_mode=str(rec.get("restart_mode", "none")) if meta["mode"] == "restart" else "standard",
                anchor_rl=safe_float(rec.get("anchor_rl", float("nan"))),
                cand_rl=safe_float(rec.get("cand_rl", float("nan"))),
                log_odds_mass=safe_float(rec.get("log_odds_mass", float("nan"))),
                p_cp=safe_float(rec.get("p_cp", float("nan"))),
                top_mass_pre=pre_stats["top_mass"],
                second_mass_pre=pre_stats["second_mass"],
                dominant_rl_pre=pre_stats["dominant_rl"],
                dominant_theta_pre=pre_stats["dominant_theta"],
                num_experts_pre=pre_stats["num_experts"],
                top_mass_decision=decision_stats["top_mass"],
                second_mass_decision=decision_stats["second_mass"],
                dominant_rl_decision=decision_stats["dominant_rl"],
                dominant_theta_decision=decision_stats["dominant_theta"],
                num_experts_decision=decision_stats["num_experts"],
                top_mass_post=post_stats["top_mass"],
                second_mass_post=post_stats["second_mass"],
                dominant_rl_post=post_stats["dominant_rl"],
                dominant_theta_post=post_stats["dominant_theta"],
                num_experts_post=post_stats["num_experts"],
                anchor_theta_decision=safe_float(anchor_row.get("theta_mean")) if anchor_row is not None else float("nan"),
                cand_theta_decision=safe_float(cand_row.get("theta_mean")) if cand_row is not None else float("nan"),
                anchor_mass_decision=safe_float(anchor_row.get("mass")) if anchor_row is not None else float("nan"),
                cand_mass_decision=safe_float(cand_row.get("mass")) if cand_row is not None else float("nan"),
            )
        )

        batch_idx += 1
        total_obs += args.batch_size

    return pd.DataFrame(batch_rows), pd.DataFrame(expert_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=202)
    parser.add_argument("--jump-scale", type=float, default=0.38)
    parser.add_argument("--drift-scale", type=float, default=0.008)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--total-T", type=int, default=600)
    parser.add_argument("--theta-noise-sd", type=float, default=0.015)
    parser.add_argument("--noise-sd", type=float, default=0.2)
    parser.add_argument("--num-particles", type=int, default=1024)
    parser.add_argument("--out-dir", type=str, default="figs/mixed_expert_mass_diag")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    phi2_of_theta, _ = build_phi2_from_theta_star(
        phi2_grid=np.linspace(3.0, 12.0, 300),
        theta_grid=np.linspace(0.0, 3.0, 600),
    )
    methods = build_methods()

    batch_frames = []
    expert_frames = []
    for method, meta in methods.items():
        print(f"Running diagnostic method: {method}")
        batch_df, expert_df = run_method(method, meta, args, phi2_of_theta)
        batch_frames.append(batch_df)
        expert_frames.append(expert_df)
        plot_method(batch_df, expert_df, out_dir, method)

    all_batch = pd.concat(batch_frames, ignore_index=True)
    all_expert = pd.concat(expert_frames, ignore_index=True)
    all_batch.to_csv(out_dir / "batch_summary.csv", index=False)
    all_expert.to_csv(out_dir / "expert_summary.csv", index=False)

    per_method = (
        all_batch.groupby("method", dropna=False)
        .agg(
            restart_count=("did_restart", "sum"),
            mean_num_experts_post=("num_experts_post", "mean"),
            mean_top_mass_post=("top_mass_post", "mean"),
            mean_log_odds_mass=("log_odds_mass", "mean"),
        )
        .reset_index()
    )
    per_method.to_csv(out_dir / "method_overview.csv", index=False)
    print(f"Saved diagnostic outputs to {out_dir}")


if __name__ == "__main__":
    main()
