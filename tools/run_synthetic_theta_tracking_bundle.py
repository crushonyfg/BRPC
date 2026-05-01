import argparse
import os
import sys
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from calib.online_calibrator import crps_gaussian
from calib.run_synthetic_mixed_thetaCmp import (
    build_phi2_from_theta_star,
    run_one_mixed,
)
from calib.run_synthetic_slope_deltaCmp import run_one_slope
from calib.run_synthetic_suddenCmp_tryThm import run_one_sudden


LABELS = {
    "B-BRPC-RRA": "B-BRPC-RRA",
    "B-BRPC-E": "B-BRPC-E",
    "C-BRPC-E": "C-BRPC-E",
    "DA": "DA",
    "BC": "BC",
}

PLOT_ORDER = ["B-BRPC-RRA", "B-BRPC-E", "C-BRPC-E", "DA", "BC"]


def _build_methods(num_particles: int, delta_bpc_lambda: float) -> Dict[str, Dict]:
    exact_base = dict(
        type="bocpd",
        use_discrepancy=False,
        bocpd_use_discrepancy=True,
        delta_update_mode="online_bpc_exact",
        delta_bpc_obs_noise_mode="sigma_eps",
        delta_bpc_predict_add_kernel_noise=False,
        delta_bpc_lambda=float(delta_bpc_lambda),
        num_particles=int(num_particles),
    )
    return {
        "B-BRPC-RRA": dict(
            type="bocpd",
            mode="restart",
            use_discrepancy=False,
            bocpd_use_discrepancy=True,
            num_particles=int(num_particles),
        ),
        "B-BRPC-E": dict(
            **exact_base,
            mode="restart",
        ),
        "C-BRPC-E": dict(
            **exact_base,
            mode="wcusum",
            controller_name="wcusum",
            controller_stat="log_surprise_mean",
            controller_wcusum_warmup_batches=3,
            controller_wcusum_window=4,
            controller_wcusum_threshold=0.25,
            controller_wcusum_kappa=0.25,
            controller_wcusum_sigma_floor=0.25,
        ),
        "DA": dict(
            type="paper_pf",
            num_particles=int(num_particles),
            paper_pf_sigma_obs_var=0.04,
            paper_pf_move_theta_std=0.15,
            paper_pf_move_logl_std=0.10,
        ),
        "BC": dict(type="bc"),
    }


def _scenario_tracking_df(
    res: Dict[str, dict],
    theta_ref: np.ndarray,
) -> pd.DataFrame:
    rows: List[dict] = []
    n_ref = int(len(theta_ref))
    for method_name in PLOT_ORDER:
        if method_name not in res:
            continue
        theta = np.asarray(res[method_name]["theta"], dtype=float)
        for batch_idx, value in enumerate(theta):
            rows.append(
                {
                    "batch_idx": int(batch_idx),
                    "series": LABELS[method_name],
                    "theta": float(value),
                }
            )
    for batch_idx, value in enumerate(theta_ref[:n_ref]):
        rows.append(
            {
                "batch_idx": int(batch_idx),
                "series": "Ground Truth",
                "theta": float(value),
            }
        )
    return pd.DataFrame(rows)


def _summarize_metrics(res: Dict[str, dict], scenario: str) -> pd.DataFrame:
    rows: List[dict] = []
    for method_name in PLOT_ORDER:
        if method_name not in res:
            continue
        data = res[method_name]
        theta = np.asarray(data.get("theta", []), dtype=float)
        theta_ref = np.asarray(data.get("theta_oracle", []), dtype=float)
        theta_var = np.asarray(data.get("theta_var", []), dtype=float)
        rmse_hist = np.asarray(data.get("rmse", []), dtype=float)
        crps_hist = np.asarray(data.get("crps_hist", []), dtype=float)
        n = min(len(theta), len(theta_ref), len(theta_var))
        theta_rmse = float(np.sqrt(np.mean((theta[:n] - theta_ref[:n]) ** 2))) if n > 0 else float("nan")
        theta_crps = (
            float(
                crps_gaussian(
                    torch.tensor(theta[:n], dtype=torch.float64),
                    torch.tensor(np.clip(theta_var[:n], 1e-12, None), dtype=torch.float64),
                    torch.tensor(theta_ref[:n], dtype=torch.float64),
                ).mean().item()
            )
            if n > 0
            else float("nan")
        )
        rows.append(
            {
                "scenario": scenario,
                "method": method_name,
                "legend": LABELS[method_name],
                "theta_rmse": theta_rmse,
                "theta_crps": theta_crps,
                "y_rmse": float(np.nanmean(rmse_hist)) if rmse_hist.size > 0 else float("nan"),
                "y_crps": float(np.nanmean(crps_hist)) if crps_hist.size > 0 else float("nan"),
                "restart_count": float(sum(1 for v in data.get("restart_mode_hist", []) if v != "none")),
                "elapsed_sec": float(data.get("elapsed_sec", float("nan"))),
            }
        )
    return pd.DataFrame(rows)


def _plot_tracking(
    res: Dict[str, dict],
    theta_ref: np.ndarray,
    cp_times: Optional[List[int]],
    batch_size: int,
    title: str,
    save_path: str,
) -> None:
    plt.figure(figsize=(12, 5))
    for method_name in PLOT_ORDER:
        if method_name not in res:
            continue
        theta = np.asarray(res[method_name]["theta"], dtype=float)
        plt.plot(theta, label=LABELS[method_name], linewidth=2)
    plt.plot(np.asarray(theta_ref, dtype=float), "k--", linewidth=2.5, label="Ground Truth")
    if cp_times:
        for cp_t in cp_times:
            plt.axvline(cp_t // batch_size, color="tab:red", linestyle="--", alpha=0.35)
    plt.xlabel("Batch Index")
    plt.ylabel("Theta")
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--num-particles", type=int, default=1024)
    parser.add_argument("--delta-bpc-lambda", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--sudden-seed", type=int, default=13)
    parser.add_argument("--slope-seed", type=int, default=13)
    parser.add_argument("--mixed-seed", type=int, default=202)
    parser.add_argument("--sudden-mag", type=float, default=2.0)
    parser.add_argument("--sudden-seg-len", type=int, default=120)
    parser.add_argument("--slope", type=float, default=0.0015)
    parser.add_argument("--mixed-drift-scale", type=float, default=0.008)
    parser.add_argument("--mixed-jump-scale", type=float, default=0.38)
    parser.add_argument("--mixed-total-T", type=int, default=600)
    args = parser.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    methods = _build_methods(args.num_particles, args.delta_bpc_lambda)
    summary_frames: List[pd.DataFrame] = []
    config_rows: List[dict] = []

    # Sudden
    sudden_res, sudden_phi_hist, sudden_oracle_hist = run_one_sudden(
        seg_len_L=int(args.sudden_seg_len),
        delta_mag=float(args.sudden_mag),
        methods=methods,
        batch_size=int(args.batch_size),
        seed=int(args.sudden_seed),
        out_dir=out_dir,
    )
    sudden_cp_times = [
        int(args.sudden_seg_len),
        int(2 * args.sudden_seg_len),
        int(3 * args.sudden_seg_len),
    ]
    torch.save(
        sudden_res,
        os.path.join(
            out_dir,
            f"sudden_mag{args.sudden_mag:.3f}_seg{args.sudden_seg_len}_bs{args.batch_size}_seed{args.sudden_seed}_results.pt",
        ),
    )
    torch.save(
        dict(phi_hist=sudden_phi_hist, oracle_hist=sudden_oracle_hist, cp_times=sudden_cp_times),
        os.path.join(
            out_dir,
            f"sudden_mag{args.sudden_mag:.3f}_seg{args.sudden_seg_len}_bs{args.batch_size}_seed{args.sudden_seed}_meta.pt",
        ),
    )
    sudden_df = _scenario_tracking_df(sudden_res, np.asarray(sudden_oracle_hist, dtype=float))
    sudden_df.to_csv(
        os.path.join(
            out_dir,
            f"sudden_mag{args.sudden_mag:.3f}_seg{args.sudden_seg_len}_bs{args.batch_size}_seed{args.sudden_seed}_theta_tracking.csv",
        ),
        index=False,
    )
    _plot_tracking(
        sudden_res,
        np.asarray(sudden_oracle_hist, dtype=float),
        sudden_cp_times,
        int(args.batch_size),
        f"Sudden Theta Tracking (mag={args.sudden_mag:.2f}, seg_len={args.sudden_seg_len}, seed={args.sudden_seed})",
        os.path.join(
            out_dir,
            f"sudden_mag{args.sudden_mag:.3f}_seg{args.sudden_seg_len}_bs{args.batch_size}_seed{args.sudden_seed}_theta_tracking.png",
        ),
    )
    summary_frames.append(_summarize_metrics(sudden_res, "sudden"))
    config_rows.append(
        {
            "scenario": "sudden",
            "seed": int(args.sudden_seed),
            "batch_size": int(args.batch_size),
            "magnitude": float(args.sudden_mag),
            "seg_len": int(args.sudden_seg_len),
        }
    )

    # Slope
    slope_res, slope_phi_hist, slope_oracle_hist = run_one_slope(
        slope=float(args.slope),
        methods=methods,
        total_T=600,
        batch_size=int(args.batch_size),
        seed=int(args.slope_seed),
        phi2_of_theta=None,
        mode=0,
    )
    torch.save(
        slope_res,
        os.path.join(
            out_dir,
            f"slope_{args.slope:.4f}_bs{args.batch_size}_seed{args.slope_seed}_results.pt",
        ),
    )
    torch.save(
        dict(phi_hist=slope_phi_hist, oracle_hist=slope_oracle_hist),
        os.path.join(
            out_dir,
            f"slope_{args.slope:.4f}_bs{args.batch_size}_seed{args.slope_seed}_meta.pt",
        ),
    )
    slope_df = _scenario_tracking_df(slope_res, np.asarray(slope_oracle_hist, dtype=float))
    slope_df.to_csv(
        os.path.join(
            out_dir,
            f"slope_{args.slope:.4f}_bs{args.batch_size}_seed{args.slope_seed}_theta_tracking.csv",
        ),
        index=False,
    )
    _plot_tracking(
        slope_res,
        np.asarray(slope_oracle_hist, dtype=float),
        None,
        int(args.batch_size),
        f"Slope Theta Tracking (slope={args.slope:.4f}, seed={args.slope_seed})",
        os.path.join(
            out_dir,
            f"slope_{args.slope:.4f}_bs{args.batch_size}_seed{args.slope_seed}_theta_tracking.png",
        ),
    )
    summary_frames.append(_summarize_metrics(slope_res, "slope"))
    config_rows.append(
        {
            "scenario": "slope",
            "seed": int(args.slope_seed),
            "batch_size": int(args.batch_size),
            "slope": float(args.slope),
            "total_T": 600,
        }
    )

    # Mixed
    phi2_of_theta, _ = build_phi2_from_theta_star(
        phi2_grid=np.linspace(3.0, 12.0, 300),
        theta_grid=np.linspace(0.0, 3.0, 600),
    )
    mixed_res, mixed_phi_hist, mixed_oracle_hist, mixed_theta_true_hist, mixed_cp_times = run_one_mixed(
        drift_scale=float(args.mixed_drift_scale),
        jump_scale=float(args.mixed_jump_scale),
        methods=methods,
        batch_size=int(args.batch_size),
        seed=int(args.mixed_seed),
        total_T=int(args.mixed_total_T),
        phi2_of_theta=phi2_of_theta,
        num_particles=int(args.num_particles),
    )
    torch.save(
        mixed_res,
        os.path.join(
            out_dir,
            f"mixed_drift{args.mixed_drift_scale:.4f}_jump{args.mixed_jump_scale:.3f}_bs{args.batch_size}_seed{args.mixed_seed}_results.pt",
        ),
    )
    torch.save(
        dict(
            phi_hist=mixed_phi_hist,
            oracle_hist=mixed_oracle_hist,
            theta_true_hist=mixed_theta_true_hist,
            cp_times=mixed_cp_times,
        ),
        os.path.join(
            out_dir,
            f"mixed_drift{args.mixed_drift_scale:.4f}_jump{args.mixed_jump_scale:.3f}_bs{args.batch_size}_seed{args.mixed_seed}_meta.pt",
        ),
    )
    mixed_df = _scenario_tracking_df(mixed_res, np.asarray(mixed_theta_true_hist, dtype=float))
    mixed_df.to_csv(
        os.path.join(
            out_dir,
            f"mixed_drift{args.mixed_drift_scale:.4f}_jump{args.mixed_jump_scale:.3f}_bs{args.batch_size}_seed{args.mixed_seed}_theta_tracking.csv",
        ),
        index=False,
    )
    _plot_tracking(
        mixed_res,
        np.asarray(mixed_theta_true_hist, dtype=float),
        list(mixed_cp_times),
        int(args.batch_size),
        f"Mixed Theta Tracking (drift={args.mixed_drift_scale:.4f}, jump={args.mixed_jump_scale:.3f}, seed={args.mixed_seed})",
        os.path.join(
            out_dir,
            f"mixed_drift{args.mixed_drift_scale:.4f}_jump{args.mixed_jump_scale:.3f}_bs{args.batch_size}_seed{args.mixed_seed}_theta_tracking.png",
        ),
    )
    summary_frames.append(_summarize_metrics(mixed_res, "mixed"))
    config_rows.append(
        {
            "scenario": "mixed",
            "seed": int(args.mixed_seed),
            "batch_size": int(args.batch_size),
            "drift_scale": float(args.mixed_drift_scale),
            "jump_scale": float(args.mixed_jump_scale),
            "total_T": int(args.mixed_total_T),
        }
    )

    pd.concat(summary_frames, ignore_index=True).to_csv(
        os.path.join(out_dir, "selected_config_method_summary.csv"),
        index=False,
    )
    pd.DataFrame(config_rows).to_csv(
        os.path.join(out_dir, "selected_config_manifest.csv"),
        index=False,
    )


if __name__ == "__main__":
    main()
