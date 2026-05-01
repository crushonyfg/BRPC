from __future__ import annotations

import argparse
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from calib.run_synthetic_mixed_thetaCmp import (
    _summarize_mixed_result,
    _summarize_restart_events,
    build_phi2_from_theta_star,
    run_one_mixed,
)


ROOT = Path(__file__).resolve().parent
FIGS = ROOT / "figs"
METHOD = "shared_onlineBPC_proxyStableMean_sigmaObs"


def fmt_lambda(lam: float) -> str:
    return f"{lam:.2f}".rstrip("0").rstrip(".").replace(".", "p")


def run_mechanism(out_dir: Path, scenario: str, lam: float, *, seeds: list[int], sudden_mag: float | None = None, sudden_seg_len: int | None = None, slope: float | None = None) -> None:
    cmd = [
        sys.executable,
        "-m",
        "calib.run_synthetic_mechanism_figures",
        "--out_dir",
        str(out_dir),
        "--scenarios",
        scenario,
        "--seeds",
        *[str(s) for s in seeds],
        "--batch-size",
        "20",
        "--num-particles",
        "1024",
        "--delta-bpc-lambda",
        str(lam),
        "--methods",
        METHOD,
    ]
    if scenario == "sudden":
        cmd += ["--sudden-mag", str(float(sudden_mag)), "--sudden-seg-len", str(int(sudden_seg_len))]
    elif scenario == "slope":
        cmd += ["--slope", str(float(slope))]
    else:
        raise ValueError(f"Unsupported scenario: {scenario}")
    subprocess.run(cmd, cwd=ROOT, check=True)


def summarize_mechanism(out_dir: Path, scenario: str, config_tag: str, lam: float) -> dict[str, float | str]:
    metric_df = pd.read_csv(out_dir / "mechanism_metric_summary.csv")
    restart_df = pd.read_csv(out_dir / "mechanism_restart_count_summary.csv")
    mrow = metric_df[(metric_df["scenario"] == scenario) & (metric_df["method"] == METHOD)].iloc[0]
    rrow = restart_df[(restart_df["scenario"] == scenario) & (restart_df["method"] == METHOD)].iloc[0]
    return {
        "scenario_family": scenario,
        "config": config_tag,
        "lambda": float(lam),
        "method": "Proxy_BOCPD",
        "y_rmse": math.sqrt(max(float(mrow["pred_mse"]), 0.0)),
        "theta_rmse": math.sqrt(max(float(mrow["theta_mismatch"]), 0.0)),
        "restart_mean": float(rrow["mean"]),
        "restart_std": float(rrow["std"]),
        "pred_mse": float(mrow["pred_mse"]),
        "theta_mismatch": float(mrow["theta_mismatch"]),
    }


def run_mixed(config_tag: str, lam: float) -> dict[str, float | str]:
    drift_scale = 0.008
    jump_scale = 0.28 if "jump028" in config_tag else 0.38
    seeds = [101, 202, 303, 404, 505]
    batch_size = 20
    total_T = 600
    import numpy as np

    phi2_grid = np.linspace(3.0, 12.0, 300)
    theta_grid = np.linspace(0.0, 3.0, 600)
    phi2_of_theta, _ = build_phi2_from_theta_star(phi2_grid=phi2_grid, theta_grid=theta_grid)
    methods = {
        "Proxy_BOCPD": dict(
            type="bocpd",
            mode="restart",
            use_discrepancy=False,
            bocpd_use_discrepancy=True,
            delta_update_mode="online_bpc_proxy_stablemean",
            delta_bpc_obs_noise_mode="sigma_eps",
            delta_bpc_predict_add_kernel_noise=False,
            delta_bpc_lambda=float(lam),
        )
    }
    metric_rows = []
    restart_rows = []
    for seed in seeds:
        res, _, _, _, _ = run_one_mixed(
            drift_scale,
            jump_scale,
            methods,
            batch_size,
            seed,
            total_T,
            phi2_of_theta,
            num_particles=1024,
        )
        data = res["Proxy_BOCPD"]
        m = _summarize_mixed_result(data)
        r = _summarize_restart_events(data)
        metric_rows.append(m)
        restart_rows.append(r)
    metric_df = pd.DataFrame(metric_rows)
    restart_df = pd.DataFrame(restart_rows)
    return {
        "scenario_family": "mixed",
        "config": config_tag,
        "lambda": float(lam),
        "method": "Proxy_BOCPD",
        "y_rmse": float(metric_df["y_rmse"].mean()),
        "theta_rmse": float(metric_df["theta_rmse"].mean()),
        "restart_mean": float(restart_df["full_restart_count"].mean()),
        "restart_std": float(restart_df["full_restart_count"].std(ddof=1)) if len(restart_df) > 1 else 0.0,
        "false_full_restart_mean": float(restart_df["false_full_restart_count"].mean()),
        "post_change_delay_mean": float(restart_df["post_change_correction_delay"].dropna().mean()) if restart_df["post_change_correction_delay"].notna().any() else float("nan"),
    }


def make_plot(df: pd.DataFrame, out_path: Path) -> None:
    families = ["sudden", "slope", "mixed"]
    fig, axes = plt.subplots(len(families), 2, figsize=(12, 11), sharex=False)
    for i, fam in enumerate(families):
        sub = df[df["scenario_family"] == fam].copy()
        if sub.empty:
            continue
        for cfg, g in sub.groupby("config", sort=False):
            g = g.sort_values("lambda")
            axes[i, 0].plot(g["lambda"], g["restart_mean"], marker="o", lw=1.6, label=cfg)
            axes[i, 1].plot(g["lambda"], g["y_rmse"], marker="o", lw=1.6, label=cfg)
        axes[i, 0].set_xscale("log", base=2)
        axes[i, 1].set_xscale("log", base=2)
        axes[i, 0].set_title(f"{fam}: restart count")
        axes[i, 1].set_title(f"{fam}: y RMSE")
        axes[i, 0].grid(True, alpha=0.25)
        axes[i, 1].grid(True, alpha=0.25)
        axes[i, 0].set_ylabel("value")
        axes[i, 1].legend(loc="best", fontsize=8)
    axes[-1, 0].set_xlabel("delta_bpc_lambda")
    axes[-1, 1].set_xlabel("delta_bpc_lambda")
    fig.suptitle("Proxy BOCPD lambda transfer sensitivity across scenario families")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-root", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lambdas = [1.0, 2.0, 4.0]
    if args.resume_root:
        run_root = Path(args.resume_root)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_root = FIGS / f"proxy_lambda_transfer_{stamp}_np1024"
    run_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, float | str]] = []

    sudden_cfgs = [
        ("mag1_seg120", dict(sudden_mag=1.0, sudden_seg_len=120)),
        ("mag2_seg120", dict(sudden_mag=2.0, sudden_seg_len=120)),
        ("mag3_seg120", dict(sudden_mag=3.0, sudden_seg_len=120)),
    ]
    slope_cfgs = [
        ("slope0p0010", dict(slope=0.0010)),
        ("slope0p0020", dict(slope=0.0020)),
        ("slope0p0025", dict(slope=0.0025)),
    ]

    for lam in lambdas:
        lam_tag = f"lambda_{fmt_lambda(lam)}"
        for cfg_tag, kwargs in sudden_cfgs:
            out_dir = run_root / "sudden" / lam_tag / cfg_tag
            if not (out_dir / "mechanism_metric_summary.csv").exists():
                run_mechanism(out_dir, "sudden", lam, seeds=[0, 1, 2, 3, 4], **kwargs)
            summary_rows.append(summarize_mechanism(out_dir, "sudden", cfg_tag, lam))
        for cfg_tag, kwargs in slope_cfgs:
            out_dir = run_root / "slope" / lam_tag / cfg_tag
            if not (out_dir / "mechanism_metric_summary.csv").exists():
                run_mechanism(out_dir, "slope", lam, seeds=[0, 1, 2, 3, 4], **kwargs)
            summary_rows.append(summarize_mechanism(out_dir, "slope", cfg_tag, lam))
        for cfg_tag in ["jump028", "jump038"]:
            mixed_dir = run_root / "mixed" / lam_tag
            mixed_dir.mkdir(parents=True, exist_ok=True)
            mixed_path = mixed_dir / f"{cfg_tag}.csv"
            if mixed_path.exists():
                mixed_row = pd.read_csv(mixed_path).iloc[0].to_dict()
            else:
                mixed_row = run_mixed(cfg_tag, lam)
                pd.DataFrame([mixed_row]).to_csv(mixed_path, index=False)
            summary_rows.append(mixed_row)

    df = pd.DataFrame(summary_rows).sort_values(["scenario_family", "config", "lambda"]).reset_index(drop=True)
    df.to_csv(run_root / "proxy_lambda_transfer_summary.csv", index=False)
    make_plot(df, run_root / "proxy_lambda_transfer_summary.png")
    print(run_root)


if __name__ == "__main__":
    main()
