from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from calib.run_synthetic_mixed_thetaCmp import (  # noqa: E402
    _summarize_mixed_result,
    build_phi2_from_theta_star,
    run_one_mixed,
)
from calib.run_synthetic_slope_deltaCmp import _summarize_slope_result, run_one_slope  # noqa: E402
from calib.run_synthetic_suddenCmp_tryThm import _summarize_sudden_result, run_one_sudden  # noqa: E402


def method_specs(num_particles: int) -> Dict[str, Dict[str, Any]]:
    base = dict(
        type="bocpd",
        use_discrepancy=False,
        bocpd_use_discrepancy=True,
        num_particles=int(num_particles),
        delta_bpc_lambda=2.0,
        delta_update_mode="online_bpc_exact",
        delta_bpc_obs_noise_mode="sigma_eps",
        delta_bpc_predict_add_kernel_noise=False,
    )
    specs: Dict[str, Dict[str, Any]] = {}
    for hazard_lambda, restart_margin, restart_cooldown in [
        (200.0, 1.0, 10),
        (400.0, 2.0, 20),
        (800.0, 4.0, 20),
        (1600.0, 4.0, 30),
    ]:
        name = f"B-BRPC-E_h{int(hazard_lambda)}_m{int(restart_margin)}_c{int(restart_cooldown)}"
        specs[name] = dict(
            base,
            mode="restart",
            hazard_lambda=float(hazard_lambda),
            restart_margin=float(restart_margin),
            restart_cooldown=int(restart_cooldown),
        )
    for window, threshold, kappa, sigma_floor in [
        (4, 0.25, 0.25, 0.25),
        (4, 0.50, 0.25, 0.25),
        (8, 0.50, 0.50, 0.50),
        (8, 1.00, 0.50, 0.50),
    ]:
        name = f"C-BRPC-E_w{int(window)}_t{str(threshold).replace('.', '')}_k{str(kappa).replace('.', '')}_sf{str(sigma_floor).replace('.', '')}"
        specs[name] = dict(
            base,
            mode="wcusum",
            controller_name="wcusum",
            controller_stat="log_surprise_mean",
            controller_wcusum_warmup_batches=3,
            controller_wcusum_window=int(window),
            controller_wcusum_threshold=float(threshold),
            controller_wcusum_kappa=float(kappa),
            controller_wcusum_sigma_floor=float(sigma_floor),
        )
    return specs


def group_label(method_name: str) -> str:
    if method_name.startswith("B-BRPC-E_"):
        return "B-BRPC-E"
    if method_name.startswith("C-BRPC-E_"):
        return "C-BRPC-E"
    return method_name


def restart_count_from_payload(payload: Dict[str, Any]) -> float:
    rm_hist = list(payload.get("restart_mode_hist", []))
    if rm_hist:
        return float(sum(1 for v in rm_hist if str(v) != "none"))
    others = list(payload.get("others", []))
    if others:
        return float(sum(1 for item in others if isinstance(item, dict) and bool(item.get("did_restart", False))))
    return 0.0


def _save_theta_tracking(
    out_dir: Path,
    prefix: str,
    results: Dict[str, Dict[str, Any]],
    truth_series: np.ndarray,
    cp_batches: Sequence[int],
) -> None:
    plot_dir = out_dir / "theta_tracking_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    png_path = plot_dir / f"{prefix}_theta_tracking.png"
    csv_path = plot_dir / f"{prefix}_theta_tracking.csv"

    rows: List[Dict[str, Any]] = []
    plt.figure(figsize=(12, 5))
    for method_name, payload in results.items():
        theta = np.asarray(payload["theta"], dtype=float).reshape(-1)
        label = method_name
        plt.plot(theta, linewidth=2, label=label)
        for b, val in enumerate(theta):
            rows.append(dict(batch_idx=int(b), series=label, theta=float(val)))

    plt.plot(truth_series, "k--", linewidth=2.5, label="Ground Truth")
    for b, val in enumerate(truth_series):
        rows.append(dict(batch_idx=int(b), series="Ground Truth", theta=float(val)))
    for cpb in cp_batches:
        plt.axvline(int(cpb), color="tab:red", linestyle="--", alpha=0.3)
    plt.xlabel("Batch Index")
    plt.ylabel("Theta")
    plt.title(prefix)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.close()
    pd.DataFrame(rows).to_csv(csv_path, index=False)


def _plot_summary(df: pd.DataFrame, out_dir: Path) -> None:
    plot_dir = out_dir / "summary_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for scenario in sorted(df["scenario"].unique()):
        sub = df[df["scenario"] == scenario].copy()
        if sub.empty:
            continue
        order = list(sub["method"].drop_duplicates())
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
        metrics = [("y_rmse", "Y RMSE"), ("theta_rmse", "Theta RMSE"), ("restart_count", "Restart Count")]
        x = np.arange(len(order))
        for ax, (metric, title) in zip(axes, metrics):
            vals = [float(sub.loc[sub["method"] == name, metric].mean()) for name in order]
            ax.plot(x, vals, marker="o", linewidth=2)
            ax.set_xticks(x)
            ax.set_xticklabels(order, rotation=35, ha="right")
            ax.set_title(f"{scenario}: {title}")
        fig.tight_layout()
        fig.savefig(plot_dir / f"{scenario}_sensitivity_metrics.png", dpi=260)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="figs/synthetic_representative_sensitivity")
    parser.add_argument("--scenarios", nargs="+", choices=["sudden", "slope", "mixed", "all"], default=["all"])
    parser.add_argument("--seed_count", type=int, default=5)
    parser.add_argument("--seed_offset", type=int, default=0)
    parser.add_argument("--num_particles", type=int, default=1024)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw_runs"
    raw_dir.mkdir(parents=True, exist_ok=True)

    phi2_of_theta, _ = build_phi2_from_theta_star(phi2_grid=np.linspace(3.0, 12.0, 300), theta_grid=np.linspace(0.0, 3.0, 600))
    methods = method_specs(num_particles=int(args.num_particles))
    all_rows: List[Dict[str, Any]] = []

    configs = [
        dict(
            scenario="sudden",
            tag_base="sudden_mag2.0_seg120_bs20",
            run=lambda seed: run_one_sudden(seg_len_L=120, delta_mag=2.0, methods=methods, batch_size=20, seed=seed),
            summarizer=_summarize_sudden_result,
            get_truth=lambda oracle_hist, extra: np.asarray(oracle_hist, dtype=float).reshape(-1),
            get_cp_batches=lambda extra: [120 // 20, 240 // 20, 360 // 20],
        ),
        dict(
            scenario="slope",
            tag_base="slope_0.0015_bs20",
            run=lambda seed: run_one_slope(slope=0.0015, methods=methods, total_T=600, batch_size=20, seed=seed, phi2_of_theta=phi2_of_theta, mode=1),
            summarizer=_summarize_slope_result,
            get_truth=lambda oracle_hist, extra: np.asarray(oracle_hist, dtype=float).reshape(-1),
            get_cp_batches=lambda extra: [],
        ),
        dict(
            scenario="mixed",
            tag_base="mixed_drift0.008_jump0.38_bs20",
            run=lambda seed: run_one_mixed(drift_scale=0.008, jump_scale=0.38, methods=methods, batch_size=20, seed=seed, total_T=600, phi2_of_theta=phi2_of_theta, num_particles=int(args.num_particles)),
            summarizer=_summarize_mixed_result,
            get_truth=lambda oracle_hist, extra: np.asarray(extra["theta_star_true"], dtype=float).reshape(-1),
            get_cp_batches=lambda extra: [int(t // 20) for t in extra["cp_times"]],
        ),
    ]
    scenario_filter = {"sudden", "slope", "mixed"} if "all" in args.scenarios else set(args.scenarios)
    configs = [cfg for cfg in configs if cfg["scenario"] in scenario_filter]

    manifest_rows: List[Dict[str, Any]] = []
    for cfg in configs:
        for seed in range(int(args.seed_offset), int(args.seed_offset) + int(args.seed_count)):
            if cfg["scenario"] == "mixed":
                res, phi_hist, oracle_hist, theta_true_hist, cp_times = cfg["run"](seed)
                extra = dict(theta_star_true=theta_true_hist, cp_times=cp_times)
            else:
                res, phi_hist, oracle_hist = cfg["run"](seed)
                extra = {}
            truth_series = cfg["get_truth"](oracle_hist, extra)
            cp_batches = cfg["get_cp_batches"](extra)
            prefix = f"{cfg['tag_base']}_seed{seed}"
            _save_theta_tracking(out_dir, prefix, res, truth_series, cp_batches)
            raw_path = raw_dir / f"{prefix}_results.pt"
            torch.save(
                dict(
                    scenario=cfg["scenario"],
                    seed=int(seed),
                    methods=res,
                    truth_series=np.asarray(truth_series, dtype=float),
                    cp_batches=list(cp_batches),
                    phi_hist=phi_hist,
                    oracle_hist=np.asarray(oracle_hist, dtype=float),
                    extra=extra,
                ),
                raw_path,
            )
            manifest_rows.append(dict(scenario=cfg["scenario"], seed=int(seed), raw_relpath=str(raw_path.relative_to(out_dir)).replace("\\", "/"), plot_prefix=prefix, num_methods=int(len(res))))
            for method_name, payload in res.items():
                metrics = cfg["summarizer"](payload)
                all_rows.append(
                    dict(
                        scenario=cfg["scenario"],
                        method=method_name,
                        family=group_label(method_name),
                        seed=int(seed),
                        theta_rmse=float(metrics["theta_rmse"]),
                        theta_crps=float(metrics["theta_crps"]) if math.isfinite(metrics["theta_crps"]) else float("nan"),
                        y_rmse=float(metrics["y_rmse"]),
                        y_crps=float(metrics["y_crps"]) if math.isfinite(metrics["y_crps"]) else float("nan"),
                        restart_count=restart_count_from_payload(payload),
                        runtime_sec=float(payload.get("elapsed_sec", float("nan"))),
                        raw_relpath=str(raw_path.relative_to(out_dir)).replace("\\", "/"),
                    )
                )

    df = pd.DataFrame(all_rows)
    summary_dir = out_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(summary_dir / "run_level.csv", index=False)
    agg = (
        df.groupby(["scenario", "method", "family"], as_index=False)
        .agg(
            theta_rmse_mean=("theta_rmse", "mean"),
            theta_rmse_std=("theta_rmse", "std"),
            y_rmse_mean=("y_rmse", "mean"),
            y_rmse_std=("y_rmse", "std"),
            restart_count_mean=("restart_count", "mean"),
            runtime_sec_mean=("runtime_sec", "mean"),
        )
    )
    agg.to_csv(summary_dir / "scenario_summary.csv", index=False)
    pd.DataFrame(manifest_rows).to_csv(summary_dir / "theta_tracking_manifest.csv", index=False)
    with (summary_dir / "selected_manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(
            dict(
                seed_count=int(args.seed_count),
                seed_offset=int(args.seed_offset),
                num_particles=int(args.num_particles),
                representative_configs=[cfg["tag_base"] for cfg in configs],
                methods=list(methods.keys()),
            ),
            fh,
            indent=2,
        )
    _plot_summary(df, out_dir)


if __name__ == "__main__":
    main()
