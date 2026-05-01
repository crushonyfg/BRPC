from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from calib.run_synthetic_highdim_projected_diag import (  # noqa: E402
    HighDimDiagSpec,
    run_one_method,
    write_summaries,
)


def run_grid() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for num_support in (32, 64, 128, 256):
        rows.append(
            dict(
                method_name="B-BRPC-F",
                run_name=f"B-BRPC-F_h400_m2_c20_ns{num_support}",
                num_support=int(num_support),
                controller_overrides=dict(hazard_lambda=400.0, restart_margin=2.0, restart_cooldown=20),
            )
        )
        rows.append(
            dict(
                method_name="B-BRPC-F",
                run_name=f"B-BRPC-F_h400_m2_c640_ns{num_support}",
                num_support=int(num_support),
                controller_overrides=dict(hazard_lambda=400.0, restart_margin=2.0, restart_cooldown=640),
            )
        )
        rows.append(
            dict(
                method_name="C-BRPC-F",
                run_name=f"C-BRPC-F_w4_t025_k025_sf025_ns{num_support}",
                num_support=int(num_support),
                controller_overrides=dict(wcusum_window=4, wcusum_threshold=0.25, wcusum_kappa=0.25, wcusum_sigma_floor=0.25),
            )
        )
    return rows


def save_combined_theta_tracking_by_support(out_dir: Path, rows: Sequence[Dict[str, float]]) -> None:
    if not rows:
        return
    plot_dir = out_dir / "theta_tracking_plots_combined"
    plot_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    manifest_rows: List[Dict[str, object]] = []
    group_cols = ["scenario", "seed", "num_support"]
    for key, grp in df.groupby(group_cols, dropna=False):
        scenario, seed, num_support = key
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
        prefix = f"{scenario}_seed{int(seed)}_ns{int(num_support)}"

        csv_rows: List[Dict[str, object]] = []
        fig, axes = plt.subplots(theta_true.shape[1], 1, figsize=(12, 2.6 * theta_true.shape[1]), sharex=True)
        if theta_true.shape[1] == 1:
            axes = [axes]
        xaxis = np.arange(theta_true.shape[0])
        for j, ax in enumerate(axes):
            ax.plot(xaxis, theta_true[:, j], "k--", lw=2.0, label="Ground Truth")
            for b, val in enumerate(theta_true[:, j]):
                csv_rows.append(dict(batch_idx=int(b), theta_idx=int(j), series="Ground Truth", theta=float(val)))
            for payload in loaded:
                label = str(payload.get("paper_label", payload.get("method", "method")))
                run_name = str(payload.get("run_name", label))
                theta_est = np.asarray(payload["theta"], dtype=float)
                series_label = run_name
                ax.plot(xaxis, theta_est[:, j], lw=1.8, label=series_label)
                for b, val in enumerate(theta_est[:, j]):
                    csv_rows.append(dict(batch_idx=int(b), theta_idx=int(j), series=series_label, theta=float(val)))
            for cpb in cp_batches:
                ax.axvline(int(cpb), color="black", alpha=0.18, lw=1.0)
            ax.set_ylabel(f"theta[{j}]")
        handles, labels = axes[0].get_legend_handles_labels()
        axes[0].legend(handles, labels, loc="best", fontsize=8)
        axes[-1].set_xlabel("batch index")
        fig.tight_layout()
        fig.savefig(plot_dir / f"{prefix}_theta_tracking.png", dpi=220)
        plt.close(fig)
        pd.DataFrame(csv_rows).to_csv(plot_dir / f"{prefix}_theta_tracking.csv", index=False)
        manifest_rows.append(dict(scenario=str(scenario), seed=int(seed), num_support=int(num_support), plot_prefix=prefix, num_methods=int(len(loaded))))
    pd.DataFrame(manifest_rows).to_csv(plot_dir / "theta_tracking_manifest.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="figs/highdim_fixedsupport_support_sweep")
    parser.add_argument("--scenarios", nargs="+", choices=["slope", "sudden", "mixed", "all"], default=["all"])
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--num_particles", type=int, default=1024)
    parser.add_argument("--delta_bpc_lambda", type=float, default=2.0)
    parser.add_argument("--total_batches", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_rff", type=int, default=10)
    parser.add_argument("--discrepancy_amp", type=float, default=0.3)
    parser.add_argument("--noise_sd", type=float, default=0.05)
    args = parser.parse_args()

    scenarios = ["slope", "sudden", "mixed"] if "all" in args.scenarios else list(args.scenarios)
    spec = HighDimDiagSpec(
        total_batches=int(args.total_batches),
        batch_size=int(args.batch_size),
        noise_sd=float(args.noise_sd),
        num_rff=int(args.num_rff),
        discrepancy_amp=float(args.discrepancy_amp),
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, float]] = []
    manifest = {
        "scenarios": scenarios,
        "seed": int(args.seed),
        "num_particles": int(args.num_particles),
        "delta_bpc_lambda": float(args.delta_bpc_lambda),
        "batch_size": int(args.batch_size),
        "total_batches": int(args.total_batches),
        "grid": run_grid(),
    }
    for scenario in scenarios:
        for cfg in run_grid():
            row = run_one_method(
                scenario=scenario,
                seed=int(args.seed),
                out_dir=out_dir,
                spec=spec,
                num_particles=int(args.num_particles),
                delta_bpc_lambda=float(args.delta_bpc_lambda),
                num_support=int(cfg["num_support"]),
                method_name=str(cfg["method_name"]),
                run_name=str(cfg["run_name"]) + f"_bs{int(args.batch_size)}_np{int(args.num_particles)}",
                controller_overrides=dict(cfg["controller_overrides"]),
            )
            rows.append(row)

    write_summaries(out_dir, rows)
    save_combined_theta_tracking_by_support(out_dir, rows)
    summary_dir = out_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    with (summary_dir / "support_sweep_manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)


if __name__ == "__main__":
    main()
