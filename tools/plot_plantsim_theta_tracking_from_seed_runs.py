from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

METHOD_LABELS = {
    "B-BRPC-RRA": "B-BRPC-RRA",
    "B-BRPC-E": "B-BRPC-E",
    "C-BRPC-E": "C-BRPC-E",
    "DA": "DA",
    "BC": "BC",
}
ORACLE_LABEL = "Ground Truth"


def _to_scalar(x) -> float:
    arr = np.asarray(x, dtype=float).reshape(-1)
    return float(arr[0])


def plot_seed_mode(seed_pt: Path, out_dir: Path, seed_label: str, mode: int) -> None:
    results = torch.load(seed_pt, map_location="cpu", weights_only=False)

    rows = []
    fig, ax = plt.subplots(figsize=(10, 5))

    oracle_plotted = False
    for method, rec in sorted(results.items()):
        theta = list(rec.get("theta_hist", []))
        gt_theta = list(rec.get("gt_theta_hist", []))
        batch_idx = list(range(len(theta)))
        if len(theta) == 0:
            continue

        display_name = METHOD_LABELS.get(method, method)
        ax.plot(batch_idx, theta, label=display_name)
        for i, val in enumerate(theta):
            rows.append(
                {
                    "seed": seed_label,
                    "mode": mode,
                    "batch_idx": i,
                    "series": display_name,
                    "theta": _to_scalar(val),
                }
            )

        if not oracle_plotted and len(gt_theta) > 0:
            ax.plot(range(len(gt_theta)), gt_theta, "k--", lw=2, label=ORACLE_LABEL)
            for i, val in enumerate(gt_theta):
                rows.append(
                    {
                        "seed": seed_label,
                        "mode": mode,
                        "batch_idx": i,
                        "series": ORACLE_LABEL,
                        "theta": _to_scalar(val),
                    }
                )
            oracle_plotted = True

    ax.set_title(f"Theta tracking ({seed_label}, mode={mode})")
    ax.set_xlabel("batch index")
    ax.set_ylabel("theta")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()

    png_path = out_dir / f"{seed_label}_mode{mode}_theta_tracking.png"
    csv_path = out_dir / f"{seed_label}_mode{mode}_theta_tracking.csv"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(rows).to_csv(csv_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot theta tracking from plantSim seed_runs payloads.")
    parser.add_argument("--run-dir", required=True, help="plantSim output directory containing seed_runs/")
    parser.add_argument("--out-subdir", default="theta_tracking_plots", help="name of output subdirectory under run-dir")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    seed_root = run_dir / "seed_runs"
    out_dir = run_dir / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    for seed_dir in sorted(seed_root.glob("seed*")):
        seed_label = seed_dir.name
        for pt_path in sorted(seed_dir.glob("plantSim_results_mode*.pt")):
            stem = pt_path.stem
            mode_str = stem.replace("plantSim_results_mode", "")
            try:
                mode = int(mode_str)
            except ValueError:
                continue
            plot_seed_mode(pt_path, out_dir, seed_label, mode)


if __name__ == "__main__":
    main()
