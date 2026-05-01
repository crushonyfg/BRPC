from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch


ROOT = Path(
    r"C:\Users\yxu59\files\autumn2025\park\DynamicCalibration\figs\plantSim_cpd_ablation_seed10_bs20_np1024_20260424_225523"
)
MODES_TO_PLOT = {"1", "2"}


def main() -> None:
    seed_dirs = sorted((ROOT / "seed_runs").glob("seed*"))
    if not seed_dirs:
        raise SystemExit("No seed_runs/seedXX directories found.")
    plot_dir = ROOT / "theta_tracking_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for seed_dir in seed_dirs:
        seed_name = seed_dir.name
        for pt_path in sorted(seed_dir.glob("plantSim_results_mode*.pt")):
            mode = pt_path.stem.replace("plantSim_results_mode", "")
            if mode not in MODES_TO_PLOT:
                continue
            results = torch.load(pt_path, map_location="cpu", weights_only=False)
            if not isinstance(results, dict) or not results:
                continue

            csv_path = plot_dir / f"{seed_name}_mode{mode}_theta_tracking.csv"
            png_path = plot_dir / f"{seed_name}_mode{mode}_theta_tracking.png"

            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["batch_idx", "method", "theta"])
                gt_written = False
                for method, rec in results.items():
                    theta_hist = list(rec["theta_hist"])
                    gt_theta_hist = list(rec["gt_theta_hist"])
                    for idx, theta in enumerate(theta_hist):
                        writer.writerow([idx, method, float(theta)])
                    if not gt_written:
                        for idx, theta in enumerate(gt_theta_hist):
                            writer.writerow([idx, "oracle_theta", float(theta)])
                        gt_written = True

            plt.figure(figsize=(10, 5))
            oracle = None
            for method, rec in results.items():
                plt.plot(rec["theta_hist"], label=method)
                if oracle is None:
                    oracle = rec["gt_theta_hist"]
            if oracle is not None:
                plt.plot(oracle, "k--", lw=2, label="oracle theta")
            plt.title(f"PlantSim theta tracking ({seed_name}, mode={mode}, batch_size=20)")
            plt.xlabel("batch index")
            plt.ylabel("theta")
            plt.legend()
            plt.tight_layout()
            plt.savefig(png_path, dpi=300)
            plt.close()
            print(f"wrote {png_path}")
            print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
