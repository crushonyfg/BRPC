from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_1d_joint_enkf_synthetic import run_one


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="figs/joint_enkf_1d_sensitivity")
    parser.add_argument("--scenarios", nargs="+", choices=["slope", "sudden", "mixed", "all"], default=["all"])
    parser.add_argument("--seed_count", type=int, default=3)
    parser.add_argument("--seed_offset", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--noise_sd", type=float, default=0.2)
    parser.add_argument("--n_ensemble", type=int, default=512)
    parser.add_argument("--theta_rw_grid", type=float, nargs="+", default=[0.015, 0.035, 0.070])
    parser.add_argument("--beta_rw_grid", type=float, nargs="+", default=[0.005, 0.015, 0.040])
    parser.add_argument("--inflation_grid", type=float, nargs="+", default=[1.00, 1.02, 1.08])
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scenarios = ["slope", "sudden", "mixed"] if "all" in args.scenarios else list(args.scenarios)
    rows: List[Dict[str, float]] = []
    for scenario in scenarios:
        for theta_rw in args.theta_rw_grid:
            for beta_rw in args.beta_rw_grid:
                for infl in args.inflation_grid:
                    for seed in range(int(args.seed_offset), int(args.seed_offset) + int(args.seed_count)):
                        print(f"[enkf-sens] scenario={scenario} theta_rw={theta_rw} beta_rw={beta_rw} infl={infl} seed={seed}")
                        rows.append(
                            run_one(
                                scenario=scenario,
                                seed=int(seed),
                                out_dir=out_dir,
                                batch_size=int(args.batch_size),
                                noise_sd=float(args.noise_sd),
                                n_ensemble=int(args.n_ensemble),
                                theta_rw_sd=float(theta_rw),
                                beta_rw_sd=float(beta_rw),
                                covariance_inflation=float(infl),
                            )
                        )
                        summary_dir = out_dir / "summary"
                        summary_dir.mkdir(parents=True, exist_ok=True)
                        df = pd.DataFrame(rows)
                        df.to_csv(summary_dir / "run_level.csv", index=False)
                        summ = (
                            df.groupby(["scenario", "theta_rw_sd", "beta_rw_sd", "covariance_inflation"], as_index=False)
                            .agg(
                                theta_rmse_mean=("theta_rmse", "mean"),
                                theta_rmse_std=("theta_rmse", "std"),
                                y_rmse_mean=("y_rmse", "mean"),
                                y_rmse_std=("y_rmse", "std"),
                                y_crps_mean=("y_crps", "mean"),
                                runtime_sec_mean=("runtime_sec", "mean"),
                            )
                        )
                        summ.to_csv(summary_dir / "scenario_summary.csv", index=False)
                        best = (
                            summ.sort_values(["scenario", "theta_rmse_mean", "y_rmse_mean"])
                            .groupby("scenario", as_index=False)
                            .head(5)
                        )
                        best.to_csv(summary_dir / "best_by_scenario.csv", index=False)
    with (out_dir / "summary" / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "method": "JointEnKF sensitivity",
                "theta_rw_grid": [float(v) for v in args.theta_rw_grid],
                "beta_rw_grid": [float(v) for v in args.beta_rw_grid],
                "inflation_grid": [float(v) for v in args.inflation_grid],
                "selection_rule": "Inspect best_by_scenario.csv; primary sort is theta_rmse_mean, tie-break y_rmse_mean.",
            },
            fh,
            indent=2,
        )


if __name__ == "__main__":
    main()
