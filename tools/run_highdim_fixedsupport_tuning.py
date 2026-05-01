from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from calib.run_synthetic_highdim_projected_diag import (
    HighDimDiagSpec,
    run_one_method,
    write_summaries,
)


def bocpd_configs() -> List[Dict[str, object]]:
    return [
        dict(
            method_name="FixedSupport_BOCPD",
            run_name="B-BRPC-F_h200_m1_c10",
            controller_overrides=dict(hazard_lambda=200.0, restart_margin=1.0, restart_cooldown=10),
        ),
        dict(
            method_name="FixedSupport_BOCPD",
            run_name="B-BRPC-F_h400_m2_c20",
            controller_overrides=dict(hazard_lambda=400.0, restart_margin=2.0, restart_cooldown=20),
        ),
        dict(
            method_name="FixedSupport_BOCPD",
            run_name="B-BRPC-F_h800_m4_c20",
            controller_overrides=dict(hazard_lambda=800.0, restart_margin=4.0, restart_cooldown=20),
        ),
        dict(
            method_name="FixedSupport_BOCPD",
            run_name="B-BRPC-F_h1600_m4_c30",
            controller_overrides=dict(hazard_lambda=1600.0, restart_margin=4.0, restart_cooldown=30),
        ),
    ]


def wcusum_configs() -> List[Dict[str, object]]:
    return [
        dict(
            method_name="Exact_wCUSUM",  # placeholder method; overwritten below
            run_name="B-BRPC-F_w4_t025_k025_sf025",
            controller_overrides=dict(wcusum_window=4, wcusum_threshold=0.25, wcusum_kappa=0.25, wcusum_sigma_floor=0.25),
        ),
        dict(
            method_name="Exact_wCUSUM",
            run_name="B-BRPC-F_w4_t050_k025_sf025",
            controller_overrides=dict(wcusum_window=4, wcusum_threshold=0.50, wcusum_kappa=0.25, wcusum_sigma_floor=0.25),
        ),
        dict(
            method_name="Exact_wCUSUM",
            run_name="B-BRPC-F_w8_t050_k050_sf050",
            controller_overrides=dict(wcusum_window=8, wcusum_threshold=0.50, wcusum_kappa=0.50, wcusum_sigma_floor=0.50),
        ),
        dict(
            method_name="Exact_wCUSUM",
            run_name="B-BRPC-F_w8_t100_k050_sf050",
            controller_overrides=dict(wcusum_window=8, wcusum_threshold=1.00, wcusum_kappa=0.50, wcusum_sigma_floor=0.50),
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="figs/highdim_fixedsupport_tuning")
    parser.add_argument("--scenarios", nargs="+", choices=["slope", "sudden", "mixed", "all"], default=["all"])
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--delta_bpc_lambda", type=float, default=2.0)
    parser.add_argument("--num_support", type=int, default=32)
    parser.add_argument("--total_batches", type=int, default=60)
    parser.add_argument("--num_rff", type=int, default=10)
    parser.add_argument("--discrepancy_amp", type=float, default=0.3)
    parser.add_argument("--noise_sd", type=float, default=0.05)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scenarios = ["slope", "sudden", "mixed"] if "all" in args.scenarios else list(args.scenarios)
    batch_sizes = [64, 128]
    particle_counts = [1024, 2048]

    rows = []
    configs = bocpd_configs() + [
        dict(method_name="FixedSupport_wCUSUM", run_name=cfg["run_name"], controller_overrides=cfg["controller_overrides"])
        for cfg in wcusum_configs()
    ]

    for scenario in scenarios:
        for batch_size in batch_sizes:
            spec = HighDimDiagSpec(
                total_batches=int(args.total_batches),
                batch_size=int(batch_size),
                noise_sd=float(args.noise_sd),
                num_rff=int(args.num_rff),
                discrepancy_amp=float(args.discrepancy_amp),
            )
            for num_particles in particle_counts:
                for cfg in configs:
                    print(
                        f"[highdim-tune] scenario={scenario} seed={args.seed} bs={batch_size} "
                        f"np={num_particles} run={cfg['run_name']}"
                    )
                    rows.append(
                        run_one_method(
                            scenario=scenario,
                            seed=int(args.seed),
                            out_dir=out_dir,
                            spec=spec,
                            num_particles=int(num_particles),
                            delta_bpc_lambda=float(args.delta_bpc_lambda),
                            num_support=int(args.num_support),
                            method_name=str(cfg["method_name"]),
                            run_name=f"{cfg['run_name']}_bs{batch_size}_np{num_particles}",
                            controller_overrides=dict(cfg["controller_overrides"]),
                        )
                    )
                    write_summaries(out_dir, rows)

    summary_dir = out_dir / "summary"
    run_df = pd.read_csv(summary_dir / "run_level.csv")
    run_df.to_csv(summary_dir / "run_level.csv", index=False)
    with (summary_dir / "tuning_manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(
            dict(
                seed=int(args.seed),
                scenarios=scenarios,
                batch_sizes=batch_sizes,
                particle_counts=particle_counts,
                delta_bpc_lambda=float(args.delta_bpc_lambda),
                num_support=int(args.num_support),
                notes="Targeted controller sweep for high-dimensional fixed-support BOCPD and wCUSUM on a single representative seed.",
            ),
            fh,
            indent=2,
        )


if __name__ == "__main__":
    main()
