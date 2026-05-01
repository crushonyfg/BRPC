from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from calib.run_synthetic_highdim_projected_diag import (  # noqa: E402
    HighDimDiagSpec,
    run_one_method,
    save_combined_theta_tracking_from_rows,
    write_summaries,
)


def selected_runs(standard_bocpd: bool = False) -> List[Dict[str, object]]:
    if standard_bocpd:
        brpcf_run = dict(
            method_name="FixedSupport_BOCPD",
            run_name="B-BRPC-F_standard_h400",
            controller_overrides=dict(controller_mode="standard", hazard_lambda=400.0),
        )
    else:
        brpcf_run = dict(
            method_name="FixedSupport_BOCPD",
            run_name="B-BRPC-F_h400_m2_c640",
            controller_overrides=dict(hazard_lambda=400.0, restart_margin=2.0, restart_cooldown=640),
        )
    return [
        brpcf_run,
        dict(
            method_name="FixedSupport_wCUSUM",
            run_name="C-BRPC-F_w4_t025_k025_sf025",
            controller_overrides=dict(wcusum_window=4, wcusum_threshold=0.25, wcusum_kappa=0.25, wcusum_sigma_floor=0.25),
        ),
        dict(
            method_name="SlidingWindow-KOH",
            run_name="BC",
            controller_overrides={},
        ),
        dict(
            method_name="DA",
            run_name="DA",
            controller_overrides={},
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="figs/highdim_selected_appendix")
    parser.add_argument("--scenarios", nargs="+", choices=["slope", "sudden", "mixed", "all"], default=["all"])
    parser.add_argument("--seed_count", type=int, default=5)
    parser.add_argument("--seed_offset", type=int, default=0)
    parser.add_argument("--num_particles", type=int, default=1024)
    parser.add_argument("--delta_bpc_lambda", type=float, default=2.0)
    parser.add_argument("--num_support", type=int, default=32)
    parser.add_argument("--total_batches", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_rff", type=int, default=10)
    parser.add_argument("--discrepancy_amp", type=float, default=0.3)
    parser.add_argument("--noise_sd", type=float, default=0.05)
    parser.add_argument("--fixedsupport_standard_bocpd", action="store_true")
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

    all_rows: List[Dict[str, float]] = []
    run_specs = selected_runs(standard_bocpd=bool(args.fixedsupport_standard_bocpd))

    manifest = {
        "scenarios": scenarios,
        "seed_count": int(args.seed_count),
        "seed_offset": int(args.seed_offset),
        "num_particles": int(args.num_particles),
        "delta_bpc_lambda": float(args.delta_bpc_lambda),
        "num_support": int(args.num_support),
        "total_batches": int(args.total_batches),
        "batch_size": int(args.batch_size),
        "fixedsupport_standard_bocpd": bool(args.fixedsupport_standard_bocpd),
        "methods": run_specs,
    }

    for scenario in scenarios:
        for seed in range(int(args.seed_offset), int(args.seed_offset) + int(args.seed_count)):
            for run_cfg in run_specs:
                row = run_one_method(
                    scenario=scenario,
                    seed=seed,
                    out_dir=out_dir,
                    spec=spec,
                    num_particles=int(args.num_particles),
                    delta_bpc_lambda=float(args.delta_bpc_lambda),
                    num_support=int(args.num_support),
                    method_name=str(run_cfg["method_name"]),
                    run_name=str(run_cfg["run_name"]) + f"_bs{int(args.batch_size)}_np{int(args.num_particles)}",
                    controller_overrides=dict(run_cfg["controller_overrides"]),
                )
                all_rows.append(row)

    write_summaries(out_dir, all_rows)
    save_combined_theta_tracking_from_rows(out_dir, all_rows)
    summary_dir = out_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    with (summary_dir / "selected_manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)


if __name__ == "__main__":
    main()
