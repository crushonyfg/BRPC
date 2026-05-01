from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch

from calib.run_synthetic_highdim_projected_diag import (
    HighDimDiagSpec,
    _save_theta_tracking,
    run_one_method,
    save_combined_theta_tracking_from_rows,
    write_summaries,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_out_dir = out_dir / "raw_runs"
    raw_out_dir.mkdir(parents=True, exist_ok=True)

    source_run_level = pd.read_csv(source_dir / "summary" / "run_level.csv")
    base_rows_df = source_run_level[source_run_level["method"] != "DA"].copy()
    base_rows: List[Dict[str, object]] = []

    for row in base_rows_df.to_dict(orient="records"):
        raw_relpath = str(row["raw_relpath"])
        src_raw = source_dir / raw_relpath
        dst_raw = out_dir / raw_relpath
        dst_raw.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_raw, dst_raw)

        payload = torch.load(dst_raw, map_location="cpu", weights_only=False)
        tag = dst_raw.stem
        _save_theta_tracking(
            out_dir=out_dir,
            tag=tag,
            theta_est=payload["theta"],
            theta_true=payload["theta_star_true"],
            theta_var_diag=payload["theta_var_diag"],
            cp_batches=payload.get("cp_batches", []),
            restart_batches=[idx for idx, mode in enumerate(payload.get("restart_mode_hist", [])) if str(mode) != "none"],
            paper_label=str(payload.get("paper_label", payload.get("method", "method"))),
        )
        base_rows.append(row)

    source_manifest_path = source_dir / "summary" / "selected_manifest.json"
    if source_manifest_path.exists():
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    else:
        raise FileNotFoundError(f"Missing selected_manifest.json under {source_dir / 'summary'}")

    scenarios = list(source_manifest["scenarios"])
    seed_count = int(source_manifest["seed_count"])
    seed_offset = int(source_manifest["seed_offset"])
    num_particles = int(source_manifest["num_particles"])
    delta_bpc_lambda = float(source_manifest["delta_bpc_lambda"])
    num_support = int(source_manifest["num_support"])
    total_batches = int(source_manifest["total_batches"])
    batch_size = int(source_manifest["batch_size"])

    spec = HighDimDiagSpec(
        total_batches=total_batches,
        batch_size=batch_size,
        noise_sd=0.05,
        num_rff=10,
        discrepancy_amp=0.3,
    )

    da_rows: List[Dict[str, object]] = []
    for scenario in scenarios:
        for seed in range(seed_offset, seed_offset + seed_count):
            row = run_one_method(
                scenario=scenario,
                seed=seed,
                out_dir=out_dir,
                spec=spec,
                num_particles=num_particles,
                delta_bpc_lambda=delta_bpc_lambda,
                num_support=num_support,
                method_name="DA",
                run_name=f"DA_bs{batch_size}_np{num_particles}",
                controller_overrides={},
            )
            da_rows.append(row)

    all_rows = base_rows + da_rows
    write_summaries(out_dir, all_rows)
    save_combined_theta_tracking_from_rows(out_dir, all_rows)

    new_manifest = dict(source_manifest)
    new_manifest["strict_da_rebuild"] = True
    new_manifest["strict_da_note"] = "DA rerun with high-dimensional strict WardPF_move generalization; non-DA methods copied from source raw payloads."
    summary_dir = out_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "selected_manifest.json").write_text(json.dumps(new_manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
