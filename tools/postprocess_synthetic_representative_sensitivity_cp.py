from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
import torch


def group_label(method_name: str) -> str:
    if method_name.startswith("B-BRPC-E_"):
        return "B-BRPC-E"
    if method_name.startswith("C-BRPC-E_"):
        return "C-BRPC-E"
    return method_name


def _finite_mean(vals: Sequence[float]) -> float:
    finite = [float(v) for v in vals if math.isfinite(float(v))]
    if not finite:
        return float("nan")
    return float(np.mean(finite))


def restart_batches_from_payload(payload: Dict[str, Any]) -> List[int]:
    rm_hist = list(payload.get("restart_mode_hist", []))
    if rm_hist:
        return [int(idx) for idx, value in enumerate(rm_hist) if str(value) != "none"]
    others = list(payload.get("others", []))
    if others:
        return [
            int(idx)
            for idx, item in enumerate(others)
            if isinstance(item, dict) and bool(item.get("did_restart", False))
        ]
    return []


def match_events_forward(gt: Sequence[int], det: Sequence[int], tol: int = 2) -> Dict[str, float]:
    gt = [int(v) for v in gt]
    det = [int(v) for v in det]
    used = [False] * len(det)
    tp = 0
    delays: List[int] = []
    for cp in gt:
        found = None
        for idx, dd in enumerate(det):
            if used[idx]:
                continue
            if cp <= dd <= cp + tol:
                found = idx
                break
        if found is not None:
            used[found] = True
            tp += 1
            delays.append(det[found] - cp)
    precision = float(tp / max(len(det), 1))
    recall = float(tp / max(len(gt), 1))
    f1 = float(2.0 * precision * recall / max(precision + recall, 1e-12))
    return {
        "precision_at2": precision,
        "recall_at2": recall,
        "f1_at2": f1,
        "mean_delay": _finite_mean(delays),
    }


def nan_event_stats() -> Dict[str, float]:
    return {
        "precision_at2": float("nan"),
        "recall_at2": float("nan"),
        "f1_at2": float("nan"),
        "mean_delay": float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--tolerance", type=int, default=2)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    raw_dir = results_dir / "raw_runs"
    summary_dir = results_dir / "summary"
    raw_files = sorted(raw_dir.glob("*_results.pt"))
    if not raw_files:
        raise FileNotFoundError(f"No raw result payloads found under {raw_dir}")

    cp_rows: List[Dict[str, Any]] = []
    for raw_path in raw_files:
        payload = torch.load(raw_path, map_location="cpu", weights_only=False)
        scenario = str(payload["scenario"])
        seed = int(payload["seed"])
        cp_batches = [int(v) for v in payload.get("cp_batches", [])]
        methods = dict(payload["methods"])
        raw_relpath = str(raw_path.relative_to(results_dir)).replace("\\", "/")
        for method_name, method_payload in methods.items():
            restart_batches = restart_batches_from_payload(method_payload)
            if scenario in {"sudden", "mixed"}:
                event_stats = match_events_forward(cp_batches, restart_batches, tol=int(args.tolerance))
            else:
                event_stats = nan_event_stats()
            cp_rows.append(
                dict(
                    scenario=scenario,
                    method=str(method_name),
                    family=group_label(str(method_name)),
                    seed=seed,
                    raw_relpath=raw_relpath,
                    tolerance=int(args.tolerance),
                    num_true_cp=int(len(cp_batches)),
                    num_restart=int(len(restart_batches)),
                    precision_at2=float(event_stats["precision_at2"]),
                    recall_at2=float(event_stats["recall_at2"]),
                    f1_at2=float(event_stats["f1_at2"]),
                    mean_delay=float(event_stats["mean_delay"]),
                )
            )

    cp_df = pd.DataFrame(cp_rows)
    cp_df.to_csv(summary_dir / "cp_quality_run_level.csv", index=False)

    merged_df = cp_df.copy()
    run_level_path = summary_dir / "run_level.csv"
    if run_level_path.exists():
        base_df = pd.read_csv(run_level_path)
        merge_cols = ["scenario", "method", "family", "seed", "raw_relpath"]
        merged_df = cp_df.merge(base_df, on=merge_cols, how="left")
        merged_df.to_csv(summary_dir / "run_level_with_cp_quality.csv", index=False)

    summary_df = (
        merged_df.groupby(["scenario", "method", "family"], as_index=False)
        .agg(
            precision_at2_mean=("precision_at2", "mean"),
            precision_at2_std=("precision_at2", "std"),
            recall_at2_mean=("recall_at2", "mean"),
            recall_at2_std=("recall_at2", "std"),
            f1_at2_mean=("f1_at2", "mean"),
            f1_at2_std=("f1_at2", "std"),
            mean_delay_mean=("mean_delay", "mean"),
            mean_delay_std=("mean_delay", "std"),
            restart_count_mean=("restart_count", "mean"),
            theta_rmse_mean=("theta_rmse", "mean"),
            y_rmse_mean=("y_rmse", "mean"),
        )
    )
    summary_df.to_csv(summary_dir / "cp_quality_scenario_summary.csv", index=False)

    range_df = (
        summary_df.groupby(["scenario", "family"], as_index=False)
        .agg(
            theta_rmse_min=("theta_rmse_mean", "min"),
            theta_rmse_max=("theta_rmse_mean", "max"),
            y_rmse_min=("y_rmse_mean", "min"),
            y_rmse_max=("y_rmse_mean", "max"),
            restart_count_min=("restart_count_mean", "min"),
            restart_count_max=("restart_count_mean", "max"),
            precision_at2_min=("precision_at2_mean", "min"),
            precision_at2_max=("precision_at2_mean", "max"),
            recall_at2_min=("recall_at2_mean", "min"),
            recall_at2_max=("recall_at2_mean", "max"),
            f1_at2_min=("f1_at2_mean", "min"),
            f1_at2_max=("f1_at2_mean", "max"),
            mean_delay_min=("mean_delay_mean", "min"),
            mean_delay_max=("mean_delay_mean", "max"),
        )
    )
    range_df.to_csv(summary_dir / "cp_quality_family_range_summary.csv", index=False)


if __name__ == "__main__":
    main()
