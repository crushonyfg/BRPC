from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from calib.run_plantSim_v3_std import summarize_metrics


def _safe_mean(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce")
    vals = vals[vals.notna()]
    return float(vals.mean()) if len(vals) else float("nan")


def summarize_sudden(root: Path, summary_dir: Path) -> None:
    records_metric: list[pd.DataFrame] = []
    records_restart: list[pd.DataFrame] = []
    pattern = re.compile(r"sudden_mag(?P<mag>[0-9p]+)_seg(?P<seg>\d+)")
    for subdir in sorted((root / "sudden").glob("sudden_mag*_seg*")):
        match = pattern.fullmatch(subdir.name)
        if match is None:
            continue
        magnitude = float(match.group("mag").replace("p", "."))
        seg_len = int(match.group("seg"))
        metric_path = subdir / "mechanism_metric_summary.csv"
        restart_path = subdir / "mechanism_restart_count_summary.csv"
        if metric_path.exists():
            dfm = pd.read_csv(metric_path)
            dfm["magnitude"] = magnitude
            dfm["seg_len"] = seg_len
            if "pred_mse" in dfm.columns and "y_rmse" not in dfm.columns:
                dfm["y_rmse"] = dfm["pred_mse"].clip(lower=0.0).map(math.sqrt)
            if "theta_mismatch" in dfm.columns and "theta_rmse" not in dfm.columns:
                dfm["theta_rmse"] = dfm["theta_mismatch"].clip(lower=0.0).map(math.sqrt)
            records_metric.append(dfm)
        if restart_path.exists():
            dfr = pd.read_csv(restart_path)
            dfr["magnitude"] = magnitude
            dfr["seg_len"] = seg_len
            records_restart.append(dfr)

    if records_metric:
        metric_df = pd.concat(records_metric, ignore_index=True)
        metric_df.to_csv(summary_dir / "sudden_metric_summary_combined.csv", index=False)
        group_cols = [c for c in ["scenario", "method"] if c in metric_df.columns]
        agg = (
            metric_df.groupby(group_cols, dropna=False)
            .agg(mean_y_rmse=("y_rmse", "mean"),
                 mean_theta_rmse=("theta_rmse", "mean"),
                 mean_pred_mse=("pred_mse", "mean") if "pred_mse" in metric_df.columns else ("y_rmse", "mean"))
            .reset_index()
        )
        agg.to_csv(summary_dir / "sudden_method_mean_summary.csv", index=False)

    if records_restart:
        restart_df = pd.concat(records_restart, ignore_index=True)
        restart_df.to_csv(summary_dir / "sudden_restart_summary_combined.csv", index=False)
        keep_cols = [c for c in ["scenario", "method"] if c in restart_df.columns]
        value_col = "restart_count" if "restart_count" in restart_df.columns else None
        if value_col is not None:
            agg_restart = (
                restart_df.groupby(keep_cols, dropna=False)
                .agg(mean_restart_count=(value_col, "mean"))
                .reset_index()
            )
            agg_restart.to_csv(summary_dir / "sudden_restart_method_mean_summary.csv", index=False)


def summarize_mixed(root: Path, summary_dir: Path) -> None:
    mixed_dir = root / "mixed_theta"
    metrics_path = mixed_dir / "all_metrics.csv"
    restart_path = mixed_dir / "restart_event_stats.csv"
    if metrics_path.exists():
        df = pd.read_csv(metrics_path)
        df.to_csv(summary_dir / "mixed_all_metrics.csv", index=False)
        agg = (
            df.groupby("method", dropna=False)
            .agg(mean_theta_rmse=("theta_rmse", "mean"),
                 mean_theta_crps=("theta_crps", "mean"),
                 mean_y_rmse=("y_rmse", "mean"),
                 mean_y_crps=("y_crps", "mean"))
            .reset_index()
        )
        agg.to_csv(summary_dir / "mixed_method_mean_summary.csv", index=False)
        by_jump = (
            df.groupby(["method", "jump_scale"], dropna=False)
            .agg(mean_theta_rmse=("theta_rmse", "mean"),
                 mean_y_rmse=("y_rmse", "mean"))
            .reset_index()
        )
        by_jump.to_csv(summary_dir / "mixed_method_by_jump_summary.csv", index=False)
    if restart_path.exists():
        dfr = pd.read_csv(restart_path)
        dfr.to_csv(summary_dir / "mixed_restart_event_stats.csv", index=False)
        agg_r = (
            dfr.groupby("method", dropna=False)
            .agg(mean_full_restart_count=("full_restart_count", "mean"),
                 mean_false_full_restart_count=("false_full_restart_count", "mean"),
                 mean_post_change_correction_delay=("post_change_correction_delay", "mean"))
            .reset_index()
        )
        agg_r.to_csv(summary_dir / "mixed_restart_method_mean_summary.csv", index=False)


def summarize_plant(root: Path, summary_dir: Path) -> None:
    plant_dir = root / "plantSim"
    rows: list[dict[str, Any]] = []
    for pt_path in sorted(plant_dir.glob("plantSim_results_mode*.pt")):
        payload = torch.load(pt_path, map_location="cpu")
        for run_key, results in payload.items():
            mode_match = re.search(r"mode(\d+)", run_key)
            batch_match = re.search(r"bs(\d+)", run_key)
            mode = int(mode_match.group(1)) if mode_match else -1
            batch_size = int(batch_match.group(1)) if batch_match else -1
            for method, result in results.items():
                metrics = summarize_metrics(result)
                restart_hist = result.get("restart_hist", [])
                restart_mean = float(pd.Series(restart_hist, dtype="float64").mean()) if len(restart_hist) else float("nan")
                rows.append({
                    "mode": mode,
                    "batch_size": batch_size,
                    "method": method,
                    "theta_rmse": metrics["theta_rmse"],
                    "theta_crps": metrics["theta_crps"],
                    "y_rmse": metrics["y_rmse"],
                    "y_crps": metrics["y_crps"],
                    "restart_mean": restart_mean,
                    "restart_count": int(sum(bool(v) for v in restart_hist)),
                })
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(summary_dir / "plant_method_mode_summary.csv", index=False)
        agg = (
            df.groupby("method", dropna=False)
            .agg(mean_theta_rmse=("theta_rmse", "mean"),
                 mean_theta_crps=("theta_crps", "mean"),
                 mean_y_rmse=("y_rmse", "mean"),
                 mean_y_crps=("y_crps", "mean"),
                 mean_restart_mean=("restart_mean", "mean"),
                 mean_restart_count=("restart_count", "mean"))
            .reset_index()
        )
        agg.to_csv(summary_dir / "plant_method_mean_summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    summary_dir = root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summarize_sudden(root, summary_dir)
    summarize_mixed(root, summary_dir)
    summarize_plant(root, summary_dir)
    print(f"Summary files written to {summary_dir}")


if __name__ == "__main__":
    main()
