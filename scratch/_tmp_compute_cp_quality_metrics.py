from __future__ import annotations

from math import isnan
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch


ROOT = Path(r"C:\Users\yxu59\files\autumn2025\park\DynamicCalibration")
MAIN_ROOT = ROOT / "figs" / "synthetic_cpd_suite_np1024_seed25_lambda2_raw_20260421_140855"
HALF_ROOT = ROOT / "figs" / "synthetic_cpd_suite_halfrefit_np1024_seed25_20260424_103503"
OUT_DIR = ROOT / "figs" / "cp_quality_exact_half_20260425"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOL = 2  # allow up to 2 batches of positive delay

METHOD_MAP = {
    "Exact_BOCPD": "B-BRPC-E",
    "Exact_wCUSUM": "C-BRPC-E",
    "HalfRefit": "B-BRPC-RRA",
}


def _load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def _true_cp_batches(obj: dict) -> list[int]:
    cp_batches = obj.get("cp_batches")
    if cp_batches is not None:
        return [int(x) for x in cp_batches]
    cp_times = obj.get("cp_times")
    if cp_times is not None:
        bs = int(obj.get("batch_size", obj.get("config", {}).get("batch_size", 1)))
        return [int(round(float(t) / bs)) for t in cp_times]
    return []


def _pred_restart_batches(obj: dict) -> list[int]:
    hist = obj.get("restart_mode_hist", [])
    return [i for i, mode in enumerate(hist) if str(mode).lower() != "none"]


def _match_events(true_cps: Iterable[int], pred_cps: Iterable[int], tol: int) -> tuple[int, int, int, list[int]]:
    true_cps = list(sorted(int(x) for x in true_cps))
    pred_cps = list(sorted(int(x) for x in pred_cps))
    used = [False] * len(pred_cps)
    tp = 0
    delays: list[int] = []

    for tau in true_cps:
        match_idx = None
        for j, r in enumerate(pred_cps):
            if used[j]:
                continue
            if tau <= r <= tau + tol:
                match_idx = j
                break
        if match_idx is not None:
            used[match_idx] = True
            tp += 1
            delays.append(pred_cps[match_idx] - tau)

    fp = sum(1 for j in range(len(pred_cps)) if not used[j])
    fn = len(true_cps) - tp
    return tp, fp, fn, delays


def _f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def _extract_rows(root: Path, wanted_methods: set[str]) -> list[dict]:
    rows: list[dict] = []
    for scenario in ("sudden", "mixed"):
        raw_dir = root / "raw_runs" / scenario
        for pt_path in sorted(raw_dir.glob("*.pt")):
            obj = _load(pt_path)
            method = str(obj.get("method"))
            if method not in wanted_methods:
                continue

            true_cps = _true_cp_batches(obj)
            pred_cps = _pred_restart_batches(obj)
            tp, fp, fn, delays = _match_events(true_cps, pred_cps, TOL)
            prec, rec, f1 = _f1(tp, fp, fn)

            config = obj.get("config", {})
            row = {
                "source_root": str(root),
                "scenario_family": str(obj.get("scenario_family", scenario)),
                "method": method,
                "legend_method": METHOD_MAP.get(method, method),
                "seed": int(obj.get("seed")),
                "batch_size": int(obj.get("batch_size", config.get("batch_size", -1))),
                "magnitude": config.get("magnitude"),
                "seg_len": config.get("seg_len"),
                "drift_scale": config.get("drift_scale"),
                "jump_scale": config.get("jump_scale"),
                "total_T": config.get("total_T"),
                "true_cp_count": len(true_cps),
                "pred_restart_count": len(pred_cps),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision_at2": prec,
                "recall_at2": rec,
                "f1_at2": f1,
                "mean_delay_at2": (float(np.mean(delays)) if delays else np.nan),
                "matched_count": len(delays),
                "true_cp_batches": ",".join(str(x) for x in true_cps),
                "pred_restart_batches": ",".join(str(x) for x in pred_cps),
                "payload_path": str(pt_path),
            }
            rows.append(row)
    return rows


def _macro_summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    agg = (
        df.groupby(group_cols, as_index=False)
        .agg(
            n_runs=("seed", "count"),
            true_cp_count_mean=("true_cp_count", "mean"),
            pred_restart_count_mean=("pred_restart_count", "mean"),
            tp_mean=("tp", "mean"),
            fp_mean=("fp", "mean"),
            fn_mean=("fn", "mean"),
            precision_at2_mean=("precision_at2", "mean"),
            recall_at2_mean=("recall_at2", "mean"),
            f1_at2_mean=("f1_at2", "mean"),
            mean_delay_at2_mean=("mean_delay_at2", "mean"),
            mean_delay_at2_std=("mean_delay_at2", "std"),
        )
    )
    return agg


def _micro_summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    sums = (
        df.groupby(group_cols, as_index=False)
        .agg(tp=("tp", "sum"), fp=("fp", "sum"), fn=("fn", "sum"), matched_count=("matched_count", "sum"))
    )
    precs = []
    recs = []
    f1s = []
    for _, row in sums.iterrows():
        prec, rec, f1 = _f1(int(row["tp"]), int(row["fp"]), int(row["fn"]))
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)
    sums["precision_at2_micro"] = precs
    sums["recall_at2_micro"] = recs
    sums["f1_at2_micro"] = f1s
    return sums


def main() -> None:
    rows = []
    rows.extend(_extract_rows(HALF_ROOT, {"HalfRefit"}))
    rows.extend(_extract_rows(MAIN_ROOT, {"Exact_BOCPD", "Exact_wCUSUM"}))
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "cp_quality_run_level.csv", index=False)

    overall_macro = _macro_summary(df, ["scenario_family", "method", "legend_method"])
    overall_micro = _micro_summary(df, ["scenario_family", "method", "legend_method"])
    overall = overall_macro.merge(overall_micro, on=["scenario_family", "method", "legend_method"], how="left")
    overall.to_csv(OUT_DIR / "cp_quality_overall_summary.csv", index=False)

    sudden_df = df[df["scenario_family"] == "sudden"].copy()
    sudden_cfg_macro = _macro_summary(sudden_df, ["scenario_family", "method", "legend_method", "magnitude", "seg_len"])
    sudden_cfg_micro = _micro_summary(sudden_df, ["scenario_family", "method", "legend_method", "magnitude", "seg_len"])
    sudden_cfg = sudden_cfg_macro.merge(
        sudden_cfg_micro,
        on=["scenario_family", "method", "legend_method", "magnitude", "seg_len"],
        how="left",
    )
    sudden_cfg.to_csv(OUT_DIR / "cp_quality_sudden_config_summary.csv", index=False)

    mixed_df = df[df["scenario_family"] == "mixed"].copy()
    mixed_cfg_macro = _macro_summary(mixed_df, ["scenario_family", "method", "legend_method", "jump_scale"])
    mixed_cfg_micro = _micro_summary(mixed_df, ["scenario_family", "method", "legend_method", "jump_scale"])
    mixed_cfg = mixed_cfg_macro.merge(
        mixed_cfg_micro,
        on=["scenario_family", "method", "legend_method", "jump_scale"],
        how="left",
    )
    mixed_cfg.to_csv(OUT_DIR / "cp_quality_mixed_config_summary.csv", index=False)

    readme = (
        "Event-based changepoint quality metrics for HalfRefit, Exact_BOCPD, and Exact_wCUSUM.\n"
        f"Matching rule: F1@{TOL} uses one-to-one greedy matching where a predicted restart counts as a TP only if it occurs in [tau, tau+{TOL}] for a true changepoint tau.\n"
        "Early restarts (before tau) do not get credit.\n"
        "mean_delay_at2 is computed over matched detections only.\n"
    )
    (OUT_DIR / "README.txt").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()
