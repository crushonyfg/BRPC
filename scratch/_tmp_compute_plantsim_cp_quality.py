from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from calib.v3_utils import JumpPlan, StreamClass


ROOT = Path(r"C:\Users\yxu59\files\autumn2025\park\DynamicCalibration")
RUN_DIR = ROOT / "figs" / "plantSim_cpd_ablation_seed10_bs20_np1024_20260424_225523"
OUT_DIR = RUN_DIR / "cp_quality"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path(r"C:\Users\yxu59\files\winter2026\park\simulation\PhysicalData_v3")
BATCH_SIZE = 20
TOL = 2
GT_EPS = 1e-6


def _load_seed_mode(seed_dir: Path, mode: int) -> dict:
    path = seed_dir / f"plantSim_results_mode{mode}.pt"
    return torch.load(path, map_location="cpu", weights_only=False)


def _pred_restart_batches(restart_hist: Iterable[bool]) -> list[int]:
    return [i for i, v in enumerate(restart_hist) if bool(v)]


def _true_cp_batches_mode1(gt_theta_hist: Iterable[float], eps: float = GT_EPS) -> list[int]:
    gt = np.asarray(list(gt_theta_hist), dtype=float)
    cps = []
    for i in range(1, len(gt)):
        if abs(gt[i] - gt[i - 1]) > eps:
            cps.append(i)
    return cps


def _true_cp_batches_mode2(seed: int, batch_size: int) -> list[int]:
    jp = JumpPlan(
        max_jumps=5,
        min_gap_theta=500.0,
        min_interval=180,
        max_interval=320,
        min_jump_span=40,
        seed=int(seed),
    )
    stream = StreamClass(0, folder=str(DATA_DIR), csv_path=None, jump_plan=jp)

    cp_batches: list[int] = []
    batch_idx = 0
    while True:
        if stream._pos >= stream._n:
            break
        rows_in_batch = 0
        jumped_this_batch = False
        while rows_in_batch < batch_size:
            if stream._pos >= stream._n:
                break
            prev_pos = stream._pos
            prev_jumps = stream._jumps_done
            stream._maybe_jump()
            if stream._jumps_done > prev_jumps or stream._pos != prev_pos:
                jumped_this_batch = True
            t, ref = stream._index[stream._pos]
            stream._pos += 1
            if stream._use_csv:
                sample = stream._read_one_sample_csv(ref)
            else:
                sample = stream._read_one_sample_excel(ref)
            if sample is None:
                continue
            rows_in_batch += 1
            stream._emitted += 1
            stream._last_t = t
        if rows_in_batch == 0:
            break
        if jumped_this_batch:
            cp_batches.append(batch_idx)
        batch_idx += 1
    return cp_batches


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


def _overall_summary(df: pd.DataFrame) -> pd.DataFrame:
    macro = (
        df.groupby(["mode", "method"], as_index=False)
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
    micro = (
        df.groupby(["mode", "method"], as_index=False)
        .agg(tp=("tp", "sum"), fp=("fp", "sum"), fn=("fn", "sum"))
    )
    precs, recs, f1s = [], [], []
    for _, row in micro.iterrows():
        p, r, f = _f1(int(row["tp"]), int(row["fp"]), int(row["fn"]))
        precs.append(p)
        recs.append(r)
        f1s.append(f)
    micro["precision_at2_micro"] = precs
    micro["recall_at2_micro"] = recs
    micro["f1_at2_micro"] = f1s
    return macro.merge(micro, on=["mode", "method"], how="left")


def _plot(df: pd.DataFrame) -> None:
    methods = sorted(df["method"].unique())
    for mode in sorted(df["mode"].unique()):
        sub = df[df["mode"] == mode].copy()
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
        axes[0].bar(sub["method"], sub["f1_at2_mean"], color="#4C78A8")
        axes[0].set_title(f"Mode {mode}: F1@{TOL}")
        axes[0].set_ylabel("F1@2")
        axes[0].tick_params(axis="x", rotation=45)
        axes[0].grid(True, axis="y", alpha=0.3)

        axes[1].bar(sub["method"], sub["mean_delay_at2_mean"], color="#F58518")
        axes[1].set_title(f"Mode {mode}: Mean Delay")
        axes[1].set_ylabel("Mean delay (batches)")
        axes[1].tick_params(axis="x", rotation=45)
        axes[1].grid(True, axis="y", alpha=0.3)

        fig.tight_layout()
        fig.savefig(OUT_DIR / f"mode{mode}_cp_quality.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    seed_root = RUN_DIR / "seed_runs"
    rows: list[dict] = []

    for seed_dir in sorted(seed_root.glob("seed*")):
        seed = int(seed_dir.name.replace("seed", ""))

        # mode 1: derive jumps from stored oracle theta
        mode1_results = _load_seed_mode(seed_dir, 1)
        sample_method = sorted(mode1_results.keys())[0]
        true_cps_mode1 = _true_cp_batches_mode1(mode1_results[sample_method]["gt_theta_hist"])

        # mode 2: reconstruct jump plan exactly
        true_cps_mode2 = _true_cp_batches_mode2(seed=seed, batch_size=BATCH_SIZE)

        for mode, true_cps, results in [
            (1, true_cps_mode1, mode1_results),
            (2, true_cps_mode2, _load_seed_mode(seed_dir, 2)),
        ]:
            for method, rec in results.items():
                pred_cps = _pred_restart_batches(rec["restart_hist"])
                tp, fp, fn, delays = _match_events(true_cps, pred_cps, TOL)
                prec, rec_, f1 = _f1(tp, fp, fn)
                rows.append(
                    {
                        "seed": seed,
                        "mode": mode,
                        "method": method,
                        "true_cp_count": len(true_cps),
                        "pred_restart_count": len(pred_cps),
                        "tp": tp,
                        "fp": fp,
                        "fn": fn,
                        "precision_at2": prec,
                        "recall_at2": rec_,
                        "f1_at2": f1,
                        "mean_delay_at2": (float(np.mean(delays)) if delays else np.nan),
                        "true_cp_batches": ",".join(str(x) for x in true_cps),
                        "pred_restart_batches": ",".join(str(x) for x in pred_cps),
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "plant_cp_quality_run_level.csv", index=False)

    overall = _overall_summary(df)
    overall.to_csv(OUT_DIR / "plant_cp_quality_overall_summary.csv", index=False)
    _plot(overall)

    readme = (
        "Mode 1 true changepoints are extracted from gt_theta_hist as piecewise-constant jumps where |delta gt_theta| > 1e-6.\n"
        "Mode 2 true changepoints are reconstructed by replaying StreamClass(0, jump_plan=JumpPlan(seed=seed)) and marking batches that contain a jump event.\n"
        f"Event matching rule: F1@{TOL} only matches predicted restarts in [tau, tau+{TOL}] to a true changepoint tau; early restarts get no credit.\n"
    )
    (OUT_DIR / "README.txt").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()
