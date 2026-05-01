import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


LABELS = {
    "B-BRPC-E": "B-BRPC-E",
    "C-BRPC-E": "C-BRPC-E",
    "HalfRefit": "B-BRPC-RRA",
    "B-WaldPF": "BOCPD-WardPFMove",
    "DA": "DA",
    "BC": "BC",
    "B-BRPC-P": "Proxy-BOCPD",
    "C-BRPC-P": "Proxy-wCUSUM",
    "B-BRPC-F": "FixedSupport-BOCPD",
    "C-BRPC-F": "FixedSupport-wCUSUM",
}


def _safe_slug(value: object) -> str:
    return "".join(ch if str(ch).isalnum() or ch in ("-", "_", ".") else "_" for ch in str(value))


def _group_key(row: pd.Series) -> Tuple:
    return (
        row["scenario_family"],
        int(row["seed"]),
        int(row["batch_size"]),
        int(row["total_T"]),
        float(row["magnitude"]) if pd.notna(row["magnitude"]) else None,
        float(row["seg_len"]) if pd.notna(row["seg_len"]) else None,
        float(row["slope"]) if pd.notna(row["slope"]) else None,
        float(row["drift_scale"]) if pd.notna(row["drift_scale"]) else None,
        float(row["jump_scale"]) if pd.notna(row["jump_scale"]) else None,
    )


def _group_name(key: Tuple) -> str:
    scenario, seed, batch_size, total_T, magnitude, seg_len, slope, drift_scale, jump_scale = key
    parts: List[str] = [str(scenario), f"seed{seed}", f"bs{batch_size}"]
    if magnitude is not None:
        parts.append(f"mag{magnitude:.3f}")
    if seg_len is not None:
        parts.append(f"seg{int(seg_len)}")
    if slope is not None:
        parts.append(f"slope{slope:.4f}")
    if drift_scale is not None:
        parts.append(f"drift{drift_scale:.4f}")
    if jump_scale is not None:
        parts.append(f"jump{jump_scale:.3f}")
    if total_T:
        parts.append(f"T{total_T}")
    return "_".join(_safe_slug(p) for p in parts)


def _load_payload(path: Path) -> Dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def _get_truth_series(payload: Dict) -> np.ndarray:
    if "theta_star_true" in payload:
        truth = np.asarray(payload["theta_star_true"], dtype=float).reshape(-1)
        if truth.size > 0 and np.isfinite(truth).any():
            return truth
    if "theta_oracle" in payload:
        truth = np.asarray(payload["theta_oracle"], dtype=float).reshape(-1)
        if truth.size > 0:
            return truth
    return np.asarray([], dtype=float)


def _cp_batches(payload: Dict) -> List[int]:
    if "cp_batches" in payload and payload["cp_batches"] is not None:
        return [int(v) for v in payload["cp_batches"]]
    if "cp_times" in payload and "batch_size" in payload:
        bs = int(payload["batch_size"])
        return [int(v) // bs for v in payload["cp_times"]]
    return []


def _plot_group(group_df: pd.DataFrame, out_dir: Path) -> None:
    key = _group_key(group_df.iloc[0])
    name = _group_name(key)

    series_rows: List[Dict] = []
    loaded: List[Tuple[str, Dict]] = []
    for _, row in group_df.iterrows():
        payload = _load_payload(Path(row["raw_path"]))
        loaded.append((str(row["method"]), payload))

    truth = _get_truth_series(loaded[0][1])
    cp_batches = _cp_batches(loaded[0][1])

    plt.figure(figsize=(12, 5))
    for method, payload in loaded:
        theta = np.asarray(payload.get("theta", []), dtype=float).reshape(-1)
        label = LABELS.get(method, method)
        plt.plot(theta, linewidth=2, label=label)
        for batch_idx, value in enumerate(theta):
            series_rows.append({"batch_idx": int(batch_idx), "series": label, "theta": float(value)})

    if truth.size > 0:
        plt.plot(truth, "k--", linewidth=2.5, label="Ground Truth")
        for batch_idx, value in enumerate(truth):
            series_rows.append({"batch_idx": int(batch_idx), "series": "Ground Truth", "theta": float(value)})

    for cp in cp_batches:
        plt.axvline(cp, color="tab:red", linestyle="--", alpha=0.3)

    plt.xlabel("Batch Index")
    plt.ylabel("Theta")
    plt.title(name)
    plt.legend(loc="best")
    plt.tight_layout()
    png_path = out_dir / f"{name}_theta_tracking.png"
    csv_path = out_dir / f"{name}_theta_tracking.csv"
    plt.savefig(png_path, dpi=300)
    plt.close()

    pd.DataFrame(series_rows).to_csv(csv_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite_dir", type=str, required=True)
    args = parser.parse_args()

    suite_dir = Path(args.suite_dir)
    raw_csv = suite_dir / "raw" / "all_runs.csv"
    if not raw_csv.exists():
        raise FileNotFoundError(f"Missing {raw_csv}")

    df = pd.read_csv(raw_csv)
    if "raw_relpath" not in df.columns:
        raise ValueError("all_runs.csv missing raw_relpath column")
    df["raw_path"] = df["raw_relpath"].apply(lambda rel: str((suite_dir / rel).resolve()))

    plot_dir = suite_dir / "theta_tracking_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    grouped = df.groupby(
        ["scenario_family", "seed", "batch_size", "total_T", "magnitude", "seg_len", "slope", "drift_scale", "jump_scale"],
        dropna=False,
    )
    manifest_rows: List[Dict] = []
    for _, group_df in grouped:
        if len(group_df) < 1:
            continue
        _plot_group(group_df, plot_dir)
        key = _group_key(group_df.iloc[0])
        manifest_rows.append(
            {
                "scenario_family": key[0],
                "seed": key[1],
                "batch_size": key[2],
                "total_T": key[3],
                "magnitude": key[4],
                "seg_len": key[5],
                "slope": key[6],
                "drift_scale": key[7],
                "jump_scale": key[8],
                "plot_prefix": _group_name(key),
                "num_methods": int(len(group_df)),
            }
        )
    pd.DataFrame(manifest_rows).to_csv(plot_dir / "theta_tracking_manifest.csv", index=False)


if __name__ == "__main__":
    main()
