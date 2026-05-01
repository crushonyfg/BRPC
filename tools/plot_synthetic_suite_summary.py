from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd


MAIN_METHODS = [
    "HalfRefit",
    "BRPC-P",
    "B-BRPC-P",
    "C-BRPC-P",
    "B-BRPC-E",
    "C-BRPC-E",
    "BRPC-F",
    "B-BRPC-F",
    "C-BRPC-F",
    "ParticleFixedSupport_None",
    "ParticleB-BRPC-F",
    "ParticleC-BRPC-F",
]


FAMILY_COLORS: Dict[str, str] = {
    "HalfRefit": "black",
    "Proxy": "#1f77b4",
    "Exact": "#d62728",
    "FixedSupport": "#2ca02c",
    "ParticleFixedSupport": "#9467bd",
    "PF-OGP": "#ff7f0e",
    "BC": "#8c564b",
    "BOCPD-PF-OGP": "#7f7f7f",
}


CPD_STYLES: Dict[str, str] = {
    "None": "--",
    "BOCPD": "-",
    "wCUSUM": "-.",
}


METRIC_COLUMNS = [
    ("y_rmse_mean", "Y RMSE"),
    ("theta_rmse_mean", "Theta RMSE"),
    ("restart_count_mean", "Restart Count"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default=None)
    return parser.parse_args()


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _method_style(method: str, family: str, cpd: str) -> Dict[str, str]:
    color = FAMILY_COLORS.get(family, "#444444")
    linestyle = CPD_STYLES.get(cpd, "-")
    if method == "HalfRefit":
        color = "black"
        linestyle = "-"
    return {"color": color, "linestyle": linestyle}


def _restrict_methods(df: pd.DataFrame) -> pd.DataFrame:
    keep = [m for m in MAIN_METHODS if m in set(df["method"].astype(str))]
    return df[df["method"].isin(keep)].copy()


def _plot_trends(df: pd.DataFrame, x_col: str, title: str, x_label: str, png_path: Path) -> None:
    methods_present = [m for m in MAIN_METHODS if m in set(df["method"].astype(str))]
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 12.0), sharex=True)
    for ax, (metric_col, metric_label) in zip(axes, METRIC_COLUMNS):
        for method in methods_present:
            sub = df[df["method"] == method].sort_values(x_col)
            if sub.empty:
                continue
            family = str(sub["family"].iloc[0]) if "family" in sub.columns else method.split("_")[0]
            cpd = str(sub["cpd"].iloc[0]) if "cpd" in sub.columns else ""
            style = _method_style(method, family, cpd)
            ax.plot(
                sub[x_col],
                sub[metric_col],
                label=method,
                linewidth=2.0,
                markersize=5.0,
                marker="o",
                color=style["color"],
                linestyle=style["linestyle"],
                alpha=0.95,
            )
        ax.set_ylabel(metric_label)
        ax.grid(alpha=0.25, linestyle=":")
    axes[-1].set_xlabel(x_label)
    fig.suptitle(title)
    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        frameon=False,
        ncol=3,
        fontsize=9,
    )
    fig.tight_layout(rect=[0.0, 0.08, 1.0, 0.97])
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=220, bbox_inches="tight", bbox_extra_artists=(legend,))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    summary_dir = Path(args.summary_dir)
    out_dir = Path(args.out_dir) if args.out_dir else summary_dir / "trend_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    sudden = _load_csv(summary_dir / "sudden_config_summary.csv")
    slope = _load_csv(summary_dir / "slope_config_summary.csv")
    mixed = _load_csv(summary_dir / "mixed_config_summary.csv")

    sudden = sudden.dropna(subset=["magnitude", "seg_len"]).copy()
    slope = slope.dropna(subset=["slope"]).copy()
    mixed = mixed.dropna(subset=["jump_scale"]).copy()

    sudden_mag = (
        sudden.groupby(["method", "family", "cpd", "magnitude"], as_index=False)[[m for m, _ in METRIC_COLUMNS]]
        .mean(numeric_only=True)
    )
    sudden_mag = _restrict_methods(sudden_mag)
    sudden_mag.to_csv(out_dir / "sudden_trends_by_magnitude.csv", index=False)
    _plot_trends(
        sudden_mag,
        x_col="magnitude",
        title="Sudden: metrics vs magnitude (averaged over seg_len)",
        x_label="Magnitude",
        png_path=out_dir / "sudden_trends_by_magnitude.png",
    )

    sudden_seg = (
        sudden.groupby(["method", "family", "cpd", "seg_len"], as_index=False)[[m for m, _ in METRIC_COLUMNS]]
        .mean(numeric_only=True)
    )
    sudden_seg = _restrict_methods(sudden_seg)
    sudden_seg.to_csv(out_dir / "sudden_trends_by_seglen.csv", index=False)
    _plot_trends(
        sudden_seg,
        x_col="seg_len",
        title="Sudden: metrics vs segment length (averaged over magnitude)",
        x_label="Segment Length",
        png_path=out_dir / "sudden_trends_by_seglen.png",
    )

    slope = _restrict_methods(slope)
    slope.to_csv(out_dir / "slope_trends.csv", index=False)
    _plot_trends(
        slope,
        x_col="slope",
        title="Slope: metrics vs slope",
        x_label="Slope",
        png_path=out_dir / "slope_trends.png",
    )

    mixed_jump = (
        mixed.groupby(["method", "family", "cpd", "jump_scale"], as_index=False)[[m for m, _ in METRIC_COLUMNS]]
        .mean(numeric_only=True)
    )
    mixed_jump = _restrict_methods(mixed_jump)
    mixed_jump.to_csv(out_dir / "mixed_trends_by_jump_scale.csv", index=False)
    _plot_trends(
        mixed_jump,
        x_col="jump_scale",
        title="Mixed: metrics vs jump scale (drift_scale fixed at 0.008)",
        x_label="Jump Scale",
        png_path=out_dir / "mixed_trends_by_jump_scale.png",
    )


if __name__ == "__main__":
    main()
