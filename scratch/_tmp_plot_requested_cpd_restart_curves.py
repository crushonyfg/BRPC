from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(r"C:\Users\yxu59\files\autumn2025\park\DynamicCalibration")
SUMMARY_DIR = ROOT / "figs" / "requested_cpd_large_20260418_223504_np1024" / "summary"
OUT_DIR = SUMMARY_DIR / "restart_trend_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

METHOD_MAP = {
    "shared_onlineBPC_exact_sigmaObs": "B-BRPC-E",
    "shared_onlineBPC_exact_sigmaObs_wCUSUM": "C-BRPC-E",
    "half_refit": "B-BRPC-RRA",
}
METHOD_ORDER = ["B-BRPC-E", "C-BRPC-E", "B-BRPC-RRA"]


def _save_plot_csv(df: pd.DataFrame, name: str) -> None:
    df.to_csv(OUT_DIR / f"{name}.csv", index=False)


def _plot_line(df: pd.DataFrame, x_col: str, title: str, out_name: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for method in METHOD_ORDER:
        sub = df[df["legend_method"] == method].sort_values(x_col)
        ax.plot(
            sub[x_col],
            sub["mean_restart_count"],
            marker="o",
            linewidth=2.0,
            label=method,
        )
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel("Mean Restart Count")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{out_name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    sudden_restart = pd.read_csv(SUMMARY_DIR / "sudden_restart_summary_combined.csv")
    sudden_restart = sudden_restart[sudden_restart["method"].isin(METHOD_MAP)].copy()
    sudden_restart["legend_method"] = sudden_restart["method"].map(METHOD_MAP)
    sudden_restart = sudden_restart.rename(columns={"mean": "mean_restart_count"})

    mag_df = (
        sudden_restart.groupby(["legend_method", "magnitude"], as_index=False)["mean_restart_count"]
        .mean()
        .sort_values(["legend_method", "magnitude"])
    )
    _save_plot_csv(mag_df, "restart_vs_magnitude")
    _plot_line(
        mag_df,
        x_col="magnitude",
        title="Sudden: Mean Restart Count vs Magnitude",
        out_name="restart_vs_magnitude",
    )

    seg_df = (
        sudden_restart.groupby(["legend_method", "seg_len"], as_index=False)["mean_restart_count"]
        .mean()
        .sort_values(["legend_method", "seg_len"])
    )
    _save_plot_csv(seg_df, "restart_vs_seglen")
    _plot_line(
        seg_df,
        x_col="seg_len",
        title="Sudden: Mean Restart Count vs Segment Length",
        out_name="restart_vs_seglen",
    )

    note = (
        "This summary directory contains sudden and mixed-theta outputs only.\n"
        "No slope sweep CSV is present here, so no slope restart curve was generated.\n"
    )
    (OUT_DIR / "README.txt").write_text(note, encoding="utf-8")


if __name__ == "__main__":
    main()
