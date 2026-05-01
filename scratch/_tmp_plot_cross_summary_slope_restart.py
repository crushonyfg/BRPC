from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(r"C:\Users\yxu59\files\autumn2025\park\DynamicCalibration")
HALF_CSV = ROOT / "figs" / "synthetic_cpd_suite_halfrefit_np1024_seed25_20260424_103503" / "summary" / "slope_config_summary.csv"
MAIN_CSV = ROOT / "figs" / "synthetic_cpd_suite_np1024_seed25_lambda2_raw_20260421_140855" / "summary" / "slope_config_summary.csv"
OUT_DIR = ROOT / "figs" / "cross_summary_restart_trend_plots_20260425"
OUT_DIR.mkdir(parents=True, exist_ok=True)

METHOD_MAP = {
    "HalfRefit": "B-BRPC-RRA",
    "Exact_BOCPD": "B-BRPC-E",
    "Exact_wCUSUM": "C-BRPC-E",
}
ORDER = ["B-BRPC-E", "C-BRPC-E", "B-BRPC-RRA"]


def main() -> None:
    df_half = pd.read_csv(HALF_CSV)
    df_main = pd.read_csv(MAIN_CSV)

    keep_half = df_half["method"] == "HalfRefit"
    keep_main = df_main["method"].isin(["Exact_BOCPD", "Exact_wCUSUM"])

    df = pd.concat([df_half[keep_half], df_main[keep_main]], ignore_index=True)
    df = df[df["slope"].notna()].copy()
    df["legend_method"] = df["method"].map(METHOD_MAP)

    plot_df = (
        df[["legend_method", "slope", "restart_count_mean"]]
        .sort_values(["legend_method", "slope"])
        .reset_index(drop=True)
    )
    plot_df.to_csv(OUT_DIR / "restart_vs_slope.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for method in ORDER:
        sub = plot_df[plot_df["legend_method"] == method]
        ax.plot(
            sub["slope"],
            sub["restart_count_mean"],
            marker="o",
            linewidth=2.0,
            label=method,
        )
    ax.set_title("Slope: Mean Restart Count vs Slope")
    ax.set_xlabel("slope")
    ax.set_ylabel("Mean Restart Count")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "restart_vs_slope.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    readme = (
        "Sources:\n"
        f"- HalfRefit: {HALF_CSV}\n"
        f"- Exact BOCPD / wCUSUM: {MAIN_CSV}\n"
        "Legend mapping:\n"
        "- B-BRPC-E = Exact_BOCPD\n"
        "- C-BRPC-E = Exact_wCUSUM\n"
        "- B-BRPC-RRA = HalfRefit\n"
    )
    (OUT_DIR / "README.txt").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()
