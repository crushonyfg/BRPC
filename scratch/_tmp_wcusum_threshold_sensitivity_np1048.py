import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import calib.run_synthetic_mechanism_figures as mech

repo = Path(r"C:/Users/yxu59/files/autumn2025/park/DynamicCalibration")
out_root = repo / "figs" / "wcusum_threshold_sensitivity_20260417_seed10_np1048"
out_root.mkdir(parents=True, exist_ok=True)

thresholds = [0.15, 0.25, 0.50, 1.00]
magnitudes = [0.5, 1.0, 2.0, 3.0]
seeds = list(range(10))
cp_batches = [6, 12, 18]

orig_default_methods = mech.default_methods

wcusum_methods = [
    "half_refit",
    "shared_onlineBPC_proxyStableMean_sigmaObs_wCUSUM",
    "shared_onlineBPC_exact_sigmaObs_wCUSUM",
    "shared_onlineBPC_fixedSupport_sigmaObs_wCUSUM",
]

label_map = {
    'half_refit': 'HalfRefit',
    'shared_onlineBPC_proxyStableMean_sigmaObs_wCUSUM': 'Proxy-wCUSUM',
    'shared_onlineBPC_exact_sigmaObs_wCUSUM': 'Exact-wCUSUM',
    'shared_onlineBPC_fixedSupport_sigmaObs_wCUSUM': 'FixedSupport-wCUSUM',
}


def build_args() -> object:
    old_argv = sys.argv[:]
    try:
        sys.argv = ["prog"]
        args = mech.parse_args()
    finally:
        sys.argv = old_argv
    args.scenarios = ["sudden"]
    args.seeds = seeds
    args.num_particles = 1048
    args.methods = list(wcusum_methods)
    args.method_set = "core"
    args.plot_all_seeds = False
    args.plot_all_heatmaps = False
    args.plot_all_runlengths = False
    return args


combined_rows = []
align_rows = []

for thr in thresholds:
    methods = orig_default_methods()
    for name in [
        "shared_onlineBPC_proxyStableMean_sigmaObs_wCUSUM",
        "shared_onlineBPC_exact_sigmaObs_wCUSUM",
        "shared_onlineBPC_fixedSupport_sigmaObs_wCUSUM",
    ]:
        methods[name] = dict(methods[name])
        methods[name]["controller_wcusum_threshold"] = float(thr)

    mech.default_methods = lambda methods=methods: methods

    thr_slug = str(thr).replace('.', 'p')
    for mag in magnitudes:
        args = build_args()
        args.sudden_mag = float(mag)
        run_dir = out_root / f"thr_{thr_slug}" / f"mag_{str(mag).replace('.', 'p')}"
        args.out_dir = str(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "mechanism_runner_config.json", "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2)
        print(f"RUN threshold={thr} magnitude={mag} out={run_dir}", flush=True)
        batch_df, evidence_df, runlength_df, raw_results, event_df = mech.run_all(args)

        metric_df = pd.read_csv(run_dir / "mechanism_metric_summary.csv")
        restart_df = pd.read_csv(run_dir / "mechanism_restart_count_summary.csv")
        merged = metric_df.merge(restart_df[["method", "mean", "std"]], on="method", how="left", suffixes=("", "_restart"))
        merged = merged.rename(columns={"mean": "restart_mean", "std": "restart_std"})
        merged["threshold"] = float(thr)
        merged["magnitude"] = float(mag)
        combined_rows.extend(merged.to_dict(orient="records"))

        for (method, seed), sub in batch_df.groupby(["method", "seed"]):
            restarts = sorted(int(v) for v in sub.loc[sub['did_restart'] == True, 'batch_idx'].tolist())
            used = set(); hits = 0; delays = []
            for cp in cp_batches:
                chosen = None
                for idx, rb in enumerate(restarts):
                    if idx in used:
                        continue
                    if cp <= rb <= cp + 2:
                        chosen = (idx, rb)
                        break
                if chosen is not None:
                    used.add(chosen[0]); hits += 1; delays.append(chosen[1] - cp)
            align_rows.append({
                "threshold": float(thr),
                "magnitude": float(mag),
                "method": method,
                "seed": int(seed),
                "restart_count": len(restarts),
                "matched_cp_hits": hits,
                "false_alarms": len(restarts) - len(used),
                "delay_mean_if_hit": (sum(delays) / len(delays)) if delays else math.nan,
            })

mech.default_methods = orig_default_methods

combined_df = pd.DataFrame(combined_rows)
combined_df['method_label'] = combined_df['method'].map(label_map).fillna(combined_df['method'])
combined_df['y_rmse'] = combined_df['pred_mse'].clip(lower=0).pow(0.5)
combined_df['theta_rmse'] = combined_df['theta_mismatch'].clip(lower=0).pow(0.5)
combined_df.to_csv(out_root / 'threshold_combined_metric_summary.csv', index=False)

align_df = pd.DataFrame(align_rows)
align_df['method_label'] = align_df['method'].map(label_map).fillna(align_df['method'])
align_df.to_csv(out_root / 'threshold_alignment_seed_metrics.csv', index=False)

summary = combined_df.groupby(['threshold', 'magnitude', 'method', 'method_label'], as_index=False).agg(
    mean_y_rmse=('y_rmse', 'mean'),
    mean_theta_rmse=('theta_rmse', 'mean'),
    mean_restart_count=('restart_mean', 'mean'),
)
summary.to_csv(out_root / 'threshold_summary.csv', index=False)

align_summary = align_df.groupby(['threshold', 'magnitude', 'method', 'method_label'], as_index=False).agg(
    restart_mean=('restart_count', 'mean'),
    matched_cp_hits_mean=('matched_cp_hits', 'mean'),
    false_alarms_mean=('false_alarms', 'mean'),
    delay_mean_if_hit=('delay_mean_if_hit', 'mean'),
)
align_summary.to_csv(out_root / 'threshold_alignment_summary.csv', index=False)

overall = summary.groupby(['threshold', 'method', 'method_label'], as_index=False).agg(
    mean_y_rmse=('mean_y_rmse', 'mean'),
    mean_theta_rmse=('mean_theta_rmse', 'mean'),
    mean_restart_count=('mean_restart_count', 'mean'),
)
overall = overall.merge(
    align_summary.groupby(['threshold', 'method', 'method_label'], as_index=False).agg(
        matched_cp_hits_mean=('matched_cp_hits_mean', 'mean'),
        false_alarms_mean=('false_alarms_mean', 'mean'),
        delay_mean_if_hit=('delay_mean_if_hit', 'mean'),
    ),
    on=['threshold', 'method', 'method_label'], how='left'
)
overall.to_csv(out_root / 'threshold_overall_summary.csv', index=False)

for method_label in ['Proxy-wCUSUM', 'Exact-wCUSUM', 'FixedSupport-wCUSUM', 'HalfRefit']:
    sub = summary[summary['method_label'] == method_label].copy()
    if sub.empty:
        continue
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    for mag in magnitudes:
        ms = sub[sub['magnitude'] == mag].sort_values('threshold')
        axes[0].plot(ms['threshold'], ms['mean_y_rmse'], marker='o', label=f'mag={mag}')
        axes[1].plot(ms['threshold'], ms['mean_restart_count'], marker='o', label=f'mag={mag}')
    axes[0].set_title(f'{method_label} sudden y-RMSE')
    axes[1].set_title(f'{method_label} sudden restart count')
    axes[0].set_xlabel('wCUSUM threshold')
    axes[1].set_xlabel('wCUSUM threshold')
    axes[0].set_ylabel('mean y-RMSE')
    axes[1].set_ylabel('mean restart count')
    axes[1].legend(loc='best', fontsize=8)
    fig.tight_layout()
    fig.savefig(out_root / f"threshold_sensitivity_{method_label.replace('-', '_')}.png", dpi=180)
    plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
for method_label in ['Proxy-wCUSUM', 'Exact-wCUSUM', 'FixedSupport-wCUSUM', 'HalfRefit']:
    sub = overall[overall['method_label'] == method_label].sort_values('threshold')
    if sub.empty:
        continue
    axes[0].plot(sub['threshold'], sub['mean_y_rmse'], marker='o', label=method_label)
    axes[1].plot(sub['threshold'], sub['mean_restart_count'], marker='o', label=method_label)
    axes[2].plot(sub['threshold'], sub['false_alarms_mean'], marker='o', label=method_label)
axes[0].set_title('Overall sudden y-RMSE')
axes[1].set_title('Overall sudden restart count')
axes[2].set_title('Overall sudden false alarms')
for ax in axes:
    ax.set_xlabel('wCUSUM threshold')
axes[0].set_ylabel('mean')
axes[1].set_ylabel('mean')
axes[2].set_ylabel('mean')
axes[2].legend(loc='best', fontsize=8)
fig.tight_layout()
fig.savefig(out_root / 'threshold_sensitivity_overall.png', dpi=180)
plt.close(fig)

md_lines = [
    '| threshold | method | method_label | mean_y_rmse | mean_theta_rmse | mean_restart_count | matched_cp_hits_mean | false_alarms_mean | delay_mean_if_hit |',
    '|---:|---|---|---:|---:|---:|---:|---:|---:|',
]
for row in overall.sort_values(['method_label', 'threshold']).itertuples(index=False):
    md_lines.append(
        f"| {row.threshold:.2f} | {row.method} | {row.method_label} | {row.mean_y_rmse:.6f} | {row.mean_theta_rmse:.6f} | {row.mean_restart_count:.6f} | {row.matched_cp_hits_mean:.6f} | {row.false_alarms_mean:.6f} | {row.delay_mean_if_hit:.6f} |"
    )
(out_root / 'threshold_overall_summary.md').write_text('\n'.join(md_lines) + '\n', encoding='utf-8')

print('WROTE', out_root / 'threshold_overall_summary.csv')
print('WROTE', out_root / 'threshold_alignment_summary.csv')
print('WROTE', out_root / 'threshold_sensitivity_overall.png')
