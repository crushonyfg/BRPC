import csv, math, subprocess, sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

repo = Path(r"C:/Users/yxu59/files/autumn2025/park/DynamicCalibration")
out_root = repo / "figs" / "wcusum_corrected_large_ablation_20260417_seed10_np1048"
out_root.mkdir(parents=True, exist_ok=True)
base_cmd = [sys.executable, "-m", "calib.run_synthetic_mechanism_figures", "--method-set", "wcusum_ablation", "--num-particles", "1048", "--seeds", *[str(i) for i in range(10)]]

runs = []
for mag in [0.5, 1.0, 2.0, 3.0]:
    tag = f"sudden_mag_{str(mag).replace('.', 'p')}"
    runs.append(dict(scenario="sudden", sweep_name="sudden_mag", sweep_value=mag, out_dir=out_root / tag, extra=["--scenarios", "sudden", "--sudden-mag", str(mag)]))
for slope in [0.0005, 0.001, 0.0015, 0.002, 0.0025]:
    tag = f"slope_{str(slope).replace('.', 'p')}"
    runs.append(dict(scenario="slope", sweep_name="slope", sweep_value=slope, out_dir=out_root / tag, extra=["--scenarios", "slope", "--slope", str(slope)]))
runs.append(dict(scenario="random_walk", sweep_name="random", sweep_value=float('nan'), out_dir=out_root / "random_walk", extra=["--scenarios", "random_walk"]))

combined = []
sudden_align_seed = []
cp_batches = [6, 12, 18]

for spec in runs:
    out_dir = spec['out_dir']
    cmd = base_cmd + spec['extra'] + ["--out_dir", str(out_dir)]
    print("RUN", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=repo, check=True)

    metric_p = out_dir / "mechanism_metric_summary.csv"
    restart_p = out_dir / "mechanism_restart_count_summary.csv"
    metric_df = pd.read_csv(metric_p)
    restart_df = pd.read_csv(restart_p)
    merged = metric_df.merge(restart_df[["method", "mean", "std"]], on="method", how="left", suffixes=("", "_restart"))
    merged["scenario"] = spec['scenario']
    merged["sweep_name"] = spec['sweep_name']
    merged["sweep_value"] = spec['sweep_value']
    merged = merged.rename(columns={"mean": "restart_mean", "std": "restart_std"})
    combined.extend(merged.to_dict(orient="records"))

    if spec['scenario'] == 'sudden':
        batch_df = pd.read_csv(out_dir / "mechanism_batch_records.csv")
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
            sudden_align_seed.append({
                'magnitude': spec['sweep_value'],
                'method': method,
                'seed': int(seed),
                'restart_count': len(restarts),
                'matched_cp_hits': hits,
                'false_alarms': len(restarts) - len(used),
                'delay_mean_if_hit': (sum(delays) / len(delays)) if delays else math.nan,
            })

combined_df = pd.DataFrame(combined)
combined_path = out_root / "combined_metric_summary.csv"
combined_df.to_csv(combined_path, index=False)

label_map = {
    'half_refit': 'HalfRefit',
    'shared_onlineBPC_proxyStableMean_sigmaObs_none': 'Proxy-None',
    'shared_onlineBPC_proxyStableMean_sigmaObs_wCUSUM': 'Proxy-wCUSUM',
    'shared_onlineBPC_exact_sigmaObs_none': 'Exact-None',
    'shared_onlineBPC_exact_sigmaObs_wCUSUM': 'Exact-wCUSUM',
    'shared_onlineBPC_fixedSupport_sigmaObs_none': 'FixedSupport-None',
    'shared_onlineBPC_fixedSupport_sigmaObs_wCUSUM': 'FixedSupport-wCUSUM',
}
combined_df['method_label'] = combined_df['method'].map(label_map).fillna(combined_df['method'])
combined_df['y_rmse'] = combined_df['pred_mse'].clip(lower=0).pow(0.5)
combined_df['theta_rmse'] = combined_df['theta_mismatch'].clip(lower=0).pow(0.5)

summary_rows = []
for scenario, sub in combined_df.groupby('scenario'):
    for method, ms in sub.groupby('method'):
        summary_rows.append({
            'scenario': scenario,
            'method': method,
            'method_label': ms['method_label'].iloc[0],
            'mean_y_rmse': ms['y_rmse'].mean(),
            'mean_theta_rmse': ms['theta_rmse'].mean(),
            'mean_restart_count': ms['restart_mean'].mean(),
        })
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(out_root / "scenario_method_mean_summary.csv", index=False)
summary_df.to_markdown(out_root / "scenario_method_mean_summary.md", index=False)

sudden_align_df = pd.DataFrame(sudden_align_seed)
if not sudden_align_df.empty:
    sudden_align_summary = sudden_align_df.groupby(['magnitude', 'method'], as_index=False).agg(
        restart_mean=('restart_count', 'mean'),
        matched_cp_hits_mean=('matched_cp_hits', 'mean'),
        false_alarms_mean=('false_alarms', 'mean'),
        delay_mean_if_hit=('delay_mean_if_hit', 'mean'),
    )
    sudden_align_summary['method_label'] = sudden_align_summary['method'].map(label_map).fillna(sudden_align_summary['method'])
    sudden_align_summary.to_csv(out_root / "sudden_alignment_summary.csv", index=False)
    sudden_align_df.to_csv(out_root / "sudden_alignment_seed_metrics.csv", index=False)

for scenario, xcol, fname in [("sudden", "sweep_value", "sudden_metric_trends.png"), ("slope", "sweep_value", "slope_metric_trends.png")]:
    sub = combined_df[combined_df['scenario'] == scenario].copy()
    if sub.empty:
        continue
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    for method, ms in sub.groupby('method_label'):
        ms = ms.sort_values(xcol)
        axes[0].plot(ms[xcol], ms['y_rmse'], marker='o', label=method)
        axes[1].plot(ms[xcol], ms['theta_rmse'], marker='o', label=method)
    axes[0].set_title(f"{scenario} Y RMSE")
    axes[1].set_title(f"{scenario} Theta RMSE")
    axes[0].set_xlabel('magnitude' if scenario == 'sudden' else 'slope')
    axes[1].set_xlabel('magnitude' if scenario == 'sudden' else 'slope')
    axes[0].set_ylabel('RMSE')
    axes[1].set_ylabel('RMSE')
    axes[1].legend(loc='best', fontsize=8)
    fig.tight_layout()
    fig.savefig(out_root / fname, dpi=160)
    plt.close(fig)

print("WROTE", combined_path)
print("WROTE", out_root / "scenario_method_mean_summary.csv")
if sudden_align_seed:
    print("WROTE", out_root / "sudden_alignment_summary.csv")

