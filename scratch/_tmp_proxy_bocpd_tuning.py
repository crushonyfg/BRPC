import csv, os, subprocess, sys
from pathlib import Path
from statistics import mean

repo = Path(r"C:/Users/yxu59/files/autumn2025/park/DynamicCalibration")
out_root = repo / "figs" / "proxy_bocpd_tuning_sudden_20260417_seed5"
out_root.mkdir(parents=True, exist_ok=True)
module = [sys.executable, "-m", "calib.run_synthetic_mechanism_figures"]
mags = [0.5, 1.0, 2.0, 3.0]
seeds = [0, 1, 2, 3, 4]
cp_batches = [6, 12, 18]
all_rows = []
seed_rows = []
for mag in mags:
    run_dir = out_root / f"mag_{str(mag).replace('.', 'p')}"
    cmd = module + [
        "--scenarios", "sudden",
        "--sudden-mag", str(mag),
        "--seeds", *map(str, seeds),
        "--method-set", "proxy_bocpd_tuning",
        "--plot-all-seeds",
        "--out_dir", str(run_dir),
    ]
    print("RUN", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=repo, check=True)

    metric_p = run_dir / "mechanism_metric_summary.csv"
    restart_p = run_dir / "mechanism_restart_count_summary.csv"
    batch_p = run_dir / "mechanism_batch_records.csv"

    metric_rows = {}
    with metric_p.open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            metric_rows[row['method']] = row

    restart_rows = {}
    with restart_p.open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            restart_rows[row['method']] = row

    per_seed = {}
    with batch_p.open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            method = row['method']
            seed = int(row['seed'])
            per_seed.setdefault((method, seed), []).append(row)

    for (method, seed), rows in sorted(per_seed.items()):
        rows = sorted(rows, key=lambda r: int(r['batch_idx']))
        restarts = [int(r['batch_idx']) for r in rows if str(r['did_restart']).lower() == 'true']
        matched = []
        used = set()
        delays = []
        for cp in cp_batches:
            chosen = None
            for idx, r in enumerate(restarts):
                if idx in used:
                    continue
                if cp <= r <= cp + 2:
                    chosen = (idx, r)
                    break
            if chosen is not None:
                used.add(chosen[0])
                matched.append(cp)
                delays.append(chosen[1] - cp)
        false_alarms = len(restarts) - len(used)
        missed = len(cp_batches) - len(matched)
        seed_rows.append({
            'magnitude': mag,
            'method': method,
            'seed': seed,
            'restart_count': len(restarts),
            'matched_cp_hits': len(matched),
            'missed_cp': missed,
            'false_alarms': false_alarms,
            'mean_delay_if_hit': (sum(delays) / len(delays)) if delays else float('nan'),
        })

    methods = sorted({m for m, _ in per_seed.keys()})
    for method in methods:
        seed_sub = [r for r in seed_rows if r['magnitude'] == mag and r['method'] == method]
        mrow = metric_rows.get(method, {})
        rrow = restart_rows.get(method, {})
        all_rows.append({
            'magnitude': mag,
            'method': method,
            'pred_mse': float(mrow.get('pred_mse', 'nan')),
            'pred_noiseless_mse': float(mrow.get('pred_noiseless_mse', 'nan')),
            'theta_mismatch': float(mrow.get('theta_mismatch', 'nan')),
            'restart_mean': float(rrow.get('mean', 'nan')),
            'restart_std': float(rrow.get('std', 'nan')),
            'matched_cp_hits_mean': mean(r['matched_cp_hits'] for r in seed_sub),
            'missed_cp_mean': mean(r['missed_cp'] for r in seed_sub),
            'false_alarms_mean': mean(r['false_alarms'] for r in seed_sub),
            'delay_mean_if_hit': mean(r['mean_delay_if_hit'] for r in seed_sub if r['mean_delay_if_hit'] == r['mean_delay_if_hit']) if any(r['mean_delay_if_hit'] == r['mean_delay_if_hit'] for r in seed_sub) else float('nan'),
        })

summary_p = out_root / "proxy_bocpd_tuning_summary.csv"
with summary_p.open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
    writer.writeheader()
    writer.writerows(all_rows)
seed_p = out_root / "proxy_bocpd_tuning_seed_metrics.csv"
with seed_p.open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=list(seed_rows[0].keys()))
    writer.writeheader()
    writer.writerows(seed_rows)
print('WROTE', summary_p)
print('WROTE', seed_p)
