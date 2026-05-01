from pathlib import Path
p = Path(r"C:/Users/yxu59/files/autumn2025/park/DynamicCalibration/rolled_cusum_modeling_workdoc.md")
text = p.read_text(encoding='utf-8')
start = text.index('### Runner methods', text.index('### Motivation'))
new_tail = '''### Runner methods

The corrected wCUSUM comparison now uses the shared online-BPC families that we
actually want to compare against `half_refit`:

- `shared_onlineBPC_proxyStableMean_sigmaObs_none`
- `shared_onlineBPC_proxyStableMean_sigmaObs_wCUSUM`
- `shared_onlineBPC_exact_sigmaObs_none`
- `shared_onlineBPC_exact_sigmaObs_wCUSUM`
- `shared_onlineBPC_fixedSupport_sigmaObs_none`
- `shared_onlineBPC_fixedSupport_sigmaObs_wCUSUM`

The method-set is still:

- `wcusum_ablation`

This method-set also includes `half_refit` as the BOCPD baseline.

### Example direct command

```bash
conda run -n jumpGP python -m calib.run_synthetic_mechanism_figures \
  --scenarios sudden \
  --sudden-mag 2.0 \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --method-set wcusum_ablation \
  --num-particles 1048 \
  --out_dir figs/wcusum_corrected_large_ablation_20260417_seed10_np1048/sudden_mag_2p0
```

### Example large-sweep command

For the corrected 1048-particle sweep across sudden, slope, and random-walk, we
used the temporary orchestration script:

```bash
conda run -n jumpGP python _tmp_wcusum_corrected_large_ablation_np1048.py
```

This writes per-setting runner outputs under:

- `figs/wcusum_corrected_large_ablation_20260417_seed10_np1048/`

and the aggregated postprocessed tables/plots under:

- `figs/wcusum_corrected_large_ablation_20260417_seed10_np1048/aggregate_post/`

### Corrected 1048-particle large-ablation results

Key aggregate outputs:

- `figs/wcusum_corrected_large_ablation_20260417_seed10_np1048/aggregate_post/combined_metric_summary.csv`
- `figs/wcusum_corrected_large_ablation_20260417_seed10_np1048/aggregate_post/scenario_method_mean_summary.csv`
- `figs/wcusum_corrected_large_ablation_20260417_seed10_np1048/aggregate_post/sudden_alignment_summary.csv`
- `figs/wcusum_corrected_large_ablation_20260417_seed10_np1048/aggregate_post/sudden_metric_trends.png`
- `figs/wcusum_corrected_large_ablation_20260417_seed10_np1048/aggregate_post/slope_metric_trends.png`

Scenario-averaged results:

- `random_walk`
  - `Proxy-None`: `mean_y_rmse = 0.5147`, `mean_restart_count = 0.0`
  - `Proxy-wCUSUM`: `mean_y_rmse = 0.5167`, `mean_restart_count = 1.6`
  - `HalfRefit`: `mean_y_rmse = 0.5178`, `mean_restart_count = 4.6`
  - `Exact-wCUSUM`: `0.5918`, `2.3`
  - `FixedSupport-wCUSUM`: `0.5848`, `2.3`
- `slope`
  - `Proxy-None`: `mean_y_rmse = 0.4106`, `mean_restart_count = 0.0`
  - `Proxy-wCUSUM`: `mean_y_rmse = 0.4355`, `mean_restart_count = 0.48`
  - `HalfRefit`: `mean_y_rmse = 0.4402`, `mean_restart_count = 6.48`
  - `Exact-wCUSUM`: `0.6116`, `3.6`
  - `FixedSupport-wCUSUM`: `0.5972`, `3.62`
- `sudden`
  - `HalfRefit`: `mean_y_rmse = 0.8540`, `mean_restart_count = 3.10`
  - `Proxy-wCUSUM`: `mean_y_rmse = 0.8563`, `mean_restart_count = 2.875`
  - `FixedSupport-wCUSUM`: `0.9137`, `3.175`
  - `Exact-wCUSUM`: `0.9469`, `3.15`
  - all three `*_none` variants stay at `0.0` restarts and are much worse on sudden.

Sudden-change alignment across magnitudes `[0.5, 1.0, 2.0, 3.0]`:

- `Proxy-wCUSUM`
  - restart mean ranges from `1.7` to `3.7`
  - false alarms stay between `0.6` and `1.3`
  - matched hits range from `1.0` to `2.4`
  - mean delay conditional on hit is `0.0` batches in this tolerance metric
- `Exact-wCUSUM`
  - restart mean ranges from `2.8` to `3.6`
  - false alarms stay between `0.7` and `1.3`
  - matched hits range from `1.7` to `2.4`
  - mean delay conditional on hit is `0.0`
- `FixedSupport-wCUSUM`
  - restart mean ranges from `2.8` to `3.5`
  - false alarms stay between `0.7` and `1.2`
  - matched hits range from `1.8` to `2.4`
  - mean delay conditional on hit is `0.0`
- `HalfRefit`
  - remains the cleanest reference detector in this synthetic setup: restart mean `3.1`, false alarms `0.1`, matched hits `3.0` across all four magnitudes.

### Interpretation

The corrected 1048-particle sweep changes the picture in two important ways.

1. The earlier "inducing" comparison should not be used here; the relevant third
   family is `fixedSupport`, i.e. the support-projected exact online-BPC path.
2. For abrupt changes, `wCUSUM` now gives a usable controller for `proxy`,
   `exact`, and `fixedSupport`, with restart counts close to the true `3`
   changepoints and much cleaner alignment than the previous over-aggressive
   BOCPD behavior.

Observed trade-off:

- On `sudden`, `Proxy-wCUSUM` is the closest to `half_refit` while keeping
  restart counts near `3`.
- On `slope` and `random_walk`, `Proxy-None` remains the strongest shared
  online-BPC variant on prediction, which is consistent with the idea that these
  regimes often do not benefit from aggressive restarting.
- `Exact-wCUSUM` and `FixedSupport-wCUSUM` are viable changepoint controllers,
  but they still pay a predictive penalty relative to `proxy` on the smoother
  regimes.
'''
text = text[:start] + new_tail
p.write_text(text, encoding='utf-8')
print('patched workdoc')
