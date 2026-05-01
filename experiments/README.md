# Experiment Map

This folder documents the main experiment families. The implementation remains
in `calib` and `tools` so that older commands continue to work.

## Synthetic

```bash
conda run -n BRPC python -m calib.experiment_synthetic_sudden --profile main --out_dir figs/sudden_main
conda run -n BRPC python -m calib.experiment_synthetic_slope --profile main --out_dir figs/slope_main
conda run -n BRPC python -m calib.experiment_synthetic_mixed --profile main --out_dir figs/mixed_main
```

Use `--profile ablation` where supported for method-table ablations.

## PlantSim

```bash
conda run -n BRPC python -m calib.experiment_plantsim --help
```

Typical data files are kept at the repository root for legacy compatibility:
`physical_data.csv`, `factory_aggregated.npz`, `factory_aggregated.csv`, and
the `*.bundle.joblib` emulator bundles.

## RCAM

```bash
conda run -n BRPC python -m calib.experiment_rcam6d --help
conda run -n BRPC python -m calib.experiment_rcam6d_hybrid --help
```
