# Experiment Map

This folder documents the main experiment families. The implementation remains
in `calib` and `tools` so that older commands continue to work.

## Synthetic

```bash
conda run -n jumpGP python -m calib.run_synthetic_suddenCmp_tryThm --profile main --out_dir figs/sudden_main
conda run -n jumpGP python -m calib.run_synthetic_slope_deltaCmp --profile main --out_dir figs/slope_main
conda run -n jumpGP python -m calib.run_synthetic_mixed_thetaCmp --profile main --out_dir figs/mixed_main
```

Use `--profile ablation` where supported for method-table ablations.

## PlantSim

```bash
conda run -n jumpGP python -m calib.run_plantSim_v3_std --help
```

Typical data files are kept at the repository root for legacy compatibility:
`physical_data.csv`, `factory_aggregated.npz`, `factory_aggregated.csv`, and
the `*.bundle.joblib` emulator bundles.

## RCAM

```bash
conda run -n jumpGP python -m calib.run_rcam6d_bocpd_pf --help
conda run -n jumpGP python -m calib.run_rcam6d_hybrid_rolled --help
```
