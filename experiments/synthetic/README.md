# Synthetic Experiments

Primary runners:

```bash
conda run -n jumpGP python -m calib.run_synthetic_suddenCmp_tryThm --profile main --out_dir figs/sudden_main
conda run -n jumpGP python -m calib.run_synthetic_slope_deltaCmp --profile main --out_dir figs/slope_main
conda run -n jumpGP python -m calib.run_synthetic_mixed_thetaCmp --profile main --out_dir figs/mixed_main
```

Use `--profile ablation` for method-table ablations when the runner supports it.
Use `--preview_only` with `run_synthetic_mixed_thetaCmp` to generate trajectory
previews without running methods.
