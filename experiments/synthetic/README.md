# Synthetic Experiments

Primary runners:

```bash
conda run -n BRPC python -m calib.experiment_synthetic_sudden --profile main --out_dir figs/sudden_main
conda run -n BRPC python -m calib.experiment_synthetic_slope --profile main --out_dir figs/slope_main
conda run -n BRPC python -m calib.experiment_synthetic_mixed --profile main --out_dir figs/mixed_main
```

Use `--profile ablation` for method-table ablations when the runner supports it.
Use `--preview_only` with `run_synthetic_mixed_thetaCmp` to generate trajectory
previews without running methods.
