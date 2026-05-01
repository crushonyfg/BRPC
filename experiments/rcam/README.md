# RCAM Experiments

Primary runners:

```bash
conda run -n BRPC python -m calib.experiment_rcam --help
conda run -n BRPC python -m calib.experiment_rcam6d --help
conda run -n BRPC python -m calib.experiment_rcam6d_hybrid --help
```

Prefer explicit `--data_csv`, `--out_csv`, and `--plot_dir` arguments because
some historical comments still include local absolute paths.
