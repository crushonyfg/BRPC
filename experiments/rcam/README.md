# RCAM Experiments

Primary runners:

```bash
conda run -n jumpGP python -m calib.run_rcam_bocpd_pf --help
conda run -n jumpGP python -m calib.run_rcam6d_bocpd_pf --help
conda run -n jumpGP python -m calib.run_rcam6d_hybrid_rolled --help
```

Prefer explicit `--data_csv`, `--out_csv`, and `--plot_dir` arguments because
some historical comments still include local absolute paths.
