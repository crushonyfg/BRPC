# AGENTS.md

Synthetic experiments exercise changepoint, slope, mixed-theta, and high-dimensional projected settings.

- Main modules live under `calib.run_synthetic_*`.
- Use `--profile main` for paper-style runs and `--profile ablation` for method-table ablations when supported.
- Keep output directories under `figs/` unless the caller explicitly requests another location.
