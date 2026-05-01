# AGENTS.md

`calib/` contains the legacy implementation package and the main experiment runners.

- Treat this folder as behavior-critical model code.
- Preserve existing runner module names unless a new paper-facing wrapper is added elsewhere.
- Prefer opt-in method names/config switches for new algorithm variants.
- When changing model behavior, update `../rolled_cusum_modeling_workdoc.md` and run `conda run -n jumpGP python -m py_compile` on touched files.
- Common entrypoints are `run_synthetic_suddenCmp_tryThm.py`, `run_synthetic_slope_deltaCmp.py`, `run_synthetic_mixed_thetaCmp.py`, `run_synthetic_cpd_suite.py`, `run_plantSim_v3_std.py`, and `run_synthetic_highdim_projected_diag.py`.
