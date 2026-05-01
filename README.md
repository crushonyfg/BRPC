# BRPC

Code for the ``Online Bayesian Calibration under Gradual and
Abrupt System Changes`` paper experiments. The repository contains
online Bayesian calibration, particle filtering, BOCPD restart logic, rolled
CUSUM controllers, and the synthetic / PlantSim / RCAM experiments used during
paper development.

## Repository Layout

```text
brpc/         Public paper-facing import namespace and experiment registry.
calib/        Core implementation and legacy experiment runners.
tools/        Orchestration, plotting, aggregation, and post-processing scripts.
experiments/  Short maps for paper-level experiment families.
docs/         Paper PDF and archival design notes.
notebooks/    Exploratory notebooks, not required for reproduction.
scratch/      Temporary diagnostics and historical one-off scripts.
```

## Installation

The expected runtime environment is the `BRPC` conda environment.

```bash
conda env create -f environment.yml
conda activate BRPC
pip install -e .
```

If you already have `BRPC`, install the repo in editable mode:

```bash
conda activate BRPC
pip install -e .
```

Quick syntax check:

```bash
conda run -n BRPC python -m py_compile calib/delta_gp.py calib/online_calibrator.py calib/run_synthetic_suddenCmp_tryThm.py
```

## Minimal API Example

```python
from brpc import CalibrationConfig, DeterministicSimulator, OnlineBayesCalibrator
```

For experiment commands:

```bash
conda run -n BRPC python -m brpc.experiments
```

## Main Experiment Entrypoints

Synthetic abrupt changepoints:

```bash
conda run -n BRPC python -m calib.run_synthetic_suddenCmp_tryThm --profile main --out_dir figs/sudden_main
conda run -n BRPC python -m calib.run_synthetic_suddenCmp_tryThm --profile ablation --out_dir figs/sudden_ablation
```

Synthetic slope / drift:

```bash
conda run -n BRPC python -m calib.run_synthetic_slope_deltaCmp --profile main --out_dir figs/slope_main
conda run -n BRPC python -m calib.run_synthetic_slope_deltaCmp --profile ablation --out_dir figs/slope_ablation
```

Mixed theta trajectories:

```bash
conda run -n BRPC python -m calib.run_synthetic_mixed_thetaCmp --preview_only --out_dir figs/mixed_preview
conda run -n BRPC python -m calib.run_synthetic_mixed_thetaCmp --profile main --out_dir figs/mixed_main
conda run -n BRPC python -m calib.run_synthetic_mixed_thetaCmp --profile ablation --out_dir figs/mixed_ablation
```

PlantSim / factory data:

```bash
conda run -n BRPC python -m calib.run_plantSim_v3_std --help
conda run -n BRPC python -m calib.run_plantSim_v3_std --csv physical_data.csv --out_dir figs/plantSim/v3_std --modes 0 1 2
```

High-dimensional projected diagnostics:

```bash
conda run -n BRPC python -m calib.run_synthetic_highdim_projected_diag --help
```

RCAM diagnostics:

```bash
conda run -n BRPC python -m calib.run_rcam6d_bocpd_pf --help
conda run -n BRPC python -m calib.run_rcam6d_hybrid_rolled --help
```

## Key Method Knobs

Most paper variants are selected by runner method tables rather than by changing
the BOCPD implementation in place. Important names used in current runners
include:

- `Proxy_BOCPD`, `Proxy_wCUSUM`
- `Exact_BOCPD`, `Exact_wCUSUM`
- `FixedSupport_BOCPD`, `FixedSupport_wCUSUM`
- `HalfRefit_BOCPD`
- `WardPFMove_BOCPD`
- `SlidingWindow-KOH`
- `JointEnKF`

Use `--help` on each runner for CLI flags. For detailed discrepancy, restart,
refresh, and predictive-law semantics, see `rolled_cusum_modeling_workdoc.md`.

## Data Files

Legacy data files are currently kept at the repository root because several
older runners use root-relative defaults:

- `physical_data.csv`
- `factory_aggregated.csv`
- `factory_aggregated.npz`
- `lhs_dataset_v2.xlsx`
- `nn_std.bundle.joblib`
- `nnz.bundle.joblib`
- `rcam_stream_windjump.csv`

New scripts should prefer explicit flags such as `--csv`, `--npz`, `--out_dir`,
and `--plot_dir`.

## Development Notes

- Use `conda run -n BRPC ...` for validation.
- Keep old behavior as the default; add new variants through explicit method
  names or config switches.
- When changing rolled-CUSUM, discrepancy, restart, refresh, or runner semantics,
  update `rolled_cusum_modeling_workdoc.md`.
- Generated figures should go under `figs/`, which is ignored by git.
"# BRPC" 
