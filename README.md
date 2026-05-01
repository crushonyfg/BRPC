# BRPC

Code for the **Online Bayesian Calibration under Gradual and Abrupt System
Changes** experiments. The repository contains Bayesian Recursive Particle
Calibration (BRPC), particle filtering, BOCPD restart logic, rolled-CUSUM
controllers, and synthetic / PlantSim / RCAM experiment runners.

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

The implementation package is still `calib` for backward compatibility. New
user-facing imports can use `brpc`.

## Installation

The expected runtime environment for this repository is `BRPC`.

```bash
conda env create -f environment.yml
conda activate BRPC
pip install -e .
```

If `BRPC` already exists:

```bash
conda activate BRPC
pip install -e .
```

Quick check:

```bash
conda run -n BRPC python -m py_compile calib/delta_gp.py calib/online_calibrator.py
conda run -n BRPC python -m brpc.experiments
```

## Paper Method Names

The public runner names and plot legends use the paper notation:

| Method | Internal choice |
| --- | --- |
| `BRPC-P` | proxy online-BPC discrepancy memory without BOCPD/wCUSUM changepoint detection |
| `B-BRPC-P` | BOCPD controller with proxy online-BPC discrepancy memory |
| `C-BRPC-P` | wCUSUM controller with proxy online-BPC discrepancy memory |
| `BRPC-E` | exact online-BPC discrepancy memory without BOCPD/wCUSUM changepoint detection |
| `B-BRPC-E` | BOCPD controller with exact online-BPC discrepancy memory |
| `C-BRPC-E` | wCUSUM controller with exact online-BPC discrepancy memory |
| `BRPC-F` | fixed-support exact online-BPC memory without BOCPD/wCUSUM changepoint detection |
| `B-BRPC-F` | BOCPD controller with fixed-support exact online-BPC memory |
| `C-BRPC-F` | wCUSUM controller with fixed-support exact online-BPC memory |
| `B-BRPC-RRA` | BOCPD restart/refit baseline |
| `B-WaldPF` | BOCPD over Ward-style PF memory |
| `BC` | sliding-window Bayesian calibration baseline |
| `EnKF` | joint ensemble Kalman filter baseline |

Legacy names such as `Exact_BOCPD` are accepted as aliases in the high-dimensional
and CPD-suite runners, but new results should use the paper names above.

## Main Experiment Entrypoints

List the registered commands:

```bash
conda run -n BRPC python -m brpc.experiments
```

Public entrypoint modules use normalized names under `calib.experiment_*`.

Synthetic abrupt changepoints:

```bash
conda run -n BRPC python -m calib.experiment_synthetic_sudden --profile main --out_dir figs/sudden_main
conda run -n BRPC python -m calib.experiment_synthetic_sudden --profile ablation --out_dir figs/sudden_ablation
```

Synthetic slope / drift:

```bash
conda run -n BRPC python -m calib.experiment_synthetic_slope --profile main --out_dir figs/slope_main
conda run -n BRPC python -m calib.experiment_synthetic_slope --profile ablation --out_dir figs/slope_ablation
```

Mixed theta trajectories:

```bash
conda run -n BRPC python -m calib.experiment_synthetic_mixed --preview_only --out_dir figs/mixed_preview
conda run -n BRPC python -m calib.experiment_synthetic_mixed --profile main --out_dir figs/mixed_main
conda run -n BRPC python -m calib.experiment_synthetic_mixed --profile cpd_ablation --out_dir figs/mixed_cpd_ablation
```

Configurable synthetic CPD suite with explicit methods:

```bash
conda run -n BRPC python -m calib.experiment_synthetic_cpd_suite --scenario all --seed-count 1 --num-particles 256 --methods BRPC-P BRPC-E BRPC-F B-BRPC-E C-BRPC-E B-BRPC-F C-BRPC-F BC B-WaldPF --out_dir figs/cpd_suite_smoke
```

High-dimensional projected diagnostics:

```bash
conda run -n BRPC python -m calib.experiment_highdim_projected --scenarios sudden --seed-count 1 --total-batches 20 --batch-size 16 --num-particles 256 --methods BRPC-P BRPC-E BRPC-F B-BRPC-F C-BRPC-F BC --out_dir figs/highdim_smoke
```

PlantSim / factory data:

```bash
conda run -n BRPC python -m calib.experiment_plantsim --help
conda run -n BRPC python -m calib.experiment_plantsim --csv physical_data.csv --out_dir figs/plantSim/v3_std --modes 0 1 2 --num_particles 1024 --profile cpd_ablation_plus
```

Use `--profile cpd_ablation_plus` when you want the no-changepoint `BRPC-P`,
`BRPC-E`, and `BRPC-F` baselines together with the BOCPD and wCUSUM variants.

## Programmatic Calibrator API

Minimal scalar-output example:

```python
import torch

from brpc import CalibrationConfig, DeterministicSimulator, OnlineBayesCalibrator
from calib.configs import BOCPDConfig, ModelConfig, PFConfig

dtype = torch.float64

def simulator(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    # x: [batch, x_dim], theta: [1, theta_dim]
    return torch.sin(5.0 * theta[:, :1] * x[:, :1]).reshape(-1)

theta_low = torch.tensor([0.0], dtype=dtype)
theta_high = torch.tensor([3.0], dtype=dtype)

def prior_sampler(n: int) -> torch.Tensor:
    u = torch.rand(n, theta_low.numel(), dtype=dtype)
    return theta_low + (theta_high - theta_low) * u

cfg = CalibrationConfig(
    model=ModelConfig(
        rho=1.0,
        sigma_eps=0.05,
        device="cpu",
        dtype=dtype,
        use_discrepancy=False,
        delta_update_mode="online_bpc_exact",
        delta_bpc_obs_noise_mode="sigma_eps",
        delta_bpc_predict_add_kernel_noise=False,
    ),
    pf=PFConfig(
        num_particles=1024,
        resample_ess_ratio=0.5,
        move_strategy="random_walk",
        random_walk_scale=0.1,
    ),
    bocpd=BOCPDConfig(
        bocpd_mode="restart",
        hazard_lambda=200,
        max_experts=5,
    ),
)

METHOD_CONFIGS = {
    "BRPC-P": dict(bocpd_mode="single_segment", controller_name="none", delta_update_mode="online_bpc_proxy_stablemean"),
    "B-BRPC-P": dict(bocpd_mode="restart", controller_name="none", delta_update_mode="online_bpc_proxy_stablemean"),
    "C-BRPC-P": dict(bocpd_mode="wcusum", controller_name="wcusum", delta_update_mode="online_bpc_proxy_stablemean"),
    "BRPC-E": dict(bocpd_mode="single_segment", controller_name="none", delta_update_mode="online_bpc_exact"),
    "B-BRPC-E": dict(bocpd_mode="restart", controller_name="none", delta_update_mode="online_bpc_exact"),
    "C-BRPC-E": dict(bocpd_mode="wcusum", controller_name="wcusum", delta_update_mode="online_bpc_exact"),
    "BRPC-F": dict(bocpd_mode="single_segment", controller_name="none", delta_update_mode="online_bpc_fixedsupport_exact"),
    "B-BRPC-F": dict(bocpd_mode="restart", controller_name="none", delta_update_mode="online_bpc_fixedsupport_exact"),
    "C-BRPC-F": dict(bocpd_mode="wcusum", controller_name="wcusum", delta_update_mode="online_bpc_fixedsupport_exact"),
}

choice = METHOD_CONFIGS["BRPC-E"]
cfg.model.delta_update_mode = choice["delta_update_mode"]
cfg.bocpd.bocpd_mode = choice["bocpd_mode"]
cfg.bocpd.controller_name = choice["controller_name"]

cal = OnlineBayesCalibrator(
    calib_cfg=cfg,
    emulator=DeterministicSimulator(simulator),
    prior_sampler=prior_sampler,
)

X_new = torch.rand(20, 1, dtype=dtype)      # [batch, x_dim]
y_new = torch.rand(20, dtype=dtype)         # scalar y: [batch]

pred_before = cal.predict_batch(X_new)      # {"mu", "var", "mu_sim", ...}
update_info = cal.step_batch(X_new, y_new)  # online update
pred_after = cal.predict_batch(X_new)
```

Use `step(x_t, y_t)` for one observation and `step_batch(X_batch, Y_batch)` for a
batch. `predict_batch(X_batch)` returns mixture predictive `mu` and `var`; for
multi-output systems these have shape `[batch, y_dim]`.

## New Data Format

For programmatic use, each online update consumes tensors:

- `X_batch`: shape `[batch_size, x_dim]`, dtype matching `cfg.model.dtype`
- `Y_batch`: scalar output `[batch_size]` or multi-output `[batch_size, y_dim]`
- `theta` particles: shape `[num_particles, theta_dim]`, sampled by `prior_sampler`

The emulator must implement:

```python
mu_eta, var_eta = emulator.predict(X_batch, theta_particles)
```

Expected shapes:

- scalar output: `mu_eta`, `var_eta` are `[batch_size, num_particles]`
- multi-output: `mu_eta`, `var_eta` are `[batch_size, num_particles, y_dim]`

`DeterministicSimulator` wraps a Python function `f(x, theta)` and returns zero
emulator variance. Use `GPEmulator` when the simulator itself is represented by
a GP over joint inputs `[x, theta]`.

## Key Configuration Parameters

Particle filter:

- `PFConfig.num_particles`: number of theta particles, usually `256` for smoke
  tests and `1024` or more for paper runs.
- `PFConfig.move_strategy`: one of `random_walk`, `liu_west`, `laplace`,
  `pmcmc`, or `none` depending on runner support.
- `PFConfig.random_walk_scale`: random-walk proposal scale in theta space.
- `PFConfig.resample_ess_ratio`: resample when ESS falls below this fraction of
  `num_particles`.

Theta dimension and range:

- The theta dimension is defined by `prior_sampler(n)`, which must return
  `[n, theta_dim]`.
- Bounds are not stored globally; encode them in `prior_sampler` and choose move
  scales compatible with those bounds.
- For box constraints, sample `theta_low + (theta_high - theta_low) * rand`.

Input and output dimensions:

- `x_dim` is `X_batch.shape[1]`.
- `theta_dim` is `prior_sampler(n).shape[1]`.
- `y_dim` is inferred from emulator output and `Y_batch`.

BOCPD / controller:

- No changepoint detection: `BOCPDConfig(bocpd_mode="single_segment", controller_name="none")`
- BOCPD mode: `BOCPDConfig(bocpd_mode="restart")`
- wCUSUM mode: `BOCPDConfig(bocpd_mode="wcusum", controller_name="wcusum")`
- Main hazard scale: `BOCPDConfig.hazard_lambda`
- Number of retained experts: `BOCPDConfig.max_experts`

Discrepancy memory:

- `BRPC-P` family: `ModelConfig(delta_update_mode="online_bpc_proxy_stablemean")`
- `BRPC-E` family: `ModelConfig(delta_update_mode="online_bpc_exact")`
- `BRPC-F` family: `ModelConfig(delta_update_mode="online_bpc_fixedsupport_exact")`
- Restart/refit baseline: `ModelConfig(delta_update_mode="refit")`

## Data Files

Legacy data files are kept at the repository root because several older runners
use root-relative defaults:

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
