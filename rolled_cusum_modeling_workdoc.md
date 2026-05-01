# Rolled CUSUM Modeling Workdoc

## 1. Scope

This document is the maintained reference for the `rolled_cusum_260324` line.
It explains:

- what mathematical model the current code is implementing,
- which switches exist,
- how the discrepancy layer is parameterized,
- which options affect PF, BOCPD, restart, or discrepancy refresh,
- and which runner method names map to which modeling choice.

This document should be updated whenever the rolled-CUSUM path gains a new option or changes its semantics.

## 2. Layered view of the model

The code is easiest to understand as four layers.

### 2.1 Observation model

For scalar output, the working model is

$$
y_t(x) = \rho \, \eta(x, \theta_t) + \delta_t(x) + \varepsilon_t,
\qquad
\varepsilon_t \sim \mathcal N(0, \sigma_\varepsilon^2).
$$

Here:

- $\eta(x,\theta)$ is the emulator / simulator term,
- $\theta_t$ is the calibration latent state tracked by PF,
- $\delta_t(x)$ is the discrepancy term,
- $\rho$ is the simulator scale factor,
- $\sigma_\varepsilon^2$ is observation noise.

### 2.2 PF layer

The PF layer tracks $\theta_t$.
The intended design remains discrepancy-free PF weighting:

$$
w_t^{(i)} \propto w_{t-1}^{(i)}
\; p\!\left(Y_t \mid X_t, \theta_t^{(i)}, \text{simulator only}\right).
$$

Interpretation:

- PF is the latent calibration tracker.
- Discrepancy is not supposed to dominate particle identifiability.
- BOCPD-side discrepancy enrichment happens after or around PF, not inside PF weights.

### 2.3 BOCPD / expert layer

Each BOCPD expert maintains:

- a particle cloud for $\theta$,
- expert history,
- and a discrepancy state used for predictive scoring and/or refresh.

The BOCPD predictive object can be written as

$$
q_{e,t}(Y_t \mid X_t)
=
\int p(Y_t \mid X_t, \theta_t, \phi_t)
\, p(\phi_t \mid \theta_t, \mathcal D_{e,t-1})
\, p(\theta_t \mid \mathcal D_{e,t-1})
\, d\phi_t \, d\theta_t,
$$

where $\phi_t$ denotes the discrepancy-side latent object.

### 2.4 Refresh / restart layer

The code currently separates two decisions:

- BOCPD restart: structural decision about experts / restart behavior.
- Rolled-CUSUM refresh: discrepancy-memory maintenance decision.

This is important:

- the standardized gate or cumulative statistic is not meant to replace BOCPD,
- it is used to decide when discrepancy memory should be refreshed,
- and it should not change PF weights by itself.

## 3. Core discrepancy parameterizations

The current rolled-CUSUM path supports several discrepancy models through `particle_delta_mode`.

### 3.1 `shared_gp`

This is the original expert-shared discrepancy design.
For expert $e$, one residual target is formed using a PF-weighted simulator mean:

$$
r_e(x) = y(x) - \rho \sum_i w_i \, \eta(x, \theta_i).
$$

Then a single GP is fit to that expert-level residual history:

$$
\delta_e(\cdot) \sim \mathcal{GP}(0, k_\psi(\cdot,\cdot)).
$$

This gives one shared posterior for the whole expert.

### 3.2 `particle_gp_shared_hyper`

This is particle-specific discrepancy with one shared GP hyperparameter setting.

For each particle $\theta_i$, define particle-specific residuals

$$
r_e^{(i)}(x) = y(x) - \rho \, \eta(x, \theta_i).
$$

Conditioned on each particle, discrepancy is a GP posterior using the same kernel hyperparameters $\psi$:

$$
\delta_e^{(i)}(\cdot) \mid \theta_i, \mathcal D_e
\sim
\mathcal{GP\ posterior}(r_e^{(i)}, \psi).
$$

Implementation idea:

- fit one shared GP on the expert-shared residual to get a stable hyperparameter setting,
- reuse that hyperparameter set for all particle-specific residual posteriors,
- reuse kernel-factorization work across particles for efficiency.

### 3.3 `particle_gp_hyper_pool`

This keeps particle-specific residuals, but replaces a single shared hyperparameter setting by a small candidate pool.

For a small set of hyperparameters $\{\psi_h\}_{h=1}^H$,

$$
p(\delta \mid \theta_i, \mathcal D_e)
\approx
\sum_{h=1}^H \omega_{ih}
\, p(\delta \mid \theta_i, \mathcal D_e, \psi_h).
$$

Implementation intent:

- hyperparameter candidates are shared across experts / particles at configuration time,
- kernel matrices are reusable for each candidate,
- only the particle-specific residual vectors differ across particles.

This is meant to approximate a lightweight hyper-mixture without exploding cost.

### 3.4 `particle_basis`

This is the particle-specific basis-form discrepancy.

For each particle,

$$
\delta_e^{(i)}(x) = \phi(x)^\top \beta_e^{(i)}.
$$

A Bayesian linear / ridge-style posterior is fit using the particle-specific residuals:

$$
r_e^{(i)} = \Phi_e \beta_e^{(i)} + \xi,
\qquad
\xi \sim \mathcal N(0, \sigma_\delta^2 I).
$$

Current basis options:

- `particle_basis_kind="linear"`
- `particle_basis_kind="rbf"`

This branch is useful for fast ablations and for checking whether a lower-rank discrepancy parameterization behaves differently from GP discrepancy.

## 4. Prediction semantics

For a fixed expert and particle, the predictive law is approximately

$$
Y \mid X, \theta_i, \mathcal D_e
\sim
\mathcal N
\left(
\rho \, \mu_\eta(X,\theta_i) + \mu_{\delta,i}(X),
\rho^2 \sigma_\eta^2(X,\theta_i) + \sigma_{\delta,i}^2(X) + \sigma_\varepsilon^2
\right).
$$

Then the particle mixture is taken using the PF weights.

Important code-level note:

- shared discrepancy returns `mu_delta, var_delta` with batch shape,
- particle-specific discrepancy returns `mu_delta, var_delta` with batch-by-particle shape,
- the prediction path must preserve that distinction.

## 5. Restart and refresh options

### 5.1 `use_dual_restart`

Controls whether the BOCPD hybrid restart logic allows dual / partial restart behavior.
This is a BOCPD-side structural switch.

### 5.2 `use_cusum`

Master switch for the discrepancy refresh patch.
If `False`, the extra rolled-CUSUM refresh logic is inactive.

### 5.3 `cusum_mode`

Current supported values:

- `"cumulative"`
- `"standardized_gate"`

#### `cumulative`

This is the earlier cumulative drift statistic:

$$
d_t = (m_t - m_{t-1})^\top (\Sigma_{t-1} + \epsilon I)^{-1} (m_t - m_{t-1}),
$$

$$
G_t = G_{t-1} + d_t.
$$

If $G_t > h$, discrepancy memory is refreshed.

This is best interpreted as a cumulative standardized drift budget, not textbook centered CUSUM.

#### `standardized_gate`

This is the current preferred lightweight gate.
Define

$$
z_t = \sqrt{d_t},
$$

with the same Mahalanobis-type increment score $d_t$ above.
Then discrepancy memory is refreshed if the standardized move exceeds a threshold, for example:

$$
z_t > \tau_{gate}
$$

for one or more consecutive batches.

Current config controls:

- `standardized_gate_threshold`
- `standardized_gate_consecutive`

Interpretation:

- half-discrepancy handles weak sustained drift at the predictive level,
- the standardized gate is a safeguard for local latent moves that PF can absorb without BOCPD restart,
- gate-triggered action is discrepancy refresh, not full restart.

### 5.4 `cusum_recent_obs`

Controls how much recent discrepancy history is retained during a refresh.
The refresh truncates discrepancy training memory to the most recent observations instead of resetting the full expert or PF state.

## 6. Discrepancy-use switches

These are easy to confuse, so they should always be documented together.

### 6.1 `use_discrepancy`

This controls whether the main predictive side uses discrepancy.

### 6.2 `bocpd_use_discrepancy`

This controls whether BOCPD-side scoring / restart attribution uses discrepancy-aware predictive laws.

A useful interpretation is:

- `use_discrepancy=False, bocpd_use_discrepancy=False`
  means nodiscrepancy on both the predictive and BOCPD-side logic selected by that runner.
- `use_discrepancy=False, bocpd_use_discrepancy=True`
  is the half-discrepancy style split.
- `use_discrepancy=True, bocpd_use_discrepancy=True`
  is the fully discrepancy-aware variant.

When reading code or experiments, always check both switches together.

## 7. Current configuration summary

### 7.1 Restart / refresh controls

- `restart_impl`
  - use `rolled_cusum_260324` for this line.
- `use_dual_restart`
- `use_cusum`
- `cusum_mode`
- `cusum_threshold`
- `cusum_recent_obs`
- `cusum_cov_eps`
- `standardized_gate_threshold`
- `standardized_gate_consecutive`

### 7.2 Hybrid BOCPD controls

- `hybrid_tau_delta`
- `hybrid_tau_theta`
- `hybrid_tau_full`
- `hybrid_delta_share_rho`
- `hybrid_pf_sigma_mode`
- `hybrid_sigma_delta_alpha`
- `hybrid_sigma_ema_beta`
- `hybrid_sigma_min`
- `hybrid_sigma_max`

### 7.3 Discrepancy controls

- `use_discrepancy`
- `bocpd_use_discrepancy`
- `particle_delta_mode`
  - `shared_gp`
  - `particle_gp_shared_hyper`
  - `particle_gp_hyper_pool`
  - `particle_basis`
- `particle_gp_hyper_candidates`
- `particle_basis_kind`
- `particle_basis_num_features`
- `particle_basis_lengthscale`
- `particle_basis_ridge`
- `particle_basis_noise`

## 8. Current interface summary

### 8.1 BOCPD implementation file

Primary implementation:

- `calib/restart_bocpd_rolled_cusum_260324_gpytorch.py`

Key responsibilities:

- preserve the hybrid restart implementation,
- add refresh logic that is no-op-safe,
- construct discrepancy states according to `particle_delta_mode`,
- fall back to shared GP if richer discrepancy construction fails.

### 8.2 Discrepancy state file

Primary discrepancy extensions:

- `calib/particle_specific_discrepancy.py`

Current state classes:

- `ParticleSpecificGPDeltaState`
- `ParticleSpecificBasisDeltaState`

### 8.3 Likelihood hook

`calib/likelihood.py` must remain aware that discrepancy prediction can be either:

- shared across particles, or
- particle-specific via `predict_for_particles(...)`.

This is a backward-compatible interface hook, not a change of the likelihood formula itself.

## 9. Runner-level method naming

Current recommended naming style is ablation-oriented.
Examples:

- `RBOCPD_half_STDGate`
- `RBOCPD_half_STDGate_particleGP`
- `RBOCPD_half_STDGate_particleBasis`

Recommendation:

- add new variants rather than mutating these names,
- keep the method name descriptive enough that a results table is readable without digging into config.

## 10. Efficiency notes

The efficiency rationale for particle-specific discrepancy is:

- residual vectors differ across particles because $\theta_i$ changes,
- but kernel matrices only depend on $X$ and hyperparameters,
- so kernel factorizations can be shared whenever hyperparameters are shared,
- and even in a small hyper-pool, the expensive matrix work is reused candidate-by-candidate.

This is why particle-specific discrepancy is practical here despite many particles.

## 11. Recommended maintenance checklist

Whenever rolled-CUSUM code changes, update this document with:

1. the exact new config switch or method name,
2. whether it changes PF, BOCPD scoring, restart, or refresh only,
3. the mathematical object being approximated,
4. the fallback / backward-compatible behavior,
5. and at least one recommended smoke-test command.

## 12. Example smoke-test style

Use the `jumpGP` environment.
Prefer a short synthetic run that exercises the modified branch, for example a small `run_one_slope(...)` call with one method entry.

The goal of the smoke test is not to validate final metrics.
It is to verify that:

- the new interface wires correctly,
- prediction shape logic is correct,
- and the modified branch can run through at least one prediction-update cycle.


## 13. Runner Invocation Reference

These are the current recommended command-line entrypoints for the synthetic rolled-CUSUM experiments.
Use the `jumpGP` environment.

### 13.1 Gradual drift main/ablation/appendix

Runner:

- `calib.run_synthetic_slope_deltaCmp`

Supported profiles:

- `--profile main`
  - current default behavior for the slope script
- `--profile ablation`
  - fixed gradual-drift ablation setting
  - `mode=1`, `slope=0.0025`, `batch_size=20`, `seeds=[101,202,303,404,505]`
  - writes `ablation_gradual_metrics.csv/.xlsx`
- `--profile appendix`
  - fixed gradual appendix comparison for shared vs particle-specific discrepancy
  - writes `appendix_extension_gradual_metrics.csv/.xlsx`

Example commands:

```bash
conda run -n jumpGP python -m calib.run_synthetic_slope_deltaCmp --profile main --out_dir figs/slope_main
conda run -n jumpGP python -m calib.run_synthetic_slope_deltaCmp --profile ablation --out_dir figs/slope_ablation
conda run -n jumpGP python -m calib.run_synthetic_slope_deltaCmp --profile appendix --out_dir figs/slope_appendix
```

### 13.2 Sudden change main/ablation

Runner:

- `calib.run_synthetic_suddenCmp_tryThm`

Supported profiles:

- `--profile main`
  - current default sudden-change grid behavior
- `--profile ablation`
  - fixed sudden-change ablation setting
  - `seg_len_L=80`, `delta_mag=2.0`, `batch_size=20`, `seeds=[101,202,303]`
  - writes `ablation_sudden_metrics.csv/.xlsx`
  - writes `ablation_sudden_restart_stats.csv/.xlsx`

Example commands:

```bash
conda run -n jumpGP python -m calib.run_synthetic_suddenCmp_tryThm --profile main --out_dir figs/sudden_main
conda run -n jumpGP python -m calib.run_synthetic_suddenCmp_tryThm --profile ablation --out_dir figs/sudden_ablation
```

### 13.3 Mixed gradual-plus-sudden runner

Runner:

- `calib.run_synthetic_mixed_thetaCmp`

Supported profiles and preview entrypoints:

- `--profile main`
  - mixed scenario grid over `drift_scale x jump_scale x seed`
  - current defaults: `drift_scale in {0.006, 0.009}`, `jump_scale in {0.28, 0.38}`, `batch_size=20`
- `--profile ablation`
  - fixed mixed ablation setting
  - `drift_scale=0.008`, `jump_scale=0.35`, `batch_size=20`, `seeds=[101,202,303,404,505]`
- `--profile preview`
  - only generate true theta trajectories, no method run
- `--preview_only`
  - backdoor mode for visually checking the latent theta paths before launching the experiment
  - writes `mixed_theta_preview.png` and `mixed_theta_preview.csv`

Example commands:

```bash
conda run -n jumpGP python -m calib.run_synthetic_mixed_thetaCmp --preview_only --out_dir figs/mixed_preview
conda run -n jumpGP python -m calib.run_synthetic_mixed_thetaCmp --profile main --out_dir figs/mixed_main
conda run -n jumpGP python -m calib.run_synthetic_mixed_thetaCmp --profile ablation --out_dir figs/mixed_ablation
```

### 13.4 Output conventions

For these runners, `--out_dir` is the main experiment artifact directory.
Typical outputs include:

- per-run `.pt` files,
- aggregate `all_metrics.csv/.xlsx`,
- `restart_mode_stats.csv/.xlsx` when restart histories are available,
- profile-specific ablation tables,
- plots saved under the chosen output directory.

When a runner gains a new profile or a new export file, update this section in the same code change.

## 12. Online Discrepancy Ablation (2026-03)

A new opt-in discrepancy-update family is available for half-discrepancy ablations.
The old behavior remains the default.

### 12.1 New switch

- `delta_update_mode="refit"`
  - existing behavior: recompute discrepancy from expert history using the current PF posterior.
- `delta_update_mode="online"`
  - new ablation behavior: discrepancy uses frozen residuals and append-only updates.
  - old residuals are not reinterpreted by future PF states.

### 12.2 Shared online discrepancy

For the shared discrepancy path, each arriving batch produces a frozen residual target

$$
\tilde r_t = y_t - \rho \sum_i w_{t,i} \eta(x_t, \theta_{t,i}),
$$

and the discrepancy GP appends these residuals over time.
Hyperparameters are fitted once on the initial frozen residual buffer and then held fixed.

This is exposed by method names such as:

- `R-BOCPD-PF-halfdiscrepancy-online`
- `RBOCPD_half_STDGate_online`

### 12.3 Particle online discrepancy

A new particle mode is available:

- `particle_delta_mode="particle_gp_online_shared_hyper"`

Its semantics are lineage-bound and frozen-residual:

- each current particle lineage owns its own discrepancy history,
- when PF resampling duplicates a parent, the children inherit the parent's discrepancy history,
- each child then appends its own new residuals from that point onward,
- old residuals are never recomputed using future particle locations.

The particle online methods use one shared GP hyperparameter setting fitted once from the shared frozen residual buffer, then keep it fixed while the particle-specific residual histories evolve online.

This is exposed by method names such as:

- `RBOCPD_half_particleGP_online`
- `RBOCPD_half_STDGate_particleGP_online`

### 12.4 Gate / refresh interpretation under online discrepancy

When `delta_update_mode="online"` is combined with the standardized gate path, gate-triggered refresh truncates the online discrepancy memory to the recent window instead of rebuilding discrepancy labels from scratch.
PF state and BOCPD structure are otherwise left unchanged by the online discrepancy update rule itself.

### 12.5 Runner coverage

The online ablation methods are wired into these runners:

- `conda run -n jumpGP python -m calib.run_synthetic_mixed_thetaCmp --profile ablation --out_dir ...`
- `conda run -n jumpGP python -m calib.run_synthetic_suddenCmp_tryThm --profile ablation --out_dir ...`
- `conda run -n jumpGP python -m calib.run_synthetic_slope_deltaCmp --profile ablation --out_dir ...`

The CLI entrypoints are unchanged; the new behavior is selected by the method table entries inside each runner.



## 14. Dynamic Online Discrepancy Filtering (2026-03)

The repository now supports a second online discrepancy family in addition to the existing frozen-residual append-only mode.
The old default behavior is still unchanged.

### 14.1 Mode split

- `delta_update_mode="refit"`
  - original behavior.
  - discrepancy labels are rebuilt offline from the current PF posterior over the retained expert history.
- `delta_update_mode="online"`
  - existing frozen-residual append-only ablation.
  - new residuals are appended, but the latent discrepancy state is still the same static GP conditioned on all appended residuals.
- `delta_update_mode="online_dynamic"`
  - new dynamic filtering ablation.
  - previous discrepancy posterior is first propagated into the next-time prior, then updated by the new batch.

This `online_dynamic` mode is the one intended to represent the stronger assumption that the discrepancy itself drifts over time and that newer batches should matter more than remote history.

### 14.2 Shared dynamic discrepancy model

For the half-discrepancy path we approximate the latent discrepancy by a small RBF basis expansion

$$
\delta_t(x) = \phi(x)^\top \beta_t,
$$

where the basis centers are fixed from the initialization buffer and the basis dimension is kept small for computational ease.

The latent coefficients evolve according to

$$
\beta_t \mid \mathcal D_{1:t-1} \sim \mathcal N\left(\beta_{t-1},\; P_{t-1}/\lambda + q I\right),
$$

with:

- `lambda = delta_dynamic_forgetting`
  - covariance inflation / forgetting factor,
- `q = delta_dynamic_process_noise_scale * variance_delta`
  - process-noise scale,
- `variance_delta`
  - the discrepancy variance fitted once from the initialization buffer.

After the propagation step, the new batch residuals

$$
r_t = y_t - \rho \sum_i w_{t,i}\,\eta(x_t, \theta_{t,i})
$$

are assimilated by a linear-Gaussian update in the basis state.

So the dynamic online path is not just "append one more residual".
It is explicitly:

1. take the previous discrepancy posterior,
2. inflate / diffuse it into a new prior,
3. update that prior with the new batch,
4. keep the state dimension fixed to the basis size.

### 14.3 Why this is computationally easier than exact GP online filtering

The dynamic path does **not** grow its state with the history length.
Instead it keeps only:

- a mean vector over basis coefficients,
- a small covariance matrix over the same coefficients,
- a bounded recent buffer used only for initialization and gate-triggered hard refresh.

So the cost scales with the chosen basis size rather than the total number of discrepancy observations.
This is why the dynamic particle version is also practical.

### 14.4 Particle dynamic discrepancy semantics

A new particle mode is available:

- `particle_delta_mode="particle_gp_dynamic_shared_hyper"`

Its semantics are lineage-bound.
For each expert:

- each particle lineage carries its own discrepancy coefficient mean,
- the shared basis covariance is reused across particles,
- resampling copies a parent's discrepancy state into each child lineage,
- each child then continues its own online dynamic updates,
- old particle-specific residuals are not recomputed from future particle locations.

This keeps the intended particle-specific temporal semantics while avoiding one growing exact GP per particle.

### 14.5 Hyperparameter and basis policy

To keep the ablation clean, `online_dynamic` keeps the discrepancy hyperparameters fixed after initialization.
The fitted hyperparameters only determine:

- the RBF basis lengthscale,
- the discrepancy variance scale,
- the observation-noise scale.

They are fitted once from the initial frozen residual buffer using the same gpytorch fitting helper that the existing online ablation uses.
After that, dynamic behavior comes from the recursive propagation and update, not from repeated hyperparameter retraining.

### 14.6 Buffer and refresh semantics

The dynamic path keeps a bounded recent buffer controlled by:

- `delta_dynamic_buffer_max_points`

This buffer is used for:

- initialization before the dynamic state is live,
- standardized-gate hard refresh,
- safe fallback reconstruction if the state must be rebuilt.

Under `STDGate` / standardized-gate refresh:

- `online` truncates the exact frozen-residual GP memory and rebuilds that static state,
- `online_dynamic` truncates the recent buffer and rebuilds the dynamic basis filter from the recent window only.

So the refresh is still a hard memory reset, while the per-step update remains a soft dynamic filtering update.

### 14.7 New config interface

`ModelConfig` now supports these additional knobs for the dynamic path:

- `delta_dynamic_num_features`
- `delta_dynamic_forgetting`
- `delta_dynamic_process_noise_scale`
- `delta_dynamic_prior_var_scale`
- `delta_dynamic_buffer_max_points`

These are opt-in and only affect `delta_update_mode="online_dynamic"`.
The existing `refit` and `online` modes keep their previous behavior.

### 14.8 Runner method names

The synthetic runners now expose dynamic-online variants with explicit names such as:

- `R-BOCPD-PF-halfdiscrepancy-onlineDynamic`
- `RBOCPD_half_STDGate_onlineDynamic`
- `RBOCPD_half_particleGP_onlineDynamic`
- `RBOCPD_half_STDGate_particleGP_onlineDynamic`

The corresponding runner entrypoints remain the same:

```bash
conda run -n jumpGP python -m calib.run_synthetic_mixed_thetaCmp --profile ablation --out_dir figs/mixed_ablation_dynamic
conda run -n jumpGP python -m calib.run_synthetic_suddenCmp_tryThm --profile ablation --out_dir figs/sudden_ablation_dynamic
conda run -n jumpGP python -m calib.run_synthetic_slope_deltaCmp --profile ablation --out_dir figs/slope_ablation_dynamic
```

### 14.9 Interpretation summary

The three discrepancy interpretations are now:

- `refit`
  - discrepancy is a history-reinterpreting predictive law attached to the current PF posterior.
- `online`
  - discrepancy is a frozen-residual static online memory.
- `online_dynamic`
  - discrepancy is a time-varying latent state filtered recursively over batches.

## 15. Experiment Execution Standards

These are the default experiment conventions unless a run explicitly overrides them.

### 15.1 Default synthetic standards

- Default particle count for synthetic BOCPD / PF experiments: `1024`.
- Default `delta_bpc_lambda` for the online-BPC comparison suite: `2.0`.
- Default large synthetic seed range: `0..24`.
- Default large synthetic configuration grids:
  - `sudden`
    - `magnitudes = [0.5, 1.0, 2.0, 3.0]`
    - `batch_sizes = [20]`
    - `seg_lens = [80, 120, 200]`
  - `slope`
    - `batch_sizes = [20]`
    - `slopes = [0.0005, 0.001, 0.0015, 0.002, 0.0025]`
  - `mixed`
    - `batch_sizes = [20]`
    - `drift_scales = [0.008]`
    - `jump_scales = [0.28, 0.38, 0.58]`
    - `total_T = 600`

### 15.2 Default method set for the large synthetic CPD suite

Unless a run is explicitly narrowed, the synthetic suite should use:

- `Proxy_None`
- `Proxy_BOCPD`
- `Proxy_wCUSUM`
- `Exact_BOCPD`
- `Exact_wCUSUM`
- `FixedSupport_None`
- `FixedSupport_BOCPD`
- `FixedSupport_wCUSUM`
- `ParticleFixedSupport_None`
- `ParticleFixedSupport_BOCPD`
- `ParticleFixedSupport_wCUSUM`
- `PF-OGP`
- `SlidingWindow-KOH`
- `BOCPD-PF-OGP`

`Exact_None` is excluded by default from long large-suite runs because the expanding-support exact path is the most numerically fragile in single-segment mode and is not needed for the core comparison.

### 15.3 Required raw payload per seed/config/method

For every retained experiment run, save enough information to regenerate tracking plots without rerunning:

- method name
- scenario/config identifiers
- seed
- batch size
- `theta`
- `theta_oracle`
- `theta_var`
- `restart_mode_hist` if applicable
- `others`
- `top0_particles_hist` if applicable
- `rmse`
- `crps_hist`
- `X_batches`
- `Y_batches`
- `y_noiseless_batches`
- `pred_mu_batches`
- `pred_var_batches`
- elapsed/runtime field

The practical rule is:

- every large experiment should keep per-run `.pt` payloads,
- and the summary directory should also keep plot-ready `.csv` files whenever a downstream paper figure is likely.

### 15.4 Mandatory confirmation before large experiments

Before launching a large experiment, explicitly confirm with the user:

- exact method list
- seed range
- scenario/configuration grid
- particle count
- key hyperparameters such as `delta_bpc_lambda`
- controller / CPD variants
- whether per-seed raw payload and tracking outputs are required
- which final metrics must be tabulated

Do not assume a previous large-experiment spec silently carries over.

### 15.5 Smoke-test cleanup rule

After a smoke test has been analyzed, delete its output folder unless the user explicitly asks to keep it.

The intent is:

- smoke tests should validate a code path,
- not pollute `figs/` with stale result folders.

### 15.6 PlantSim conventions

- Default particle count for PlantSim comparison runs: `1024`.
- For seeded PlantSim runs, preserve `seed_runs/seedXX/plantSim_results_modeY.pt`.
- For explicit single-seed PlantSim runs, preserve or regenerate theta-tracking plots.
- If inline theta-tracking plotting is disabled in the runner, a companion postprocess path must be kept available.
- For exploratory single-seed PlantSim figure runs with `modes 0/1/2`, if the user has specified the standing truncation rule, `mode 1` must be capped with:
  - `--max-batches-by-mode 1:250`
  - do not silently run full `mode 1`
- If rerunning only a subset of modes into an existing PlantSim output folder, preserve the summary rows for untouched modes when rewriting `plant_method_mode_seed_summary.csv` and `plant_method_mode_summary.csv`.

`run_plantSim_v3_std.py` now supports:

- `--seeds`
- `--max-batches-by-mode`
- profile `half_exact_damove_bc`

Example:

```bash
conda run -n jumpGP python -m calib.run_plantSim_v3_std --out_dir figs/plantSim_example --profile half_exact_da_bc --modes 0 1 2 --batch_size 4 --num_particles 1024 --seeds 13 --max-batches-by-mode 1:250
```

Move-step DA variant example:

```bash
conda run -n jumpGP python -m calib.run_plantSim_v3_std --out_dir figs/plantSim_example_damove --profile half_exact_damove_bc --modes 0 1 2 --batch_size 4 --num_particles 1024 --seeds 13
```

For this profile, `DA` means the generalized Ward-style paper PF with move-step:

- `type="paper_pf"`
- PlantSim uses 5D standardized `x`
- `paper_pf_design_x_points = 32`
- `paper_pf_design_theta_points = 7`
- default `paper_pf_sigma_obs_var = sigma_eps_s^2`
- default `paper_pf_move_theta_std_s = 0.15 / theta_sd`
- `paper_pf_move_logl_std = 0.10`

### 15.7 Synthetic representative theta-tracking bundle

For one-off paper-style theta-tracking figures on synthetic data, use:

```bash
conda run -n jumpGP python tools/run_synthetic_theta_tracking_bundle.py --out_dir figs/synthetic_theta_tracking_bundle_example --num-particles 1024 --delta-bpc-lambda 2
```

This tool runs one representative configuration for:

- `sudden`
- `slope`
- `mixed`

and compares:

- `HalfRefit`
- `Exact_BOCPD`
- `Exact_wCUSUM`
- `DA`
- `BC`

It writes:

- `*_results.pt`
- `*_meta.pt`
- `*_theta_tracking.csv`
- `*_theta_tracking.png`
- `selected_config_method_summary.csv`
- `selected_config_manifest.csv`

`DA` in this bundle should default to the Ward-style paper PF with the move-step diagnostic enabled:

- `type="paper_pf"`
- `paper_pf_sigma_obs_var = 0.04`
- `paper_pf_move_theta_std = 0.15`
- `paper_pf_move_logl_std = 0.10`

## 16. Ward et al. (2021) paper PF reproduction

### 16.1 New synthetic method path

- Added a fresh opt-in synthetic method branch:
  - `type="paper_pf"`
- Code hook:
  - `calib/paper_pf_digital_twin.py`
- Native runner wiring:
  - `calib/run_synthetic_suddenCmp_tryThm.py`
  - `calib/run_synthetic_slope_deltaCmp.py`
  - `calib/run_synthetic_mixed_thetaCmp.py`

### 16.2 Intended interpretation

- This path is meant to reproduce the PF described in:
  - Ward et al. (2021), *Continuous calibration of a digital twin: Comparison of particle filter and Bayesian calibration approaches*.
- Implemented paper semantics:
  - particles over `(theta, l)`
  - fixed `rho = 1`
  - fixed measurement variance
  - GP emulator built from prior simulator runs over the calibration range
  - model-discrepancy GP included in the particle likelihood
  - resample every step
  - no rejuvenation / move kernel after resampling
- This is intentionally different from the repo’s older `DA` / `PFWithGPPrediction` path, which:
  - uses simulator-only PF likelihood,
  - then refits a residual GP for prediction.

### 16.3 Explicit assumptions required by the paper’s missing details

- The paper does **not** specify the lognormal parameters used for the PF length-scale prior.
- Therefore the synthetic reproduction fixes:
  - `paper_pf_prior_l_median = 0.30`
  - `paper_pf_prior_l_logsd = 0.50`
  - `paper_pf_l_min = 0.05`
  - `paper_pf_l_max = 3.00`
- The paper wording around `sigma^2` is ambiguous because Table 1 is written in terms of precision hyperparameters. The current reproduction interprets the Gamma priors as precision priors and uses inverse prior-mean precisions as fixed variances:
  - emulator variance `= 1 / E[lambda_eta] = 1.0`
  - discrepancy variance `= 1 / E[lambda_b] ≈ 0.03`
  - measurement variance `= 1 / E[lambda_e] ≈ 0.003`
- These assumptions should be stated whenever this path is reported as a “paper PF reproduction,” because they are not fully recoverable from the article text alone.

### 16.4 Representative comparison command

- Tool:
  - `tools/run_ward_pf_vs_brpce.py`
- Purpose:
  - compare `WardPaperPF` against `BRPC-E`
  - where `BRPC-E` is exact online-BPC with no BOCPD and no wCUSUM:
    - `type="bocpd"`
    - `mode="single_segment"`
    - `controller_name="none"`
    - `delta_update_mode="online_bpc_exact"`

Example command:

```bash
conda run -n jumpGP python tools/run_ward_pf_vs_brpce.py \
  --out_dir figs/wardpf_vs_brpce_20260425 \
  --num-particles 1024 \
  --delta-bpc-lambda 2 \
  --batch-size 20 \
  --sudden-seed 13 \
  --slope-seed 13 \
  --mixed-seed 202 \
  --sudden-mag 2.0 \
  --sudden-seg-len 120 \
  --slope 0.0015 \
  --mixed-drift-scale 0.008 \
  --mixed-jump-scale 0.38 \
  --mixed-total-T 600
```

Useful sensitivity flags for this tool:

- `--paper-pf-sigma-obs-var`
- `--paper-pf-discrepancy-var`
- `--paper-pf-emulator-var`
- `--paper-pf-prior-l-median`
- `--paper-pf-prior-l-logsd`
- `--include-move-variant`
- `--paper-pf-move-theta-std`
- `--paper-pf-move-logl-std`

These were added because the default paper-style `sigma^2` choice can be much smaller than the synthetic benchmark noise level (`noise_sd = 0.2`, variance `0.04`), which can cause the paper PF to collapse onto a nearly constant theta path after very few batches.

### 16.5 Move-step diagnostic variant

- A non-paper diagnostic variant is now available:
  - `WardPaperPF_Move`
- Interpretation:
  - same likelihood as `WardPaperPF`
  - after systematic resampling, apply a simple random-walk rejuvenation step:
    - `theta <- theta + N(0, move_theta_std^2)`
    - `l <- l * exp(N(0, move_logl_std^2))`
  - then clamp back to the configured feasible ranges
- This is **not** part of the original Ward et al. method. It is a diagnostic PF-plus-move variant used to test whether the paper PF under-tracks because of particle impoverishment.

### 16.6 Synthetic-comparison DA convention

For synthetic comparison bundles and synthetic-suite experiments, the default `DA` baseline should be interpreted as the move-step Ward-paper PF:

- `type="paper_pf"`
- `paper_pf_sigma_obs_var = 0.04`
- `paper_pf_move_theta_std = 0.15`
- `paper_pf_move_logl_std = 0.10`

This is a reporting convention for synthetic baseline comparisons. It does **not** require retroactively changing unrelated historical result folders.

### 16.7 DA-only synthetic suite command

To run the synthetic large-experiment grid for the default `DA` baseline only:

```bash
conda run -n jumpGP python -m calib.run_synthetic_cpd_suite \
  --out_dir figs/synthetic_cpd_suite_da_seed5_np1024 \
  --scenario all \
  --seed_count 5 \
  --num_particles 1024 \
  --delta_bpc_lambda 2 \
  --methods DA
```

This uses the standard synthetic large-experiment grids:

- sudden:
  - `magnitudes = [0.5, 1.0, 2.0, 3.0]`
  - `seg_lens = [80, 120, 200]`
  - `batch_sizes = [20]`
- slope:
  - `slopes = [0.0005, 0.001, 0.0015, 0.002, 0.0025]`
  - `batch_sizes = [20]`
- mixed:
  - `drift_scales = [0.008]`
  - `jump_scales = [0.28, 0.38, 0.58]`
  - `total_T = 600`
  - `batch_sizes = [20]`

Expected outputs include:

- `raw/all_runs.csv`
- `raw/errors.csv`
- `raw_runs/<scenario>/*.pt`
- `summary/*.csv`
- `summary/raw_payload_manifest.csv`

Expected outputs:

- raw run payloads:
  - `*_results.pt`
  - `*_meta.pt`
- plot-ready tracking CSVs:
  - `*_theta_tracking.csv`
- theta-tracking figures:
  - `*_theta_tracking.png`
- summary tables:
  - `selected_config_method_summary.csv`
  - `selected_config_manifest.csv`

### 16.8 WardPF-move + R-BOCPD synthetic comparison

A new synthetic comparison method is available:

- `WardPFMove_BOCPD`

Interpretation:

- BOCPD controller over experts
- each expert carries a `WardPaperPFConfig` particle filter with move step
- default synthetic settings:
  - `type="bocpd_paper_pf"`
  - `paper_pf_sigma_obs_var = 0.04`
  - `paper_pf_move_theta_std = 0.15`
  - `paper_pf_move_logl_std = 0.10`

This path exists to compare BOCPD over Ward-style PF memory against exact online-BPC with BOCPD (`Exact_BOCPD`, i.e. B-BRPC-E).

Example 5-seed large-experiment command:

```bash
conda run -n jumpGP python -m calib.run_synthetic_cpd_suite \
  --out_dir figs/wardpfmove_bocpd_vs_exact_seed5_np1024_lambda2 \
  --scenario all \
  --seed_count 5 \
  --num_particles 1024 \
  --delta_bpc_lambda 2 \
  --methods WardPFMove_BOCPD Exact_BOCPD
```

This uses the standard synthetic large-experiment grids:

- sudden:
  - `magnitudes = [0.5, 1.0, 2.0, 3.0]`
  - `seg_lens = [80, 120, 200]`
  - `batch_sizes = [20]`
- slope:
  - `slopes = [0.0005, 0.001, 0.0015, 0.002, 0.0025]`
  - `batch_sizes = [20]`
- mixed:
  - `drift_scales = [0.008]`
  - `jump_scales = [0.28, 0.38, 0.58]`
  - `total_T = 600`
  - `batch_sizes = [20]`

### 16.9 Theta-tracking output rule

For any synthetic experiment that saves per-run raw payloads:

- theta-tracking must be treated as a required output, not an optional afterthought
- either:
  - the runner writes `*_theta_tracking.png` / `*_theta_tracking.csv` directly, or
  - the same turn must add and document a companion postprocess that reconstructs those plots from raw payloads
- for any single `seed x configuration` comparison involving multiple methods, the default presentation should be a **single combined theta-tracking figure** with all methods on that figure
- per-method theta-tracking figures may still be written for diagnostics, but they do not replace the combined per-seed/per-configuration figure
- raw payloads therefore must continue to preserve enough fields to redraw theta tracking later:
  - `theta`
  - `theta_oracle` and/or `theta_star_true`
  - `batch_size`
  - `cp_times` or `cp_batches` where applicable

Generic postprocess tool for suite-style outputs:

```bash
conda run -n jumpGP python tools/plot_synthetic_theta_tracking_from_raw_runs.py \
  --suite_dir figs/<synthetic_suite_dir>
```

Expected outputs:

- `theta_tracking_plots/*.png`
- `theta_tracking_plots/*.csv`
- `theta_tracking_plots/theta_tracking_manifest.csv`

### 16.10 Moderate high-dimensional projected diagnostic

A dedicated high-dimensional synthetic diagnostic runner is available:

```bash
conda run -n jumpGP python -m calib.run_synthetic_highdim_projected_diag \
  --out_dir figs/highdim_projected_diag_brpcf \
  --scenarios all \
  --seed_count 5 \
  --num_particles 1024 \
  --delta_bpc_lambda 2 \
  --num_support 32 \
  --total_batches 60 \
  --batch_size 64
```

Purpose:

- defensive appendix-style scale diagnostic, not a new main benchmark
- moderate-dimensional input channel:
  - `d_x = 20`
  - `d_theta = 5`
- simulator:
  - additive ridge response with 5 simulator features over the 20D input
- discrepancy:
  - orthogonalized random Fourier feature basis projected off the simulator feature span

Data generation modes:

- `--data_mode orthogonalized_rff`
  - default and backward-compatible
  - physical response is the simulator at the latent 5D path plus an orthogonalized RFF discrepancy
  - theta RMSE is evaluated against the latent 5D path, which is also the L2 projection target under the constructed orthogonality
- `--data_mode physical_projected`
  - opt-in projected-calibration stress test
  - physical response is generated from a different nonlinear function class:
    - `zeta_t(x) = sum_j a_{t,j} g_j(x) + 0.4 h(x)`
  - simulator remains `y_s(x, theta) = phi(x)^T theta`
  - theta RMSE is evaluated against the offline L2 projection target:
    - `theta_t^dagger = (Phi^T Phi + lambda I)^(-1) Phi^T zeta_t(X_ref)`
  - default reference design:
    - `--projection_ref_size 50000`
    - `--projection_ridge 1e-6`
  - this mode does not assume the physical response is simulator plus an orthogonal discrepancy

Supported methods in this runner:

- `FixedSupport_None`
  - paper label: `BRPC-F`
  - fixed-support online-BPC with no BOCPD and no wCUSUM
  - implementation uses the single-segment controller with `controller_name = "none"`
- `Proxy_BOCPD`
  - paper label: `B-BRPC-P`
- `Proxy_wCUSUM`
  - paper label: `C-BRPC-P`
- `FixedSupport_BOCPD`
  - paper label: `B-BRPC-F`
- `Exact_BOCPD`
  - paper label: `B-BRPC-E`
- `Exact_wCUSUM`
  - paper label: `C-BRPC-E`
- `HalfRefit_BOCPD`
  - paper label: `B-BRPC-RRA`
- `SlidingWindow-KOH`
  - paper label: `BC`

Why fixed support rather than proxy for the first-pass high-dimensional BRPC path:

- the original `Proxy_BOCPD` path keeps a growing shared support and is not a sensible first-pass high-dimensional stress test at `batch_size = 64`
- `B-BRPC-F` keeps the discrepancy state at a fixed support size and is the intended scalable BRPC path for this diagnostic

Default diagnostic scenarios:

- `slope`
  - smooth 5D drift with sinusoidal modulation
- `sudden`
  - 3 abrupt jumps at roughly `0.25`, `0.50`, and `0.75` of the stream
- `mixed`
  - smooth drift plus 2 abrupt jumps at roughly `0.35` and `0.70`

Default diagnostics and outputs:

- run-level summary:
  - `summary/run_level.csv`
- aggregated scenario summary:
  - `summary/scenario_summary.csv`
- raw payloads:
  - `raw_runs/*.pt`
- theta tracking:
  - `theta_tracking_plots/*_theta_tracking.png`
  - `theta_tracking_plots/*_theta_tracking.csv`
- manifest:
  - `summary/manifest.json`

Notes on scale:

- the GPT-proposed conceptual benchmark suggested `B = 64` and `T = 300 or 600`
- for the current first-pass BRPC-F diagnostic, the runner defaults to `total_batches = 60`
- this keeps the 20D input / 5D parameter structure intact while keeping BOCPD + PF runtime reasonable at `1024` particles

Representative multi-method command:

```bash
conda run -n jumpGP python -m calib.run_synthetic_highdim_projected_diag \
  --out_dir figs/highdim_projected_diag_compare \
  --scenarios all \
  --seed_count 1 \
  --seed_offset 13 \
  --num_particles 1024 \
  --delta_bpc_lambda 2 \
  --num_support 32 \
  --total_batches 60 \
  --batch_size 64 \
  --methods Proxy_BOCPD Proxy_wCUSUM FixedSupport_BOCPD Exact_BOCPD Exact_wCUSUM HalfRefit_BOCPD SlidingWindow-KOH
```

Representative physical-projected command for `DA`, `BC`, and `C-BRPC-F(lambda=1,t=0.35)`:

```bash
conda run -n jumpGP python -m calib.run_synthetic_highdim_projected_diag \
  --out_dir figs/highdim_physical_projected_diag_l1_t035 \
  --data_mode physical_projected \
  --scenarios all \
  --seed_count 1 \
  --seed_offset 0 \
  --num_particles 1024 \
  --delta_bpc_lambda 1 \
  --num_support 128 \
  --total_batches 60 \
  --batch_size 128 \
  --projection_ref_size 50000 \
  --methods DA SlidingWindow-KOH FixedSupport_wCUSUM \
  --wcusum_threshold 0.35 \
  --wcusum_window 4 \
  --wcusum_kappa 0.25 \
  --wcusum_sigma_floor 0.25
```

Stronger jump physical-projected variant:

```bash
conda run -n jumpGP python -m calib.run_synthetic_highdim_projected_diag \
  --out_dir figs/highdim_physical_projected_diag_strongjump_l1_t025_seed5 \
  --data_mode physical_projected \
  --scenarios sudden mixed \
  --seed_count 5 \
  --seed_offset 0 \
  --num_particles 1024 \
  --delta_bpc_lambda 1 \
  --num_support 128 \
  --total_batches 60 \
  --batch_size 128 \
  --projection_ref_size 50000 \
  --sudden_jump_scale 1.2 \
  --mixed_jump_scale 1.0 \
  --methods DA SlidingWindow-KOH FixedSupport_wCUSUM \
  --wcusum_threshold 0.25 \
  --wcusum_window 4 \
  --wcusum_kappa 0.25 \
  --wcusum_sigma_floor 0.25
```

Strict high-dimensional WardPF-move DA plus wider BC-window rerun:

```bash
conda run -n jumpGP python -m calib.run_synthetic_highdim_projected_diag \
  --out_dir figs/highdim_physical_projected_diag_strongjump_strictda_l1_t035_bc4_seed5 \
  --data_mode physical_projected \
  --scenarios sudden mixed \
  --seed_count 5 \
  --seed_offset 0 \
  --num_particles 1024 \
  --delta_bpc_lambda 1 \
  --num_support 128 \
  --total_batches 60 \
  --batch_size 128 \
  --projection_ref_size 50000 \
  --sudden_jump_scale 1.2 \
  --mixed_jump_scale 1.0 \
  --bc_window_batches 4 \
  --methods DA SlidingWindow-KOH FixedSupport_wCUSUM \
  --wcusum_threshold 0.35 \
  --wcusum_window 4 \
  --wcusum_kappa 0.25 \
  --wcusum_sigma_floor 0.25
```

BRPC-F no-controller high-dimensional run:

```bash
conda run -n jumpGP python -m calib.run_synthetic_highdim_projected_diag \
  --out_dir figs/highdim_physical_projected_diag_strongjump_brpcf_l1_bc4_bs32_seed5 \
  --data_mode physical_projected \
  --scenarios slope sudden mixed \
  --seed_count 5 \
  --seed_offset 0 \
  --num_particles 1024 \
  --delta_bpc_lambda 1 \
  --num_support 128 \
  --total_batches 60 \
  --batch_size 32 \
  --projection_ref_size 50000 \
  --sudden_jump_scale 1.2 \
  --mixed_jump_scale 1.0 \
  --bc_window_batches 4 \
  --methods FixedSupport_None
```

### 16.11 Joint EnKF 1D synthetic baseline

A joint Ensemble Kalman Filter baseline is available for representative 1D synthetic scenarios:

```bash
conda run -n jumpGP python tools/run_1d_joint_enkf_synthetic.py \
  --out_dir figs/joint_enkf_1d_synthetic_seed5 \
  --scenarios all \
  --seed_count 5 \
  --seed_offset 0 \
  --batch_size 20 \
  --n_ensemble 512 \
  --theta_rw_sd 0.035 \
  --beta_rw_sd 0.015
```

Interpretation:

- method name: `JointEnKF`
- state is joint:
  - `z = [theta, beta_0, ..., beta_q]`
- observation law:
  - `y(x) = y_s(x, theta) + b(x)^T beta + eps`
- EnKF updates `theta` and discrepancy-basis coefficients `beta` together from each batch
- this is not a theta-only filter followed by a separate residual GP fit

Representative scenarios in this runner:

- `slope`
  - theta-driven slope path, `theta_slope = 0.0015`, `total_T = 600`
- `sudden`
  - `seg_len_L = 120`, `delta_mag = 2.0`, `total_T = 480`
- `mixed`
  - `drift_scale = 0.008`, `jump_scale = 0.38`, `total_T = 600`

Expected outputs:

- `summary/run_level.csv`
- `summary/scenario_summary.csv`
- `summary/manifest.json`
- `raw_runs/*.pt`
- `theta_tracking_plots/*_theta_tracking.png`
- `theta_tracking_plots/*_theta_tracking.csv`

Sensitivity and selected-parameter run added for the reviewer-facing EnKF baseline:

```bash
conda run -n jumpGP python tools/run_1d_joint_enkf_sensitivity.py \
  --out_dir figs/joint_enkf_1d_sensitivity_seed3_grid \
  --scenarios all \
  --seed_count 3 \
  --seed_offset 0 \
  --batch_size 20 \
  --n_ensemble 512 \
  --theta_rw_grid 0.015 0.035 0.07 \
  --beta_rw_grid 0.005 0.015 0.04 \
  --inflation_grid 1.0 1.02 1.08
```

The sensitivity helper writes:

- `summary/run_level.csv`
- `summary/scenario_summary.csv`
- `summary/best_by_scenario.csv`
- `summary/manifest.json`

Selection rule used for the 5-seed appendix run:

- tune by mean theta-RMSE averaged over `slope`, `sudden`, and `mixed`
- do not tune separate EnKF parameters per scenario
- selected global setting:
  - `theta_rw_sd = 0.035`
  - `beta_rw_sd = 0.04`
  - `covariance_inflation = 1.0`

Top global sensitivity rows from `figs/joint_enkf_1d_sensitivity_seed3_grid`:

| theta_rw_sd | beta_rw_sd | inflation | avg theta-RMSE | avg y-RMSE | avg y-CRPS |
|---:|---:|---:|---:|---:|---:|
| 0.035 | 0.040 | 1.00 | 0.1467 | 0.7484 | 0.4421 |
| 0.035 | 0.040 | 1.02 | 0.1498 | 0.7432 | 0.4393 |
| 0.015 | 0.040 | 1.00 | 0.1549 | 0.7251 | 0.4316 |
| 0.015 | 0.040 | 1.02 | 0.1620 | 0.7135 | 0.4254 |

Formal 1D synthetic 5-seed EnKF run:

```bash
conda run -n jumpGP python tools/run_1d_joint_enkf_synthetic.py \
  --out_dir figs/joint_enkf_1d_synthetic_best_global_seed5 \
  --scenarios all \
  --seed_count 5 \
  --seed_offset 0 \
  --batch_size 20 \
  --n_ensemble 512 \
  --theta_rw_sd 0.035 \
  --beta_rw_sd 0.04 \
  --covariance_inflation 1.0
```

5-seed summary:

| scenario | theta-RMSE | y-RMSE | y-CRPS |
|---|---:|---:|---:|
| slope | 0.1094 | 0.5481 | 0.2954 |
| sudden | 0.1600 | 0.9823 | 0.6005 |
| mixed | 0.3463 | 0.6540 | 0.3943 |

### 16.12 Empirical transport-stability diagnostic

Assumption 1 should be diagnosed with the pre-update transported predictive precision, not with the post-update precision after assimilating the current batch. The diagnostic script therefore reports two replay quantities:

```text
gamma_prior_t =
lambda_max(M_{t-1}^{-1/2} A_t^T (P_t^-)^{-1} A_t M_{t-1}^{-1/2})
```

and

```text
gamma_post_t =
lambda_max(M_{t-1}^{-1/2} A_t^T C_t^{-1} A_t M_{t-1}^{-1/2}).
```

Here `P_t^- = Q_t + A_t C_{t-1} A_t^T` is the transported predictive covariance and `C_t` is the covariance after the current-batch likelihood update. `gamma_prior` is the closest empirical analogue of Assumption 1. `gamma_post` measures posterior sharpening and can exceed one even when the transport step is stable.

The earlier one-column diagnostic used `C_t^{-1}` and should be interpreted as `gamma_post`, not as Assumption 1 verification.

Diagnostic command:

```bash
conda run -n jumpGP python tools/compute_transport_gamma_diagnostic.py \
  --out_dir figs/transport_gamma_true_prior_post_synth_plant012_seed5_dedup \
  --synthetic_raw_dir figs/synthetic_representative_sensitivity_seed5_np1024_20260426_234803/raw_runs \
  --include_plant \
  --plant_modes 0 1 2 \
  --plant_seed_count 5 \
  --plant_batch_size 4 \
  --plant_max_batches 250 \
  --lengthscale 1.0 \
  --variance 0.01 \
  --obs_noise 0.0025 \
  --lambda_delta 1.0
```

Outputs:

- `gamma_run_level.csv`
- `gamma_summary.csv`
- `manifest.json`

`lambda_delta` only affects `gamma_post` through the effective likelihood noise `R_eff = lambda_delta R`; `gamma_prior` is a transport-only quantity.

Summary from `figs/transport_gamma_true_prior_post_synth_plant012_seed5_dedup`:

| dataset | scenario | gamma_prior median | gamma_prior p90 | gamma_prior max | frac prior <= 1 | gamma_post median | gamma_post p90 | gamma_post max | frac post <= 1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PlantSim | mode0 | 0.1668 | 0.5630 | 0.9110 | 1.0000 | 0.5846 | 1.3127 | 2.1409 | 0.7510 |
| PlantSim | mode1 | 0.1668 | 0.5630 | 0.9110 | 1.0000 | 0.5846 | 1.3127 | 2.1409 | 0.7510 |
| PlantSim | mode2 | 0.1532 | 0.5278 | 0.9877 | 1.0000 | 0.5846 | 1.2799 | 2.1394 | 0.7726 |
| Synthetic 1D | slope | 0.9999 | 1.0000 | 1.0000 | 0.9724 | 1.0667 | 1.2990 | 1.9867 | 0.0000 |
| Synthetic 1D | sudden | 1.0000 | 1.0000 | 1.0000 | 0.9391 | 1.0836 | 1.3329 | 1.9844 | 0.0000 |
| Synthetic 1D | mixed | 0.9999 | 1.0000 | 1.0000 | 0.9517 | 1.0666 | 1.2990 | 1.9844 | 0.0000 |

By default, synthetic designs are deduplicated and reported as `exact_gp_replay` because this script replays a generic exact-GP transport on the saved `X_batches`; it is not a comparison of E/F/P internal state representations. Use `--keep_synthetic_methods` only to reproduce duplicate method labels from saved payloads.

### 16.13 Joint EnKF PlantSim baseline

PlantSim EnKF uses the same joint state idea as the 1D runner:

```text
z = [theta_s, beta_0, ..., beta_q],
```

where `theta_s` and `x` are in the standardized PlantSim space. The simulator channel is the trained standardized NN emulator, and the discrepancy channel uses RFF basis features over the standardized 5D plant input.

Runner:

```bash
conda run -n jumpGP python tools/run_plantsim_joint_enkf.py \
  --out_dir figs/plantsim_joint_enkf_best_global_seed5 \
  --modes 0 1 2 \
  --seed_count 5 \
  --seed_offset 0 \
  --batch_size 4 \
  --max-batches-by-mode 1:250 \
  --n_ensemble 512 \
  --theta_rw_sd 0.035 \
  --beta_rw_sd 0.04 \
  --covariance_inflation 1.0 \
  --num_basis 32 \
  --basis_lengthscale 1.0
```

Outputs:

- `summary/run_level.csv`
- `summary/scenario_summary.csv`
- `summary/manifest.json`
- `raw_runs/*.pt`
- `theta_tracking_plots/*_theta_tracking.png`
- `theta_tracking_plots/*_theta_tracking.csv`

5-seed summary from `figs/plantsim_joint_enkf_best_global_seed5`:

| scenario | theta-RMSE | y-RMSE | y-CRPS |
|---|---:|---:|---:|
| mode0 | 5.8881 | 115859.1509 | 96874.5835 |
| mode1 | 9.0465 | 82678.5437 | 68780.8173 |
| mode2 | 5.1040 | 122224.0745 | 101716.3782 |

`SlidingWindow-KOH` in this high-dimensional runner is implemented as a generalized sliding-window KOH profile step:

- simulator remains linear in the 5D calibration parameter
- discrepancy is modeled with an RBF GP over the 20D input
- at each update, the KOH parameter estimate is the generalized least-squares solution under the current GP covariance on the sliding window

Notes on the proxy path in this diagnostic:

- `Proxy_BOCPD` / `Proxy_wCUSUM` use the same `online_bpc_proxy_stablemean` memory parameterization as the main synthetic benchmark
- unlike fixed support, proxy still carries a growing support within each segment
- therefore these proxy variants are diagnostic-only here; they are not the preferred scalable high-dimensional path

Targeted fixed-support controller tuning helper:

```bash
conda run -n jumpGP python tools/run_highdim_fixedsupport_tuning.py \
  --out_dir figs/highdim_fixedsupport_tuning \
  --scenarios all \
  --seed 13 \
  --delta_bpc_lambda 2 \
  --num_support 32 \
  --total_batches 60
```

This helper runs a targeted controller grid on the moderate high-dimensional diagnostic for:

- `FixedSupport_BOCPD` (`B-BRPC-F`)
- `FixedSupport_wCUSUM` (`C-BRPC-F`)

with:

- batch sizes:
  - `64`
  - `128`
- particle counts:
  - `1024`
  - `2048`

BOCPD tuning grid:

- `hazard_lambda = 200, restart_margin = 1, restart_cooldown = 10`
- `hazard_lambda = 400, restart_margin = 2, restart_cooldown = 20`
- `hazard_lambda = 800, restart_margin = 4, restart_cooldown = 20`
- `hazard_lambda = 1600, restart_margin = 4, restart_cooldown = 30`

wCUSUM tuning grid:

- `window = 4, threshold = 0.25, kappa = 0.25, sigma_floor = 0.25`
- `window = 4, threshold = 0.50, kappa = 0.25, sigma_floor = 0.25`
- `window = 8, threshold = 0.50, kappa = 0.50, sigma_floor = 0.50`
- `window = 8, threshold = 1.00, kappa = 0.50, sigma_floor = 0.50`

Selected multi-seed appendix helper for the moderate high-dimensional diagnostic:

```bash
conda run -n jumpGP python tools/run_highdim_selected_appendix.py \
  --out_dir figs/highdim_selected_appendix \
  --scenarios all \
  --seed_count 5 \
  --seed_offset 0 \
  --num_particles 1024 \
  --delta_bpc_lambda 2 \
  --num_support 32 \
  --total_batches 60 \
  --batch_size 128
```

This helper runs the selected appendix method set:

- `DA`
  - high-dimensional move-step direct-assimilation PF baseline
  - this is the default `DA` interpretation for the high-dimensional synthetic diagnostic
- `BC`
- `B-BRPC-F`
  - `hazard_lambda = 400`
  - `restart_margin = 2`
  - `restart_cooldown = 640`
  - note:
    - `restart_cooldown` is observation-level in the current R-BOCPD implementation
    - with `batch_size = 128`, this corresponds to a 5-batch cooldown
- `C-BRPC-F`
  - `window = 4`
  - `threshold = 0.25`
  - `kappa = 0.25`
  - `sigma_floor = 0.25`

Expected outputs:

- `summary/run_level.csv`
- `summary/scenario_summary.csv`
- `summary/selected_manifest.json`
- `theta_tracking_plots/*.png`
- `theta_tracking_plots/*.csv`
- `theta_tracking_plots_combined/*.png`
- `theta_tracking_plots_combined/*.csv`

High-dimensional runner theta-tracking rule:

- every `calib.run_synthetic_highdim_projected_diag` run must write combined theta-tracking plots and CSVs, not only per-method plots
- required combined outputs:
  - `theta_tracking_plots_combined/*_theta_tracking.png`
  - `theta_tracking_plots_combined/*_theta_tracking.csv`
  - `theta_tracking_plots_combined/theta_tracking_manifest.csv`
- one combined figure should be written per `scenario x seed x data_mode x batch_size x total_batches`
- all methods run for that seed/configuration should appear on the same combined figure
- if a run is interrupted after raw payloads are written, regenerate combined plots from `summary/run_level.csv` and `raw_runs/*.pt` before reporting results

Standard-BOCPD variant of the selected high-dimensional appendix helper:

```bash
conda run -n jumpGP python tools/run_highdim_selected_appendix.py \
  --out_dir figs/highdim_selected_appendix_standard \
  --scenarios all \
  --seed_count 5 \
  --seed_offset 0 \
  --num_particles 1024 \
  --delta_bpc_lambda 2 \
  --num_support 128 \
  --total_batches 60 \
  --batch_size 128 \
  --fixedsupport_standard_bocpd
```

Interpretation:

- `B-BRPC-F` uses standard BOCPD:
  - `bocpd_mode = "standard"`
  - no hard restart controller
  - no cooldown semantics
  - `hazard_lambda = 400`
- `C-BRPC-F` remains the selected wCUSUM controller:
  - `window = 4`
  - `threshold = 0.25`
  - `kappa = 0.25`
  - `sigma_floor = 0.25`
- this selected standard-BOCPD appendix run uses `num_support = 128`
- `DA` and `BC` are included unchanged

Important correction on the high-dimensional `DA` baseline:

- the earlier high-dimensional `DA` path based on `HighDimMovePF` is **not** a strict WardPF implementation
- it is a direct-assimilation move-step PF analogue only
- it does **not** carry the discrepancy lengthscale inside the particle state
- it does **not** evaluate the Ward-style joint emulator-plus-discrepancy Gaussian likelihood
- therefore those earlier high-dimensional `DA` results must not be used as evidence for the joint-PF-vs-separated-discrepancy question

Strict high-dimensional `DA` correction workflow:

```bash
conda run -n jumpGP python tools/rebuild_highdim_selected_appendix_strict_da.py \
  --source_dir figs/highdim_selected_appendix_standard_seed5_np1024_bs128_ns128_20260427_101723 \
  --out_dir figs/highdim_selected_appendix_standard_strictDA_seed5_np1024_bs128_ns128
```

This helper:

- copies the non-`DA` raw payloads from the source selected appendix directory
- reruns only `DA` under the strict high-dimensional WardPF generalization
- regenerates:
  - `summary/run_level.csv`
  - `summary/scenario_summary.csv`
  - `summary/selected_manifest.json`
  - `theta_tracking_plots/*.png`
  - `theta_tracking_plots/*.csv`
  - `theta_tracking_plots_combined/*.png`
  - `theta_tracking_plots_combined/*.csv`

Interpretation of the strict high-dimensional `DA`:

- particles over `(theta, l)`
- `theta` is vector-valued with `d_theta = 5`
- `l` is a shared scalar lengthscale carried in each particle and updated by log-random-walk move steps
- emulator likelihood is built from a GP over a prior simulator design on the joint `(x, theta)` input
- discrepancy enters through the Ward-style batch covariance term
- this is the correct high-dimensional baseline for any comparison about joint PF versus separated discrepancy updates

One-seed fixed-support support/cooldown sweep helper:

```bash
conda run -n jumpGP python tools/run_highdim_fixedsupport_support_sweep.py \
  --out_dir figs/highdim_fixedsupport_support_sweep \
  --scenarios all \
  --seed 13 \
  --num_particles 1024 \
  --delta_bpc_lambda 2 \
  --total_batches 60 \
  --batch_size 128
```

This helper is for pre-appendix tuning only. It runs:

- `B-BRPC-F_h400_m2_c20`
- `B-BRPC-F_h400_m2_c640`
- `C-BRPC-F_w4_t025_k025_sf025`

over:

- `num_support in {32, 64, 128, 256}`

Expected outputs:

- `summary/run_level.csv`
- `summary/scenario_summary.csv`
- `summary/support_sweep_manifest.json`
- `theta_tracking_plots_combined/*_theta_tracking.png`
- `theta_tracking_plots_combined/*_theta_tracking.csv`

Plotting rule for this sweep:

- one combined theta-tracking figure per `scenario x seed x num_support`
- all competing methods for that support size must appear on the same figure

DA default rule for synthetic experiments:

- whenever a synthetic comparison includes `DA`, the default meaning is the move-step `WardPF_move` family baseline
- in one-dimensional synthetic experiments this is the existing `WardPF_move` path
- in the moderate high-dimensional diagnostic the default `DA` should be the strict high-dimensional WardPF generalization with particles over `(theta, l)`
- the old direct-assimilation move-step PF analogue should be treated as debug-only and, if needed, exposed under a separate non-default method name such as `DA_Analogue`

Representative one-dimensional sensitivity helper:

```bash
conda run -n jumpGP python tools/run_synthetic_representative_sensitivity.py \
  --out_dir figs/synthetic_representative_sensitivity \
  --seed_count 5 \
  --seed_offset 0 \
  --num_particles 1024
```

This helper is intentionally not a full benchmark sweep. It runs representative controller-sensitivity experiments on the one-dimensional synthetic benchmark for:

- `B-BRPC-E` BOCPD settings:
  - `h200_m1_c10`
  - `h400_m2_c20`
  - `h800_m4_c20`
  - `h1600_m4_c30`
- `C-BRPC-E` wCUSUM settings:
  - `w4_t025_k025_sf025`
  - `w4_t050_k025_sf025`
  - `w8_t050_k050_sf050`
  - `w8_t100_k050_sf050`

Representative configurations:

- sudden:
  - `magnitude = 2.0`
  - `seg_len = 120`
  - `batch_size = 20`
- slope:
  - `slope = 0.0015`
  - `batch_size = 20`
  - uses the established theta-driven slope path (`mode = 1`)
- mixed:
  - `drift_scale = 0.008`
  - `jump_scale = 0.38`
  - `batch_size = 20`
  - `total_T = 600`

Expected outputs:

- `summary/run_level.csv`
- `summary/scenario_summary.csv`
- `summary/selected_manifest.json`
- `summary/theta_tracking_manifest.csv`
- `summary_plots/*.png`
- `theta_tracking_plots/*.png`
- `theta_tracking_plots/*.csv`

Event-quality postprocess for this representative sensitivity run:

```bash
conda run -n jumpGP python tools/postprocess_synthetic_representative_sensitivity_cp.py \
  --results_dir figs/synthetic_representative_sensitivity_seed5_np1024_20260426_234803 \
  --tolerance 2
```

This postprocess reads the saved `raw_runs/*.pt` payloads and computes event-based
restart quality metrics for the representative one-dimensional sensitivity study
without rerunning the model experiments.

Metric definition:

- only `sudden` and `mixed` receive event metrics
- `slope` has no true discrete changepoints, so its event metrics are recorded as `NaN`
- a detection counts as a true positive only if it occurs in the forward window
  `[tau, tau + 2]` for a true changepoint batch `tau`
- no credit is given for early restarts

Expected outputs:

- `summary/cp_quality_run_level.csv`
- `summary/run_level_with_cp_quality.csv`
- `summary/cp_quality_scenario_summary.csv`
- `summary/cp_quality_family_range_summary.csv`

Current empirical takeaways from the completed representative one-dimensional sensitivity run
(`figs/synthetic_representative_sensitivity_seed5_np1024_20260426_234803`):

- the main qualitative controller conclusion is stable across the tested hyperparameter grids
- `B-BRPC-E` remains a high-restart controller family
  - slope restart range:
    - `13.2 -- 13.4`
  - sudden restart range:
    - `9.6 -- 9.8`
  - mixed restart range:
    - `12.6 -- 13.0`
- `C-BRPC-E` remains a lower-restart controller family
  - slope restart range:
    - `4.6 -- 5.0`
  - sudden restart range:
    - `3.0 -- 3.2`
  - mixed restart range:
    - `3.2 -- 3.4`
- event-quality metrics confirm that `C-BRPC-E` is not merely under-restarting
  - sudden:
    - `B-BRPC-E`
      - `precision@2 = 0.286 -- 0.293`
      - `recall@2 = 0.933`
      - `F1@2 = 0.436 -- 0.444`
      - `mean delay = 0.10 -- 0.17`
    - `C-BRPC-E`
      - `precision@2 = 0.700 -- 0.867`
      - `recall@2 = 0.733 -- 0.867`
      - `F1@2 = 0.714 -- 0.867`
      - `mean delay = 0.0`
  - mixed:
    - `B-BRPC-E`
      - `precision@2 = 0.154 -- 0.159`
      - `recall@2 = 1.0`
      - `F1@2 = 0.267 -- 0.275`
      - `mean delay = 0.2 -- 0.3`
    - `C-BRPC-E`
      - `precision@2 = 0.533 -- 0.600`
      - `recall@2 = 0.9 -- 1.0`
      - `F1@2 = 0.667 -- 0.747`
      - `mean delay = 0.1 -- 0.3`
- interpretation:
  - `B-BRPC-E` misses very few true changepoints but over-restarts heavily, so its precision is poor
  - `C-BRPC-E` substantially reduces restart burden and improves event-level quality, at the cost of a predictable `y`-RMSE tradeoff

Current empirical takeaways from the completed single-seed high-dimensional comparison
(`figs/highdim_projected_diag_compare_seed13_20260426`):

- the moderate-dimensional projected diagnostic does not break the projected-calibration story
- the main failure mode is still controller-side, not state-side
- in this generator, `BC` is competitive because:
  - the simulator is linear in the 5D calibration parameter
  - the discrepancy basis is explicitly orthogonalized against the simulator feature span
- fixed-support BRPC remains the more relevant scalable BRPC path than expanding-support Exact or Proxy

Single-seed comparison summary:

- slope:
  - `B-BRPC-F`
    - `theta_rmse = 0.139`
    - `y_rmse = 0.413`
    - `restart_count = 33`
  - `C-BRPC-E`
    - `theta_rmse = 0.155`
    - `y_rmse = 0.476`
    - `restart_count = 9`
  - `BC`
    - `theta_rmse = 0.042`
    - `y_rmse = 0.370`
    - `restart_count = 0`
- sudden:
  - `B-BRPC-F`
    - `theta_rmse = 0.151`
    - `y_rmse = 0.439`
    - `restart_count = 37`
    - `F1@2 = 0.150`
  - `C-BRPC-E`
    - `theta_rmse = 0.162`
    - `y_rmse = 0.478`
    - `restart_count = 9`
    - `F1@2 = 0.500`
  - `B-BRPC-RRA`
    - `theta_rmse = 0.104`
    - `y_rmse = 0.379`
    - `restart_count = 23`
  - `BC`
    - `theta_rmse = 0.093`
    - `y_rmse = 0.423`
    - `restart_count = 0`
- mixed:
  - `B-BRPC-F`
    - `theta_rmse = 0.100`
    - `y_rmse = 0.383`
    - `restart_count = 31`
    - `F1@2 = 0.121`
  - `C-BRPC-E`
    - `theta_rmse = 0.180`
    - `y_rmse = 0.502`
    - `restart_count = 9`
    - `F1@2 = 0.182`
  - `B-BRPC-RRA`
    - `theta_rmse = 0.113`
    - `y_rmse = 0.372`
    - `restart_count = 24`
  - `BC`
    - `theta_rmse = 0.071`
    - `y_rmse = 0.389`
    - `restart_count = 0`

Interpretation of this high-dimensional single-seed comparison:

- `B-BRPC-F` tracks reasonably in theta and y, so the projected fixed-support state is viable in `d_x=20, d_theta=5`
- the dominant issue is still over-restart under BOCPD-style control
- score-based control reduces restart burden sharply, but can be conservative and can sacrifice predictive accuracy
- because the generator is structured and orthogonalized, `BC` should be interpreted as a strong structured baseline here, not as evidence that sliding-window KOH dominates BRPC in general

Practical lesson from the recent high-dimensional tuning cycle:

- the first selected high-dimensional appendix setup with `num_support = 32` is too fragile to be treated as the default scalable fixed-support setting
- for the next multi-seed appendix pass, the selected fixed-support runs were moved to:
  - `num_support = 128`
  - `batch_size = 128`
  - `num_particles = 1024`
- the current appendix-standard run also switches `B-BRPC-F` from hard-restart R-BOCPD to standard BOCPD
  - motivation:
    - isolate the predictive-mixture effect without layering an additional hard-restart controller
    - avoid letting cooldown tuning dominate the appendix narrative
