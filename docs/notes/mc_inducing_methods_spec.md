# MC-Inducing Discrepancy Methods Spec
## For Codex implementation on top of the current half-discrepancy / DOC framework

This document specifies two new discrepancy-update families to implement on top of the **existing half-discrepancy / DOC framework**:

1. **Shared expert-level MC-inducing discrepancy posterior**
2. **Particle-specific MC-inducing discrepancy posterior**

It also specifies a separate implementation of the **digital-twin PF style** method from Ward et al. (PF with discrepancy directly inside the particle likelihood), with an **optional BOCPD wrapper**.

The goal of this document is to be concrete enough that Codex can directly generate code from it.

---

## 0. Scope and design principles

### 0.1 What must remain unchanged from the current half-discrepancy framework

The current main method has three structural principles that should remain intact unless explicitly running an ablation:

1. **BOCPD / R-BOCPD remains the outer segmentation and restart mechanism.**
2. **The PF update for `theta` remains discrepancy-free.**
3. **Discrepancy enters expert-level predictive scoring, not the PF weighting step.**

This means the default implementation order for one incoming batch `(X_t, Y_t)` is:

1. Use the **pre-update expert states** to compute expert evidence / BOCPD losses.
2. Run BOCPD / restart / pruning.
3. For the surviving experts, update the `theta` particle filters using the discrepancy-free PF likelihood.
4. Then update the discrepancy state for those surviving experts.

This prequential ordering is important. Changepoint detection must be based on pre-update predictive evidence, not on models that have already absorbed the current batch.

### 0.2 What is new in this spec

This spec adds a new discrepancy representation:

- use **inducing variables** `u = delta(Z)` at inducing points `Z`
- represent the discrepancy posterior either as:
  - a **shared expert-level weighted particle approximation**, or
  - a **particle-specific weighted particle approximation** conditioned on each `theta` particle

This is **not** the current main method. It is a new online discrepancy family and should be implemented as an **ablation / optional mode**.

### 0.3 Important conceptual warning

The current main `half-discrepancy` refit method uses a **current-endpoint residual target** and effectively **rebuilds the discrepancy training target using the current anchor**. A pure online posterior update over discrepancy does **not** do that automatically. Therefore:

- the methods below are **not mathematically identical** to the current full refit design;
- they are best interpreted as **online discrepancy approximations / alternatives**.

This is fine, but the code and comments should say so clearly.

---

## 1. Baseline DOC / half-discrepancy objects to preserve

For each active BOCPD expert `e`, maintain:

- a `theta` particle system

  \[
  q_{e,t}(\theta)
  =
  \sum_{i=1}^N w_{e,t}^{(i)} \, \delta_{\theta_{e,t}^{(i)}}(\theta),
  \]

- an expert log-mass / BOCPD mass,
- expert history buffers `(X_hist, Y_hist)`,
- a discrepancy state.

The existing PF update remains:

\[
\theta_{e,t}^{(i)} \sim p(\theta_t \mid \theta_{e,t-1}^{(i)}),
\]

\[
p_{\mathrm{PF}}(Y_t \mid X_t,\theta_{e,t}^{(i)})
=
\mathcal{N}\!\big(
Y_t;\,
\mu_s(X_t,\theta_{e,t}^{(i)}),
\Sigma_s(X_t,\theta_{e,t}^{(i)}) + \sigma_{\mathrm{PF}}^2 I
\big),
\]

\[
\widetilde w_{e,t}^{(i)}
\propto
w_{e,t-1}^{(i)}\,
p_{\mathrm{PF}}(Y_t \mid X_t,\theta_{e,t}^{(i)}),
\qquad
w_{e,t}^{(i)}
=
\frac{\widetilde w_{e,t}^{(i)}}{\sum_j \widetilde w_{e,t}^{(j)}}.
\]

This PF step should remain discrepancy-free in the default half-discrepancy versions below.

---

## 2. Shared expert-level MC-inducing discrepancy posterior

This is the recommended version if the goal is to stay closest to the current half-discrepancy principle.

### 2.1 Representation

For each expert `e`, choose inducing points

\[
Z_e = \{z_{e,1},\dots,z_{e,J}\},
\qquad
u_e := \delta_e(Z_e)\in\mathbb{R}^J.
\]

Optionally use a small discrete set of GP hyperparameter configurations

\[
\ell_1,\dots,\ell_C,
\]

where each `ell_c` includes e.g. lengthscale(s), variance, and noise.

Then represent the discrepancy posterior for expert `e` as

\[
q_{e,t}(u,\ell)
=
\sum_{m=1}^{M}\sum_{c=1}^{C}
\alpha_{e,t}^{(m,c)}\,
\delta_{u_{e,t}^{(m)}}(u)\,
\delta_{\ell_c}(\ell).
\]

Recommended first implementation:

- **start with `C = 1`** (fixed hyperparameters within the segment),
- only sample / weight `u`,
- add `C > 1` later if needed.

Then the simplified version is

\[
q_{e,t}(u)
=
\sum_{m=1}^{M}
\alpha_{e,t}^{(m)}\,
\delta_{u_{e,t}^{(m)}}(u).
\]

### 2.2 Inducing-point GP mapping

Given inducing variables `u` at inducing points `Z_e`, the discrepancy at batch inputs `X_t` is approximated by the GP conditional mean

\[
\delta_e(X_t;u,\ell)
\approx
A_{e,t}^{(\ell)} u,
\qquad
A_{e,t}^{(\ell)}
=
K_{X_t Z_e}^{(\ell)} \left(K_{Z_e Z_e}^{(\ell)}\right)^{-1}.
\]

Optionally include conditional covariance

\[
S_{\delta,e,t}^{(\ell)}
=
K_{X_t X_t}^{(\ell)}
-
K_{X_t Z_e}^{(\ell)}
\left(K_{Z_e Z_e}^{(\ell)}\right)^{-1}
K_{Z_e X_t}^{(\ell)}.
\]

In the first implementation, it is acceptable to use only the conditional mean `A u` and treat the inducing approximation as deterministic conditional on `u`. If conditional covariance is included, do so consistently in the expert-level predictive covariance.

### 2.3 Pre-update expert scoring for BOCPD

At batch `t`, before any PF update, compute BOCPD evidence using the **pre-update** states:

- `theta` particles from `q_{e,t-1}(theta)`
- discrepancy posterior particles from `q_{e,t-1}(u,ell)`

For one pair `(theta_i, u_m, ell_c)` define

\[
Y_t \mid X_t,\theta_i,u_m,\ell_c
\sim
\mathcal N\!\Big(
\mu_s(X_t,\theta_i) + A_{e,t}^{(\ell_c)}u_m,\;
\Sigma_s(X_t,\theta_i) + S_{\delta,e,t}^{(\ell_c)} + \sigma^2 I
\Big).
\]

Then the expert predictive density is

\[
p_{\mathrm{BOCPD}}(Y_t \mid X_t,e)
\approx
\sum_{i=1}^N w_{e,t-1}^{(i)}
\sum_{m=1}^{M}\sum_{c=1}^{C}
\alpha_{e,t-1}^{(m,c)}
\,
p(Y_t \mid X_t,\theta_i,u_m,\ell_c).
\]

If `C = 1`, remove the sum over `c`.

The BOCPD loss is

\[
\ell_{e,t}
=
-\log p_{\mathrm{BOCPD}}(Y_t \mid X_t,e).
\]

This loss is then passed into the existing R-BOCPD update and restart logic.

### 2.4 BOCPD and restart

**Do not replace BOCPD.** Reuse the existing R-BOCPD logic exactly as in the current framework.

The sequence is:

1. compute pre-update expert loss `ell_{e,t}`
2. update expert masses / run lengths
3. apply restart rule
4. prune to max experts while keeping the anchor
5. only then update PF and discrepancy states for surviving experts

### 2.5 PF update for theta

After BOCPD restart / pruning, for each surviving expert update `theta` with the **discrepancy-free PF likelihood** only.

This must not use `u` or `ell`.

### 2.6 Discrepancy posterior update after PF update

After the PF step, form the updated mixture anchor

\[
\bar\mu_{e,t}(X_t)
=
\sum_{i=1}^{N} w_{e,t}^{(i)}\,\mu_s(X_t,\theta_{e,t}^{(i)}).
\]

Define the batch residual

\[
r_{e,t}
=
Y_t - \bar\mu_{e,t}(X_t).
\]

Then update the shared discrepancy posterior particles by

\[
\widetilde\alpha_{e,t}^{(m,c)}
=
\alpha_{e,t-1}^{(m,c)}\,
p(r_{e,t}\mid X_t, u_m,\ell_c),
\]

\[
\alpha_{e,t}^{(m,c)}
=
\frac{
\widetilde\alpha_{e,t}^{(m,c)}
}{
\sum_{m',c'} \widetilde\alpha_{e,t}^{(m',c')}
}.
\]

Here

\[
r_{e,t} \mid u_m,\ell_c
\sim
\mathcal N\!\Big(
A_{e,t}^{(\ell_c)}u_m,\;
S_{\delta,e,t}^{(\ell_c)} + \sigma_\delta^2 I
\Big).
\]

In the simplest implementation, set `sigma_delta^2 = sigma_obs^2` or use the same effective observation noise already used for discrepancy fitting.

### 2.7 ESS and resampling for discrepancy particles

As with standard SMC, the discrepancy particle weights will degenerate.

Compute

\[
\mathrm{ESS}_{e,t}
=
\frac{1}{\sum_{m,c} (\alpha_{e,t}^{(m,c)})^2}.
\]

If `ESS < threshold`:

1. resample discrepancy particles,
2. reset weights to equal,
3. optionally perform a rejuvenation move on `u`.

### 2.8 Rejuvenation move (optional but recommended)

After resampling, to avoid collapse, propose

\[
u' = u + \xi,
\qquad
\xi \sim \mathcal N(0,\tau_u^2 I),
\]

and accept / reject with a local MH rule based on the discrepancy likelihood under the current residual batch or short recent buffer.

This can be postponed in the first implementation. A first version can do:

- weight update
- ESS check
- resample only
- no rejuvenation

but comments should note that rejuvenation is expected later.

### 2.9 Hyperparameters `ell`

Recommended implementation stages:

#### Stage 1 (recommended)
- `C = 1`
- fit `ell_e` once at segment initialization or at restart
- keep `ell_e` fixed within that segment
- only particle-approximate `u`

#### Stage 2
- optional small discrete hyperparameter bank, e.g. `C = 3` or `5`
- each `ell_c` is a candidate hyperparameter configuration
- maintain weights jointly over `(u_m, ell_c)`

Do **not** start with large `C`. It will be too expensive and too unstable.

### 2.10 Recommended defaults

For a first implementation:

- `J = 16` or `24` inducing points
- `M = 8` or `16` discrepancy particles
- `C = 1`
- inducing points chosen from recent / segment data by deterministic subsampling or k-means
- restart reinitializes discrepancy posterior from scratch

---

## 3. Particle-specific MC-inducing discrepancy posterior

This is an ablation. It is intentionally more flexible and more likely to hurt identifiability.

### 3.1 Representation

For each expert `e` and each `theta` particle `i`, maintain its own discrepancy posterior

\[
q_{e,t}^{(i)}(u,\ell)
=
\sum_{m,c}
\alpha_{e,t}^{(i,m,c)}
\delta_{u_{e,t}^{(i,m)}}(u)
\delta_{\ell_c}(\ell).
\]

If `C = 1`:

\[
q_{e,t}^{(i)}(u)
=
\sum_{m}
\alpha_{e,t}^{(i,m)}
\delta_{u_{e,t}^{(i,m)}}(u).
\]

### 3.2 Particle-specific residual target

After the PF update, for particle `i` define

\[
r_{e,t}^{(i)}
=
Y_t - \mu_s(X_t,\theta_{e,t}^{(i)}).
\]

Then update that particle's discrepancy posterior using

\[
\widetilde\alpha_{e,t}^{(i,m,c)}
=
\alpha_{e,t-1}^{(i,m,c)}\,
p(r_{e,t}^{(i)} \mid X_t,u_{i,m},\ell_c),
\]

followed by normalization.

### 3.3 Expert scoring

For BOCPD scoring, the expert predictive density becomes

\[
p_{\mathrm{BOCPD}}(Y_t \mid X_t,e)
\approx
\sum_{i=1}^{N} w_{e,t-1}^{(i)}
\sum_{m,c}
\alpha_{e,t-1}^{(i,m,c)}
\,
p(Y_t\mid X_t,\theta_i,u_{i,m},\ell_c).
\]

This is much more flexible than the shared version and is expected to blur parameter discrimination.

### 3.4 Why this is an ablation, not the default

This version is much closer to particle-specific discrepancy, which the current paper argues is harmful for identifiability and for BOCPD evidential separation. It should therefore be implemented as an **ablation only**.

### 3.5 Recommended defaults for this ablation

Because it is expensive:

- start with `J = 8` or `12`
- `M = 4` or `8`
- `C = 1`
- only run on smaller studies first

---

## 4. Residual-history issue and two update modes

This part is crucial.

### 4.1 Current half-discrepancy refit behavior

The current main method effectively rebuilds discrepancy from a **current-endpoint anchor**, meaning historical residual targets are recomputed using the current particle-mixture anchor.

This is **not** the same as a pure online append-only posterior update.

### 4.2 Pure online mode

In a pure online discrepancy-posterior update, past discrepancy pseudo-observations are effectively frozen. New data update the discrepancy posterior incrementally, but old targets are not rewritten.

### 4.3 Hybrid refresh mode (recommended optional feature)

To better approximate the current refit design while still being online, add an optional **refresh / rebuild** mechanism:

- every `R` batches, or
- whenever restart occurs, or
- whenever a separate refresh rule triggers,

rebuild the discrepancy posterior from a refreshed residual buffer.

This hybrid mode is recommended because it provides a middle ground between:
- expensive full refit every batch
- pure frozen-target online update

### 4.4 Required code flags

Add a discrepancy update policy enum, e.g.

- `refit`
- `online_shared_mc_inducing`
- `online_particle_mc_inducing`
- `online_shared_mc_inducing_refresh`
- `online_particle_mc_inducing_refresh`

---

## 5. Batch-update order for DOC + MC-inducing discrepancy

The batch update order must remain:

### Step A. Pre-update BOCPD scoring
For each current expert `e`, compute expert predictive density and BOCPD loss using:
- pre-update `theta` particles,
- pre-update discrepancy posterior particles.

### Step B. BOCPD update / restart / pruning
Reuse the current R-BOCPD mass update and restart rule.

### Step C. PF update
For each surviving expert, run discrepancy-free PF update.

### Step D. Discrepancy posterior update
For each surviving expert, update its discrepancy posterior particles (`shared` or `particle-specific`) using the new batch residual.

This order should be reflected in code and comments.

---

## 6. Digital-twin PF style method (Ward-style PF)

This is a separate method family. It does **not** follow the half-discrepancy principle.

Implement it as a separate mode.

### 6.1 Observation model

Use the Ward-style model

\[
y = \rho \eta + \delta + \varepsilon.
\]

For one particle `j`, maintain calibration parameters `theta_j`, and optionally additional hyperparameters such as GP lengthscale `l_j` and scaling `rho_j`.

### 6.2 Per-particle kernel matrix

For particle `j`, build the GP covariance matrix

\[
K^{(j)}
=
\begin{bmatrix}
k_\eta(X_d,X_d,t,t \mid l_j)
&
\rho_j\,k_\eta(X_d,X_Y,t,\theta_j \mid l_j)
\\[4pt]
\rho_j\,k_\eta(X_Y,X_d,\theta_j,t \mid l_j)
&
\rho_j^2 k_\eta(X_Y,X_Y,\theta_j,\theta_j \mid l_j)
+
k_Y(X_Y,X_Y,\theta_j,\theta_j \mid l_j)
+
\sigma^2 I
\end{bmatrix}.
\]

Then compute the GP marginal likelihood

\[
\xi_j = p(d_i, Y_i \mid 0, K^{(j)}).
\]

Use this to update particle weights.

### 6.3 Important note on data usage

This Ward-style PF is **sequential**, meaning the current step uses the current data / current batch and updates weights sequentially. It is **not** maintaining one ever-growing giant kernel matrix over the entire history.

History enters through the evolving particle weights / resampling, not through one cumulative covariance matrix.

### 6.4 Ward-style PF algorithm

For each batch or time step:

1. compute per-particle marginal likelihood `xi_j`
2. normalize weights
3. resample particles
4. propagate / jitter particles if desired

### 6.5 Optional BOCPD wrapper for Ward-style PF

Also implement an optional BOCPD wrapper around this PF:

- for each BOCPD expert, maintain a Ward-style PF state
- pre-update expert scoring uses the expert's current Ward-style PF predictive likelihood
- BOCPD restart logic remains the same
- after restart / pruning, update the surviving experts' PF states

Code should support both:

- `ward_pf_no_bocpd`
- `ward_pf_with_bocpd`

### 6.6 Why this method is separate

Ward-style PF puts discrepancy directly into the particle likelihood. This does **not** preserve the half-discrepancy identifiability structure. It is a different method family and should be implemented separately for comparison.

---

## 7. Recommended class structure

Below is a recommended class layout for Codex to implement.

### 7.1 Shared discrepancy posterior particles

```python
class SharedInducingDiscrepancyState:
    Z: Tensor              # [J, dx]
    u_particles: Tensor    # [M, J]
    weights: Tensor        # [M]
    hyper_specs: list      # optional, length C or singleton
    # optional cached matrices / Cholesky factors
```

Methods:

- `predict_batch(X_batch) -> (mu_delta_samples, var_delta_samples)`
- `update_weights_from_residual(resid_batch, X_batch)`
- `ess()`
- `resample()`
- `rejuvenate()`
- `refresh_from_history(X_hist, Y_hist, anchor_mean_fn)`

### 7.2 Particle-specific discrepancy posterior particles

```python
class ParticleSpecificInducingDiscrepancyState:
    Z: Tensor                    # [J, dx] or per-particle Z if needed
    u_particles: Tensor         # [N, M, J]
    weights: Tensor             # [N, M]
    hyper_specs: list           # optional
```

Methods similar to shared, but indexed by `theta` particle.

### 7.3 Expert structure

Existing `Expert` should keep:

- PF state
- discrepancy state
- histories
- run length
- log mass

No change to BOCPD interface if possible.

### 7.4 Configuration flags

Add config fields like:

```python
discrepancy_mode:
    - "shared_gp_refit"
    - "shared_mc_inducing"
    - "shared_mc_inducing_refresh"
    - "particle_mc_inducing"
    - "particle_mc_inducing_refresh"
    - "ward_pf"

inducing_num_points: int
mc_num_particles: int
mc_num_hyper_candidates: int
mc_resample_ess_ratio: float
mc_rejuvenate: bool
mc_rejuvenate_scale: float
mc_refresh_every: int
```

For Ward-style PF:

```python
ward_use_bocpd: bool
ward_sample_rho: bool
ward_sample_lengthscale: bool
ward_num_particles: int
```

---

## 8. Computational complexity guidance

### 8.1 Shared expert-level MC-inducing
This is the recommended default.

Rough cost per expert per batch:

- BOCPD scoring: `O(N * M * K_batch * J)` if implemented naively
- discrepancy update: `O(M * K_batch * J)`

This is usually affordable for:
- `N = 1000`
- `M = 8 or 16`
- `J = 16 or 24`
- small batch size such as `20`

### 8.2 Particle-specific MC-inducing
Much more expensive:

- BOCPD scoring: `O(N * M * K_batch * J)`
- discrepancy update: `O(N * M * K_batch * J)`

But memory is much larger because discrepancy particles are stored per theta particle.

This should be treated as an ablation only.

---

## 9. Concrete implementation stages

### Stage 1: shared expert-level MC-inducing, fixed hyperparameters
Implement:
- `C = 1`
- `shared` discrepancy particles only
- no rejuvenation initially
- resampling only when ESS is small
- refresh only on BOCPD restart

### Stage 2: shared expert-level MC-inducing with periodic refresh
Add:
- optional refresh every `R` batches
- optional rebuild from a recent window or full segment history

### Stage 3: particle-specific MC-inducing ablation
Implement only after Stage 1 works.

### Stage 4: Ward-style PF with and without BOCPD
Implement as separate method family.

---

## 10. Code-level batch pseudocode for the shared expert-level version

```text
for each incoming batch (X_t, Y_t):

    # A. pre-update scoring
    for each current expert e:
        use pre-update theta particles and pre-update discrepancy particles
        compute p_BOCPD(Y_t | X_t, e)
        define loss ell_{e,t} = -log p_BOCPD(...)

    # B. BOCPD update
    update expert masses using existing R-BOCPD equations
    apply restart rule
    prune experts if needed

    # C. PF update on surviving experts
    for each surviving expert e:
        propagate theta particles
        update theta weights using discrepancy-free PF likelihood
        resample/rejuvenate PF if needed

    # D. discrepancy posterior update
    for each surviving expert e:
        compute updated theta-mixture anchor
        form batch residual r_{e,t} = Y_t - mu_bar_{e,t}(X_t)
        update discrepancy particle weights with p(r_{e,t} | u_m)
        if ESS is low:
            resample discrepancy particles
            optionally rejuvenate

        if refresh policy triggers:
            rebuild discrepancy posterior from refreshed residual history
```

---

## 11. Code-level batch pseudocode for Ward-style PF

```text
for each incoming batch (X_t, Y_t):

    for each PF particle j:
        compute per-particle GP marginal likelihood xi_j
        update PF particle weight

    normalize weights
    resample if needed
    propagate / jitter PF particles if desired

    if using BOCPD:
        wrap the above per-expert, and let BOCPD handle
        expert scoring / restart / pruning first
```

---

## 12. What to log for experiments

For each method, log at least:

- `theta_rmse`
- `theta_crps`
- `y_rmse`
- `y_crps`
- expert count over time
- restart counts by type
- discrepancy ESS over time
- if using hyperparameter particles: posterior mass over `ell_c`
- average BOCPD expert entropy
- runtime per batch

For MC-inducing methods also log:

- mean discrepancy ESS
- number of discrepancy resampling events
- number of discrepancy refresh events
- average inducing posterior variance if available

---

## 13. Naming conventions for experiments

Recommended method names:

- `DOC` = current shared-GP refit main method
- `DOC-SharedMCInducing`
- `DOC-SharedMCInducing-Refresh`
- `DOC-ParticleMCInducing`
- `DOC-ParticleMCInducing-Refresh`
- `WardPF`
- `WardPF-BOCPD`

If you need to preserve old names for compatibility, provide aliases.

---

## 14. Final design recommendations

### Recommended default to implement first
**Shared expert-level MC-inducing discrepancy posterior with fixed hyperparameters within segment.**

Why:
- closest to the current half-discrepancy philosophy,
- does not reintroduce particle-specific discrepancy,
- computation is manageable,
- easy to compare against current `shared_gp` refit.

### Particle-specific version
Implement only as an ablation.

### Ward-style PF
Implement as a separate comparison baseline, not as a modification of DOC.

---

## 15. Summary in plain language

- Keep BOCPD and restart exactly as in the current DOC framework.
- Keep the PF update for `theta` discrepancy-free.
- Add a new online discrepancy posterior approximation based on inducing variables.
- Prefer a **shared expert-level** discrepancy posterior first.
- Implement the **particle-specific** version only as an ablation.
- Implement the **digital-twin PF / Ward-style PF** as a separate method family, optionally with BOCPD on top.
- Make the code explicit that these online discrepancy methods are approximations to the current refit design, not identical replacements.
