# Online Approximated Regime-Aware Bayesian Projected Calibration
## Mathematical note and implementation spec

This document gives a complete mathematical and algorithmic description of a new viewpoint on the current method family:

- current method: `R-BOCPD-PF-particleGP-halfdiscrepancy`
- new viewpoint: **online approximated regime-aware Bayesian projected calibration (online BPC)**

The document has four goals:

1. explain the current method in a unified way;
2. reinterpret it as an **online, regime-aware approximation to Bayesian projected calibration**;
3. derive the key mathematical update for the per-particle discrepancy posterior;
4. specify exactly what must change in the current implementation to obtain:
   - a **particle-specific online BPC** version, and
   - a **shared online BPC** version.

This document is intended to be concrete enough for Codex to directly generate code from it.

---

# 1. High-level picture

The core claim is:

> The current method can be reinterpreted as a **regime-aware sequential particle approximation to online Bayesian projected calibration**.

The main ingredients are:

- a sequential posterior approximation for the calibration parameter `theta` using a particle filter;
- a discrepancy posterior update conditioned on the current `theta` state;
- a BOCPD / R-BOCPD outer layer that decides when the current local problem is no longer valid and should be restarted.

This reframing is useful because it gives a single coherent story for:

- why `theta` is updated using a discrepancy-free likelihood,
- why discrepancy is updated conditionally after `theta`,
- why BOCPD should score the full predictive model `y_s + delta`,
- and how to replace the current discrepancy refit by a proper sequential posterior update.

---

# 2. Problem setup

We observe streaming mini-batches

\[
\mathcal{B}_t = \{(x_{t,k}, y_{t,k})\}_{k=1}^{K_t}, \qquad t=1,2,\dots
\]

with observation model

\[
y_{t,k} = y_s(x_{t,k}, \theta_t^\star) + \delta_t^\star(x_{t,k}) + \varepsilon_{t,k},
\qquad
\varepsilon_{t,k} \sim \mathcal N(0,\sigma^2).
\]

Here:

- \(y_s(\cdot,\theta)\) is the simulator or emulator,
- \(\theta_t^\star\) is the latent calibration parameter,
- \(\delta_t^\star\) is model discrepancy.

We want an online method that:

1. tracks \(\theta_t^\star\) over time,
2. preserves prediction quality,
3. handles gradual drift and sudden changes,
4. avoids discrepancy–parameter confounding.

---

# 3. Current method as a baseline

The current family `R-BOCPD-PF-particleGP-halfdiscrepancy` has the following structure.

## 3.1 Theta step (PF, discrepancy-free)

For each active expert \(e\), maintain particles

\[
\{(\theta_{e,t}^{(j)}, w_{e,t}^{(j)})\}_{j=1}^N.
\]

Propagate:

\[
\theta_{e,t}^{(j)} \sim p(\theta_t \mid \theta_{e,t-1}^{(j)}).
\]

Reweight using a discrepancy-free pseudo-likelihood:

\[
p_{\mathrm{PF}}(Y_t \mid X_t,\theta_{e,t}^{(j)})
=
\mathcal{N}\!\big(
Y_t;\,
\mu_s(X_t,\theta_{e,t}^{(j)}),
\Sigma_s(X_t,\theta_{e,t}^{(j)}) + \sigma_{\mathrm{PF}}^2 I
\big).
\]

\[
\widetilde w_{e,t}^{(j)}
\propto
w_{e,t-1}^{(j)}\,
p_{\mathrm{PF}}(Y_t \mid X_t,\theta_{e,t}^{(j)}),
\qquad
w_{e,t}^{(j)}
=
\frac{\widetilde w_{e,t}^{(j)}}{\sum_\ell \widetilde w_{e,t}^{(\ell)}}.
\]

This is the current `theta` search channel.

## 3.2 Discrepancy step (current half-discrepancy)

The current method updates discrepancy **after** the PF step, using residuals built from the updated `theta` state.

This discrepancy does **not** enter the PF weighting step. It only enters the expert-level predictive scoring / BOCPD evidence.

Operationally, this is a projected / decoupled update:

- first update `theta`,
- then update discrepancy conditioned on the current `theta` anchor,
- then use `y_s + delta` for BOCPD scoring.

## 3.3 BOCPD / R-BOCPD step

BOCPD sits outside the local updates and decides whether the current segment-local approximation is still valid.

It should score the **full predictive decomposition**

\[
Y_t \approx y_s(X_t,\theta_t) + \delta_t(X_t),
\]

not only the simulator term.

This is because both `theta` and `delta` are local states that may become invalid after a regime change.

---

# 4. Why this is naturally an online BPC view

The key projected-calibration idea is:

> `theta` and `delta` must not compete freely inside the same search channel.

In the current method, this is implemented operationally by:

- updating `theta` using a discrepancy-free likelihood;
- updating `delta` only after `theta`, conditioned on the current `theta` state;
- using the full predictive model only for scoring / changepoint evidence.

That is already a projected-calibration structure.

The proposed new viewpoint is:

> treat this as an **online Bayesian projected calibration problem**, where both `theta` and `delta` are updated by local KL-regularized Bayesian steps, and BOCPD is the regime-validity controller.

This gives a single unified interpretation.

---

# 5. Online BPC: abstract formulation

For one expert, at batch \(t\), define two local posterior objects:

- \(q_t(\theta)\): local posterior approximation for `theta`,
- \(q_t(\delta \mid \cdot)\): local discrepancy posterior update.

## 5.1 Theta step as a KL-regularized local Bayesian update

Define

\[
q_t(\theta)
=
\arg\min_q
\left\{
-\eta_\theta \, \mathbb E_q\!\left[\log p_{\mathrm{PF}}(Y_t \mid X_t,\theta)\right]
+
\mathrm{KL}(q \,\|\, q_{t-1})
\right\}.
\]

Interpretation:

- fit the current batch,
- but stay close to the previous posterior,
- so the update is local / trust-region-like.

The current PF is a **particle approximation** to this local Bayesian step.

## 5.2 Discrepancy step as a KL-regularized conditional Bayesian update

This is the key new derivation.

For each current `theta` particle \( \theta_t^{(j)} \), define its batch residual

\[
r_t^{(j)} := Y_t - \mu_s(X_t,\theta_t^{(j)}).
\]

Now maintain a discrepancy posterior for that particle:

\[
q_t^{(j)}(\delta).
\]

The online BPC discrepancy step is

\[
q_t^{(j)}(\delta)
=
\arg\min_q
\left\{
-\eta_\delta \,\mathbb E_q\!\left[\log p(r_t^{(j)} \mid \delta)\right]
+
\mathrm{KL}\!\left(q \,\|\, q_{t-1}^{(j)}(\delta)\right)
\right\}.
\]

Important:

- the old state is \(q_{t-1}^{(j)}(\delta)\),
- the current `theta` enters only through the new residual \(r_t^{(j)}\),
- the current `theta` is **not** a condition inside the old prior.

This is the clean formulation.

---

# 6. Exact derivation of the discrepancy update

We now solve

\[
q^\star
=
\arg\max_q
\left\{
\mathbb E_q[\log p(r \mid \delta)]
-
\lambda_\delta \,\mathrm{KL}(q \,\|\, q_{\mathrm{old}})
\right\}.
\]

This is equivalent to the minimization above after a sign change.

## 6.1 Expand the KL term

\[
\mathrm{KL}(q \,\|\, q_{\mathrm{old}})
=
\int q(\delta)\log \frac{q(\delta)}{q_{\mathrm{old}}(\delta)}\,d\delta.
\]

So the objective is

\[
\mathcal J(q)
=
\int q(\delta)\log p(r \mid \delta)\,d\delta
-
\lambda_\delta
\int q(\delta)\log \frac{q(\delta)}{q_{\mathrm{old}}(\delta)}\,d\delta.
\]

with the normalization constraint

\[
\int q(\delta)\,d\delta = 1.
\]

## 6.2 Variational first-order condition

Introduce multiplier \(\alpha\):

\[
\mathcal L(q)
=
\int q(\delta)\log p(r \mid \delta)\,d\delta
-
\lambda_\delta
\int q(\delta)\log \frac{q(\delta)}{q_{\mathrm{old}}(\delta)}\,d\delta
+
\alpha\left(\int q(\delta)\,d\delta - 1\right).
\]

Set functional derivative to zero:

\[
\frac{\delta \mathcal L}{\delta q(\delta)}
=
\log p(r \mid \delta)
-
\lambda_\delta
\left(
\log \frac{q(\delta)}{q_{\mathrm{old}}(\delta)} + 1
\right)
+
\alpha
=0.
\]

Rearrange:

\[
\log q(\delta)
=
\log q_{\mathrm{old}}(\delta)
+
\frac{1}{\lambda_\delta}\log p(r \mid \delta)
+
\text{const}.
\]

Therefore

\[
q^\star(\delta)
\propto
q_{\mathrm{old}}(\delta)\,
p(r \mid \delta)^{1/\lambda_\delta}.
\]

This is the exact solution.

## 6.3 Interpretation

The KL-regularized discrepancy step is exactly a **tempered Bayes posterior update**:

\[
q_t^{(j)}(\delta)
\propto
q_{t-1}^{(j)}(\delta)\,
p(r_t^{(j)} \mid \delta)^{1/\lambda_\delta}.
\]

Special cases:

- \(\lambda_\delta = 1\): standard Bayes update;
- \(\lambda_\delta > 1\): conservative / trust-region update;
- \(\lambda_\delta < 1\): aggressive update.

This is the central mathematical formula.

---

# 7. If discrepancy is a Gaussian process

Assume

\[
q_{t-1}^{(j)}(\delta)
=
\mathcal{GP}\!\left(m_{t-1}^{(j)},\,C_{t-1}^{(j)}\right).
\]

Assume the residual likelihood is Gaussian:

\[
r_t^{(j)} \mid \delta
\sim
\mathcal N\!\big(\delta(X_t),\,\Sigma_r\big).
\]

Then

\[
p(r_t^{(j)} \mid \delta)^{1/\lambda_\delta}
\propto
\exp\!\left(
-\frac{1}{2\lambda_\delta}
(r_t^{(j)}-\delta(X_t))^\top \Sigma_r^{-1}(r_t^{(j)}-\delta(X_t))
\right),
\]

which is equivalent to a Gaussian likelihood with inflated noise:

\[
r_t^{(j)} \mid \delta
\sim
\mathcal N\!\big(\delta(X_t),\,\lambda_\delta \Sigma_r\big).
\]

Therefore the posterior is still a GP:

\[
q_t^{(j)}(\delta)
=
\mathcal{GP}\!\left(m_t^{(j)},\,C_t^{(j)}\right).
\]

The update formulas are standard GP posterior formulas with noise \(\lambda_\delta \Sigma_r\).

Define

\[
K_{t-1}^{(j)} := C_{t-1}^{(j)}(X_t,X_t),
\qquad
k_{t-1}^{(j)}(x) := C_{t-1}^{(j)}(x,X_t).
\]

Then

\[
m_t^{(j)}(x)
=
m_{t-1}^{(j)}(x)
+
k_{t-1}^{(j)}(x)
\Big(
K_{t-1}^{(j)}+\lambda_\delta \Sigma_r
\Big)^{-1}
\Big(
r_t^{(j)}-m_{t-1}^{(j)}(X_t)
\Big),
\]

\[
C_t^{(j)}(x,x')
=
C_{t-1}^{(j)}(x,x')
-
k_{t-1}^{(j)}(x)
\Big(
K_{t-1}^{(j)}+\lambda_\delta \Sigma_r
\Big)^{-1}
k_{t-1}^{(j)}(x')^\top.
\]

This is the exact GP form of the discrepancy update.

---

# 8. Inducing-variable form (recommended for implementation)

The current code family already has a particle-GP flavor, so inducing variables are the natural practical state.

Let \(Z\) be inducing points and define

\[
u := \delta(Z)\in\mathbb R^J.
\]

Approximate

\[
\delta(X_t) \approx A_t u,
\qquad
A_t := K_{X_t Z}K_{ZZ}^{-1}.
\]

Maintain, for each theta particle \(j\),

\[
q_{t-1}^{(j)}(u)
=
\mathcal N\!\left(m_{u,t-1}^{(j)},\,S_{u,t-1}^{(j)}\right).
\]

Then the residual model is

\[
r_t^{(j)} \mid u
\sim
\mathcal N(A_t u,\Sigma_r).
\]

The KL-regularized update gives

\[
q_t^{(j)}(u)
\propto
q_{t-1}^{(j)}(u)\,
\mathcal N(r_t^{(j)}; A_t u,\Sigma_r)^{1/\lambda_\delta},
\]

which is Gaussian with

\[
S_{u,t}^{(j)\,-1}
=
S_{u,t-1}^{(j)\,-1}
+
\frac{1}{\lambda_\delta}A_t^\top \Sigma_r^{-1} A_t,
\]

\[
m_{u,t}^{(j)}
=
S_{u,t}^{(j)}
\left[
S_{u,t-1}^{(j)\,-1}m_{u,t-1}^{(j)}
+
\frac{1}{\lambda_\delta}A_t^\top \Sigma_r^{-1}r_t^{(j)}
\right].
\]

This is likely the best practical state representation.

---

# 9. Joint posterior interpretation

With theta particles and per-particle discrepancy posteriors, the joint posterior approximation for one expert is

\[
q_t(\theta,\delta)
\approx
\sum_{j=1}^N
w_t^{(j)}
\,
\delta_{\theta_t^{(j)}}(\theta)
\,
q_t^{(j)}(\delta).
\]

This is not yet a set of point samples \((\theta_j,\delta_j)\). It is:

- particle approximation in `theta`,
- conditional posterior family in `delta`.

If one additionally samples \(M\) discrepancy draws from each \(q_t^{(j)}(\delta)\), then one gets joint point particles

\[
q_t(\theta,\delta)
\approx
\sum_{j=1}^N \sum_{m=1}^M
w_t^{(j)} \alpha_t^{(j,m)}
\delta_{(\theta_t^{(j)}, \delta_t^{(j,m)})}.
\]

This is where a true particle representation of the full joint posterior appears.

---

# 10. Why BOCPD should score the full predictive model

The online BPC view implies that one local expert carries two local states:

- \(q_t(\theta)\),
- \(q_t(\delta)\).

Both are trust-region / local Bayesian objects.

Therefore the BOCPD validity test should be based on the full predictive decomposition:

\[
Y_t \approx y_s(X_t,\theta_t) + \delta_t(X_t),
\]

not just the simulator term \(y_s\).

Why:

- the `theta` local update may fail after a jump;
- the `delta` local posterior may also become stale after a jump;
- BOCPD should therefore test whether the **entire local predictive decomposition** is still coherent.

This is the conceptual reason that BOCPD should use discrepancy-aware expert predictive likelihood, not simulator-only likelihood.

---

# 11. Regime-aware online BPC

With BOCPD included, the full interpretation is:

1. **within a regime**
   - update `theta` by a local KL-regularized Bayesian step, approximated by PF;
   - update discrepancy by a local KL-regularized conditional GP posterior step;
2. **across regimes**
   - BOCPD tests whether that local online-BPC problem is still valid;
   - if not, restart the local learner and truncate stale discrepancy memory.

This yields the name:

> **online approximated regime-aware Bayesian projected calibration**

---

# 12. What must change in the current `R-BOCPD-PF-particleGP-halfdiscrepancy`

This is the implementation-critical section.

Current method:
- has PF over `theta`,
- has BOCPD over experts,
- has discrepancy correction after PF,
- but discrepancy is not maintained as a proper per-particle posterior state.

The main required modification is:

> each theta particle must now carry its own discrepancy posterior state.

## 12.1 Current expert state

Current expert approximately stores:
- theta particles and weights,
- BOCPD log mass,
- histories,
- some discrepancy state used for prediction / scoring.

## 12.2 New particle-specific online BPC expert state

For each expert `e`, store:

- theta particles:
  - `theta_particles[j]`
  - `theta_weights[j]`
- for each theta particle `j`, a discrepancy posterior state:
  - if full GP:
    - GP posterior mean and covariance representation;
  - if inducing-variable GP:
    - `m_u[j]`
    - `S_u[j]`
    - inducing points `Z[j]` or shared `Z`.

Thus expert state becomes conceptually:

\[
\left\{
\big(
\theta_t^{(j)},\, w_t^{(j)},\, q_t^{(j)}(\delta)
\big)
\right\}_{j=1}^N.
\]

This is the main structural change.

---

# 13. New particle-specific batch update order

For one expert and one incoming batch \((X_t,Y_t)\):

## Step A. Pre-update BOCPD scoring

Before updating either theta or delta for batch \(t\), compute the expert predictive likelihood using the **pre-update** states:

\[
q_{t-1}(\theta,\delta)
\approx
\sum_{j=1}^N w_{t-1}^{(j)}\,\delta_{\theta_{t-1}^{(j)}}(\theta)\,q_{t-1}^{(j)}(\delta).
\]

The predictive density is

\[
p_{\mathrm{BOCPD}}(Y_t \mid X_t,e)
=
\sum_{j=1}^N
w_{t-1}^{(j)}
\int
p(Y_t \mid X_t,\theta_{t-1}^{(j)},\delta)\,
q_{t-1}^{(j)}(\delta)\,d\delta.
\]

If \(q_{t-1}^{(j)}(\delta)\) is Gaussian, this integral may be analytic or approximated by Gaussian moment matching.

Use this predictive density for:
- BOCPD loss,
- mass update,
- restart decision.

## Step B. BOCPD update / restart / pruning

Keep the existing R-BOCPD logic.

## Step C. Theta PF update

For surviving experts only, propagate and reweight theta particles using the discrepancy-free PF likelihood, exactly as now.

This step is unchanged.

## Step D. Discrepancy posterior update for each theta particle

For each particle \(j\):

1. compute
   \[
   r_t^{(j)} = Y_t - \mu_s(X_t,\theta_t^{(j)});
   \]
2. update
   \[
   q_t^{(j)}(\delta)
   \propto
   q_{t-1}^{(j)}(\delta)\,
   p(r_t^{(j)} \mid \delta)^{1/\lambda_\delta}.
   \]

If using inducing variables \(u\), update \(m_u^{(j)}, S_u^{(j)}\) using the formulas in Section 8.

This is the central modification.

---

# 14. Shared version

The shared version is the computationally lighter relaxation and should be implemented as a second variant.

## 14.1 Shared discrepancy posterior

Instead of per-particle discrepancy states \(q_t^{(j)}(\delta)\), maintain one shared discrepancy posterior for the whole expert:

\[
q_t(\delta).
\]

Then after the PF update, define a shared theta anchor, for example the particle-mixture mean prediction

\[
\bar\mu_t(X_t)
=
\sum_{j=1}^N w_t^{(j)}\,\mu_s(X_t,\theta_t^{(j)}).
\]

Define the shared residual

\[
r_t := Y_t - \bar\mu_t(X_t).
\]

Update the shared discrepancy posterior by

\[
q_t(\delta)
=
\arg\min_q
\left\{
-\eta_\delta\,\mathbb E_q[\log p(r_t \mid \delta)]
+
\mathrm{KL}(q \,\|\, q_{t-1}(\delta))
\right\},
\]

equivalently,

\[
q_t(\delta)
\propto
q_{t-1}(\delta)\,
p(r_t \mid \delta)^{1/\lambda_\delta}.
\]

This version is cleaner computationally and is the natural sequential generalization of the current shared half-discrepancy spirit.

## 14.2 Shared predictive density for BOCPD

The expert predictive law becomes

\[
p_{\mathrm{BOCPD}}(Y_t \mid X_t,e)
=
\sum_{j=1}^N
w_{t-1}^{(j)}
\int
p(Y_t \mid X_t,\theta_{t-1}^{(j)},\delta)\,
q_{t-1}(\delta)\,d\delta.
\]

Compared with the particle-specific version, this is less flexible and should preserve identifiability better.

---

# 15. Relationship between the new variants and the current method

## 15.1 Current half-discrepancy
- theta updated without discrepancy;
- discrepancy updated after theta;
- discrepancy used in BOCPD scoring;
- no proper sequential posterior state per theta particle.

## 15.2 New particle-specific online BPC
- theta updated without discrepancy;
- each theta particle has its own discrepancy posterior state;
- discrepancy posterior updated by KL-regularized GP Bayes update;
- expert predictive density integrates over each particle's discrepancy posterior.

## 15.3 New shared online BPC
- theta updated without discrepancy;
- one shared discrepancy posterior per expert;
- discrepancy posterior updated by KL-regularized GP Bayes update using shared residual;
- expert predictive density integrates over the shared discrepancy posterior.

---

# 16. Recommended experimental order

## Stage 1
Implement the **shared** version first.

Reasons:
- much lighter computationally,
- easier to debug,
- naturally comparable with current half-discrepancy,
- clearer BOCPD behavior.

## Stage 2
Implement the **particle-specific** version.

Reasons:
- theoretically cleaner as conditional online BPC,
- closer to the most principled derivation,
- but more expensive and likely more fragile.

## Stage 3
Compare all three:
- current half-discrepancy,
- shared online BPC,
- particle-specific online BPC.

Primary metrics:
- theta RMSE / CRPS,
- y RMSE / CRPS,
- BOCPD restart behavior,
- runtime,
- expert entropy / sharpness.

---

# 17. Practical implementation notes for Codex

## 17.1 New classes

Suggested new classes:

```python
class ParticleDeltaPosteriorState:
    # for one theta particle
    gp_mode: str  # "full_gp" or "inducing_gp"
    # if inducing:
    Z: Tensor
    mean_u: Tensor
    cov_u: Tensor
    noise_scale: float
    lambda_delta: float
```

```python
class SharedDeltaPosteriorState:
    gp_mode: str
    Z: Tensor
    mean_u: Tensor
    cov_u: Tensor
    noise_scale: float
    lambda_delta: float
```

## 17.2 New methods

For particle-specific version:

```python
update_particle_delta_posterior(
    delta_state_j,
    X_batch,
    Y_batch,
    theta_particle_j,
    simulator_fn,
)
```

This method should:

1. compute \(r_t^{(j)}\),
2. update the GP posterior using the tempered-noise formulas.

For shared version:

```python
update_shared_delta_posterior(
    shared_delta_state,
    X_batch,
    Y_batch,
    theta_particles,
    theta_weights,
    simulator_fn,
)
```

This method should:

1. compute shared anchor \(\bar\mu_t\),
2. compute shared residual \(r_t\),
3. update the GP posterior.

## 17.3 New BOCPD predictive routines

Particle-specific version:

```python
expert_predictive_loglik_particle_specific(
    expert,
    X_batch,
    Y_batch,
)
```

Shared version:

```python
expert_predictive_loglik_shared(
    expert,
    X_batch,
    Y_batch,
)
```

Both should evaluate the pre-update predictive density.

---

# 18. Minimal pseudocode

## 18.1 Particle-specific online BPC

```text
for each batch t:
    # pre-update scoring
    for each expert e:
        compute p_BOCPD(Y_t | X_t, e)
        using {theta_{t-1}^{(j)}, w_{t-1}^{(j)}, q_{t-1}^{(j)}(delta)}
    update BOCPD masses
    restart / prune

    for each surviving expert e:
        # theta step
        propagate theta particles
        update theta weights with discrepancy-free PF likelihood
        resample if needed

        # delta step
        for each theta particle j:
            r_t^{(j)} = Y_t - mu_s(X_t, theta_t^{(j)})
            update q_t^{(j)}(delta)
            using q_t^{(j)}(delta) ∝ q_{t-1}^{(j)}(delta) p(r_t^{(j)}|delta)^{1/lambda_delta}
```

## 18.2 Shared online BPC

```text
for each batch t:
    # pre-update scoring
    for each expert e:
        compute p_BOCPD(Y_t | X_t, e)
        using {theta_{t-1}^{(j)}, w_{t-1}^{(j)}, q_{t-1}(delta)}
    update BOCPD masses
    restart / prune

    for each surviving expert e:
        # theta step
        propagate theta particles
        update theta weights with discrepancy-free PF likelihood
        resample if needed

        # shared delta step
        mu_bar_t = sum_j w_t^{(j)} mu_s(X_t, theta_t^{(j)})
        r_t = Y_t - mu_bar_t
        update q_t(delta)
        using q_t(delta) ∝ q_{t-1}(delta) p(r_t|delta)^{1/lambda_delta}
```

---

# 19. Final interpretation

The cleanest interpretation is:

> The method is an online, regime-aware particle approximation to Bayesian projected calibration.

- PF approximates the local KL-regularized Bayesian update of the projected calibration parameter target;
- discrepancy is updated as a conditional tempered GP posterior given the current residual;
- BOCPD evaluates whether the resulting local predictive decomposition is still valid.

This is the full mathematical story.

---

# 20. Implemented exact batch-support variant

The implementation added in the current codebase follows the exact online-BPC update in Section 7,
but uses a lightweight *batch-support posterior representation* rather than storing the full history.

For the shared version, the state at time `t` stores:

- the most recent support inputs `X_t`,
- the posterior mean `m_t` of `delta(X_t)`,
- the posterior covariance `S_t` of `delta(X_t)`.

To move from batch `t` to batch `t+1`, the code computes the GP-prior propagation
from the old support `X_t` to the new support `X_{t+1}`:

- `A = K(X_{t+1}, X_t) K(X_t, X_t)^{-1}`
- `m^-_{t+1} = A m_t`
- `S^-_{t+1} = K(X_{t+1}, X_{t+1}) - A K(X_t, X_{t+1}) + A S_t A^T`

Then it applies the tempered Bayes update using observation noise
`lambda_delta * sigma_delta^2 I`:

- `m_{t+1} = m^-_{t+1} + S^-_{t+1} (S^-_{t+1} + lambda_delta sigma_delta^2 I)^{-1} (r_{t+1} - m^-_{t+1})`
- `S_{t+1} = S^-_{t+1} - S^-_{t+1} (S^-_{t+1} + lambda_delta sigma_delta^2 I)^{-1} S^-_{t+1}`

This keeps the update cost at the current batch size rather than the entire retained history.

For the particle-specific version, all lineages share the same support inputs and the same support
covariance, while each lineage keeps its own support posterior mean vector. After PF resampling,
children copy the parent support posterior mean and then assimilate their own new residual batch.
This preserves the lineage-bound semantics while keeping the covariance update shared.

