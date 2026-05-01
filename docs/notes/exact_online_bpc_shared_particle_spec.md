# Exact Online-BPC Spec
## Regime-aware online Bayesian projected calibration with expanding-support GP posteriors
## For modifying `R-BOCPD-PF-particleGP-halfdiscrepancy`

This document specifies an **exact / theory-consistent** implementation target for the online-BPC idea.

It is written for Codex and focuses on:

1. the exact mathematical object we want to represent,
2. how this differs from the current compressed-state implementation,
3. a **shared** version and a **particle-specific** version,
4. what must change in the current `R-BOCPD-PF-particleGP-halfdiscrepancy` code.

Immediate recommendation:

> **Implement the shared exact online-BPC version first**, run it, and compare it against current `halfdiscrepancy`.
> The particle-specific version is also specified below, but it will be much more expensive.

---

# 0. Executive summary

We want to stop using the current discrepancy update that compresses the discrepancy state onto only a very small support (often only the previous batch support).

Instead, we want a theory-consistent online-BPC recursion:

- maintain a GP posterior state on an **expanding support**
- update that posterior using a **tempered batch likelihood**
- use the resulting posterior to compute the full predictive likelihood for BOCPD

The most important design change is:

> **after updating with batch `t`, keep the posterior on the full expanded support**
> \[
> S_t = S_{t-1}\cup X_t,
> \]
> rather than compressing it back to only the current batch support \(X_t\).

This is the main difference from the existing implementation.

---

# 1. Problem setup

We observe streaming mini-batches

\[
\mathcal B_t = \{(x_{t,k}, y_{t,k})\}_{k=1}^{K_t}, \qquad t=1,2,\dots
\]

with observation model

\[
y_{t,k} = y_s(x_{t,k},\theta_t^\star) + \delta_t^\star(x_{t,k}) + \varepsilon_{t,k},
\qquad
\varepsilon_{t,k}\sim\mathcal N(0,\sigma^2).
\]

We want an online method with three components:

1. **theta step**: local Bayesian update for \(	heta\), approximated by PF
2. **delta step**: conditional GP posterior update for discrepancy
3. **regime controller**: R-BOCPD restart / pruning logic

The current half-discrepancy design is preserved in one important sense:

- discrepancy **does not** enter the PF weight update for \(	heta\)
- discrepancy **does** enter the BOCPD predictive likelihood

This is intentional and should remain unchanged.

---

# 2. Why exact online-BPC differs from the current implementation

The current compressed implementation effectively stores a discrepancy posterior only on a very small support (often only the previous batch support), then propagates that state forward.

That is **not** theory-consistent for an exact GP posterior recursion.

For a GP, the correct sequential update should carry forward the previous posterior as a function-space object (or an exact finite-dimensional representation on an expanding support), not compress it to only the last batch.

The exact version we want should therefore:

- preserve all past support points inside the current expert segment,
- update the discrepancy posterior on the expanded support,
- never project the posterior back down to only the current batch support.

This is the core requirement.

---

# 3. Theta step (unchanged from current half-discrepancy)

For each active expert \(e\), maintain particles

\[
\{(\theta_{e,t}^{(j)}, w_{e,t}^{(j)})\}_{j=1}^N.
\]

Propagate:

\[
\theta_{e,t}^{(j)} \sim p(\theta_t \mid \theta_{e,t-1}^{(j)}).
\]

Reweight using the discrepancy-free pseudo-likelihood:

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

This remains the local search channel for \(	heta\).

**Do not inject discrepancy into this PF weight update.**

---

# 4. Exact online-BPC discrepancy update: abstract statement

The clean abstract online-BPC discrepancy update is:

\[
q_t(\delta)
=
\arg\min_q
\left\{
-\mathbb E_q[\log p(r_t \mid \delta_{X_t})]
+
\lambda_\delta\,\mathrm{KL}\!\big(q \,\|\, \widetilde q_{t\mid t-1}\big)
\right\},
\]

where:

- \(r_t\) is the current batch residual,
- \(\widetilde q_{t\mid t-1}\) is the predictive prior induced by the previous posterior on the **expanded support**,
- the KL penalty enforces local continuity to the previous posterior,
- the likelihood acts only on the new batch support \(X_t\).

The exact solution is

\[
q_t(\delta_{S_t})
\propto
\widetilde q_{t\mid t-1}(\delta_{S_t})\,
p(r_t \mid \delta_{X_t})^{1/\lambda_\delta},
\]

where the current support is

\[
S_t = S_{t-1}\cup X_t.
\]

The crucial point is:

> the new posterior lives on \(S_t\), not only on \(X_t\).

---

# 5. Support notation

For one expert, define:

- current batch support:
  \[
  X_t
  \]
- previous support:
  \[
  S_{t-1}
  \]
- expanded support:
  \[
  S_t := S_{t-1}\cup X_t.
  \]

We will use a finite-dimensional GP state on \(S_t\).

Let the discrepancy values on support \(S_t\) be written as

\[
\delta_{S_t}.
\]

---

# 6. Shared exact online-BPC

This is the recommended first implementation.

## 6.1 Shared anchor and residual

After the PF update of the surviving expert \(e\), define the shared theta anchor

\[
\bar\mu_{e,t}(X_t)
=
\sum_{j=1}^N
w_{e,t}^{(j)}\,\mu_s(X_t,\theta_{e,t}^{(j)}).
\]

Then define the current batch residual

\[
r_{e,t} := Y_t - \bar\mu_{e,t}(X_t).
\]

This residual is the observation used in the discrepancy update.

## 6.2 Previous posterior state

Maintain a GP posterior on the previous support

\[
q_{e,t-1}(\delta_{S_{e,t-1}})
=
\mathcal N\!\big(m_{e,t-1}, C_{e,t-1}\big).
\]

This is the exact finite-dimensional representation of the discrepancy posterior restricted to the full support accumulated so far in this expert segment.

## 6.3 Predictive prior on the expanded support

Let

\[
S_{e,t} = S_{e,t-1}\cup X_t.
\]

Using the GP prior kernel \(k_\phi\), extend the previous posterior to the new support:

\[
\widetilde q_{e,t\mid t-1}(\delta_{S_{e,t}})
=
q_{e,t-1}(\delta_{S_{e,t-1}})
\,p(\delta_{X_t}\mid \delta_{S_{e,t-1}}).
\]

This is Gaussian:

\[
\widetilde q_{e,t\mid t-1}(\delta_{S_{e,t}})
=
\mathcal N\!\big(\widetilde m_{e,t\mid t-1}, \widetilde C_{e,t\mid t-1}\big).
\]

Codex should build this explicitly as the block Gaussian predictive extension of the old posterior onto the union support \(S_{e,t}\).

## 6.4 Tempered Bayesian update

Let \(H_t\) be the selection matrix that extracts the coordinates of \(\delta_{S_{e,t}}\) corresponding to \(X_t\). Then

\[
\delta_{X_t} = H_t \delta_{S_{e,t}}.
\]

Model the residual likelihood as

\[
r_{e,t} \mid \delta_{X_t}
\sim
\mathcal N(\delta_{X_t}, \Sigma_r).
\]

Then the exact online-BPC discrepancy update is

\[
q_{e,t}(\delta_{S_{e,t}})
=
\arg\min_q
\left\{
-\mathbb E_q[\log p(r_{e,t}\mid \delta_{X_t})]
+
\lambda_\delta\,\mathrm{KL}\!\left(
q(\delta_{S_{e,t}})
\,\|\, 
\widetilde q_{e,t\mid t-1}(\delta_{S_{e,t}})
\right)
\right\}.
\]

Its exact solution is

\[
q_{e,t}(\delta_{S_{e,t}})
\propto
\widetilde q_{e,t\mid t-1}(\delta_{S_{e,t}})
\,
p(r_{e,t}\mid \delta_{X_t})^{1/\lambda_\delta}.
\]

Because both terms are Gaussian, the posterior is Gaussian:

\[
q_{e,t}(\delta_{S_{e,t}})
=
\mathcal N(m_{e,t}, C_{e,t}).
\]

## 6.5 Closed-form Gaussian update

The tempered likelihood is equivalent to observation noise \(\lambda_\delta \Sigma_r\). Therefore

\[
C_{e,t}^{-1}
=
\widetilde C_{e,t\mid t-1}^{-1}
+
\frac{1}{\lambda_\delta}
H_t^\top \Sigma_r^{-1} H_t,
\]

\[
m_{e,t}
=
C_{e,t}
\left[
\widetilde C_{e,t\mid t-1}^{-1}\widetilde m_{e,t\mid t-1}
+
\frac{1}{\lambda_\delta}
H_t^\top \Sigma_r^{-1} r_{e,t}
\right].
\]

This is the exact shared online-BPC GP update on the expanded support.

## 6.6 Important implementation rule

After the update, **keep** the state \((S_{e,t}, m_{e,t}, C_{e,t})\).

Do **not** compress the support back to only \(X_t\).

This is the defining requirement of the exact version.

---

# 7. Particle-specific exact online-BPC

This is the more theoretically pristine, but much more expensive, version.

## 7.1 Per-particle discrepancy posterior

For each theta particle \(j\) inside expert \(e\), maintain

\[
q_{e,t-1}^{(j)}(\delta_{S_{e,t-1}^{(j)}})
=
\mathcal N\!\big(m_{e,t-1}^{(j)}, C_{e,t-1}^{(j)}\big).
\]

This is a separate discrepancy posterior state for each theta particle.

## 7.2 Current residual for particle \(j\)

After PF update, define

\[
r_{e,t}^{(j)} := Y_t - \mu_s(X_t,\theta_{e,t}^{(j)}).
\]

## 7.3 Expanded support for particle \(j\)

Let

\[
S_{e,t}^{(j)} := S_{e,t-1}^{(j)} \cup X_t.
\]

Extend the old posterior to this expanded support via the GP predictive law:

\[
\widetilde q_{e,t\mid t-1}^{(j)}(\delta_{S_{e,t}^{(j)}})
=
q_{e,t-1}^{(j)}(\delta_{S_{e,t-1}^{(j)}})\,
p(\delta_{X_t}\mid \delta_{S_{e,t-1}^{(j)}}).
\]

Again this is Gaussian:

\[
\widetilde q_{e,t\mid t-1}^{(j)}(\delta_{S_{e,t}^{(j)}})
=
\mathcal N\!\big(\widetilde m_{e,t\mid t-1}^{(j)}, \widetilde C_{e,t\mid t-1}^{(j)}\big).
\]

## 7.4 Tempered Bayesian update for particle \(j\)

Define the selection matrix \(H_t^{(j)}\) that extracts the current batch part from \(S_{e,t}^{(j)}\). Then

\[
q_{e,t}^{(j)}(\delta_{S_{e,t}^{(j)}})
\propto
\widetilde q_{e,t\mid t-1}^{(j)}(\delta_{S_{e,t}^{(j)}})
\,
p(r_{e,t}^{(j)}\mid \delta_{X_t})^{1/\lambda_\delta}.
\]

Equivalently,

\[
C_{e,t}^{(j)\,-1}
=
\widetilde C_{e,t\mid t-1}^{(j)\,-1}
+
\frac{1}{\lambda_\delta}
H_t^{(j)\top}\Sigma_r^{-1}H_t^{(j)},
\]

\[
m_{e,t}^{(j)}
=
C_{e,t}^{(j)}
\left[
\widetilde C_{e,t\mid t-1}^{(j)\,-1}\widetilde m_{e,t\mid t-1}^{(j)}
+
\frac{1}{\lambda_\delta}
H_t^{(j)\top}\Sigma_r^{-1}r_{e,t}^{(j)}
\right].
\]

Again, keep the full expanded support state after the update.

## 7.5 Important interpretation

The particle-specific expert now carries

\[
\left\{
\theta_{e,t}^{(j)},\, w_{e,t}^{(j)},\, q_{e,t}^{(j)}(\delta)
\right\}_{j=1}^N,
\]

so the expert-level joint posterior approximation is

\[
q_{e,t}(\theta,\delta)
\approx
\sum_{j=1}^N
w_{e,t}^{(j)}
\,
\delta_{\theta_{e,t}^{(j)}}(\theta)\,
q_{e,t}^{(j)}(\delta).
\]

---

# 8. BOCPD predictive likelihoods

BOCPD must use the **pre-update** discrepancy-aware predictive distribution.

That means:

- first evaluate expert predictive likelihood using \(t-1\) states,
- then run BOCPD update / restart,
- then update theta and delta for surviving experts.

## 8.1 Shared version BOCPD likelihood

For pre-update expert \(e\):

\[
q_{e,t-1}(\delta_{S_{e,t-1}})
=
\mathcal N(m_{e,t-1}, C_{e,t-1}).
\]

On current batch inputs \(X_t\), the discrepancy predictive marginal is

\[
\delta(X_t)\mid q_{e,t-1}
\sim
\mathcal N\!\big(
m_{e,t-1}(X_t),\,
C_{e,t-1}(X_t,X_t)
\big).
\]

Therefore for theta particle \(j\),

\[
p(Y_t\mid X_t,\theta_{e,t-1}^{(j)}, q_{e,t-1}(\delta))
=
\mathcal N\!\Big(
Y_t;\,
\mu_s(X_t,\theta_{e,t-1}^{(j)}) + m_{e,t-1}(X_t),\,
\Sigma_s(X_t,\theta_{e,t-1}^{(j)})
+
C_{e,t-1}(X_t,X_t)
+
\sigma^2 I
\Big).
\]

So the shared BOCPD likelihood is

\[
p_{\mathrm{BOCPD}}(Y_t\mid X_t,e)
=
\sum_{j=1}^N
w_{e,t-1}^{(j)}
\,
\mathcal N\!\Big(
Y_t;\,
\mu_s(X_t,\theta_{e,t-1}^{(j)}) + m_{e,t-1}(X_t),\,
\Sigma_s(X_t,\theta_{e,t-1}^{(j)})
+
C_{e,t-1}(X_t,X_t)
+
\sigma^2 I
\Big).
\]

This is a Gaussian mixture.

## 8.2 Particle-specific version BOCPD likelihood

For each particle \(j\), use its own discrepancy posterior:

\[
\delta(X_t)\mid q_{e,t-1}^{(j)}
\sim
\mathcal N\!\big(
m_{e,t-1}^{(j)}(X_t),\,
C_{e,t-1}^{(j)}(X_t,X_t)
\big).
\]

Then

\[
p(Y_t\mid X_t,\theta_{e,t-1}^{(j)}, q_{e,t-1}^{(j)}(\delta))
=
\mathcal N\!\Big(
Y_t;\,
\mu_s(X_t,\theta_{e,t-1}^{(j)}) + m_{e,t-1}^{(j)}(X_t),\,
\Sigma_s(X_t,\theta_{e,t-1}^{(j)})
+
C_{e,t-1}^{(j)}(X_t,X_t)
+
\sigma^2 I
\Big).
\]

So the particle-specific BOCPD likelihood is

\[
p_{\mathrm{BOCPD}}(Y_t\mid X_t,e)
=
\sum_{j=1}^N
w_{e,t-1}^{(j)}
\,
\mathcal N\!\Big(
Y_t;\,
\mu_s(X_t,\theta_{e,t-1}^{(j)}) + m_{e,t-1}^{(j)}(X_t),\,
\Sigma_s(X_t,\theta_{e,t-1}^{(j)})
+
C_{e,t-1}^{(j)}(X_t,X_t)
+
\sigma^2 I
\Big).
\]

Again this is a Gaussian mixture.

## 8.3 BOCPD loss

Use

\[
\ell_{e,t} = -\log p_{\mathrm{BOCPD}}(Y_t\mid X_t,e).
\]

Then reuse the existing R-BOCPD mass update, restart rule, and pruning logic.

---

# 9. Hyperparameters

For the **exact benchmark implementation**, do **not** re-optimize GP hyperparameters every batch.

Instead:

- choose one fixed GP hyperparameter set \(\phi\) per experiment, or
- initialize \(\phi\) once from a reasonable calibration batch / pre-fit.

Reason:

- the exact online-BPC recursion is already defined at the posterior level,
- re-optimizing \(\phi\) every batch would change the model family itself,
- this would make it harder to isolate whether the exact posterior recursion works.

Therefore for the exact benchmark:

- **kernel hyperparameters fixed**
- **posterior recursion exact**
- **support expanding**

This is the cleanest theoretical benchmark.

---

# 10. What must change in the current code

The current `R-BOCPD-PF-particleGP-halfdiscrepancy` must be modified in the following way.

## 10.1 Shared version changes

Each expert should now store:

```python
class SharedExactDeltaState:
    support_X: Tensor          # all support points accumulated in this expert segment
    mean: Tensor               # posterior mean on support_X
    cov: Tensor                # posterior covariance on support_X
    kernel_hyperparams: ...
    lambda_delta: float
    obs_noise_cov: ...
```

Required new operations:

1. `extend_support_with_batch(X_batch)`  
   builds the expanded support \(S_t = S_{t-1} ∪ X_t\)

2. `build_predictive_prior_on_expanded_support()`  
   computes \((\widetilde m_{t|t-1}, \widetilde C_{t|t-1})\)

3. `update_with_residual_batch(r_t, X_batch)`  
   performs the exact Gaussian posterior update using the formulas above

4. `predict_batch(X_batch)`  
   returns \(m(X_batch)\) and \(C(X_batch,X_batch)\)

## 10.2 Particle-specific version changes

Each expert should now store, for each theta particle:

```python
class ParticleExactDeltaState:
    support_X: Tensor
    mean: Tensor
    cov: Tensor
    kernel_hyperparams: ...
    lambda_delta: float
    obs_noise_cov: ...
```

So an expert stores something like:

```python
theta_particles: Tensor               # [N, d_theta]
theta_weights: Tensor                 # [N]
delta_states: list[ParticleExactDeltaState]   # length N
```

Required new operations:

1. after PF resampling, copy / inherit the parent particle's delta state;
2. for each surviving particle, extend its support with the current batch;
3. for each particle, update its delta state using its residual
   \[
   r_t^{(j)} = Y_t - \mu_s(X_t,	heta_t^{(j)})
   \]

## 10.3 Important no-compression rule

Do **not** replace the support with only the newest batch.

The state after batch `t` must still live on the full expanded support.

This rule must be enforced in both shared and particle-specific implementations.

---

# 11. Batch update order

The update order for one batch must be:

## Step A: pre-update BOCPD scoring
For each current expert `e`, compute the discrepancy-aware predictive likelihood using the pre-update state.

## Step B: BOCPD update
Run existing R-BOCPD mass update, restart, pruning.

## Step C: theta PF update
For each surviving expert, propagate and update theta particles using the discrepancy-free PF likelihood.

## Step D: delta posterior update
For each surviving expert:

- shared version:
  - compute shared anchor residual
  - exact GP posterior update on expanded support

- particle-specific version:
  - compute residual per theta particle
  - exact GP posterior update on expanded support for each particle

This order must be preserved.

---

# 12. Complexity warning

## Shared exact online-BPC
Feasible as a benchmark.

State size grows with total support size in the expert segment:

\[
|S_t| = \sum_{u=r_e}^t K_u.
\]

Covariance matrix is \(|S_t|\times |S_t|\).

This may still be manageable for moderate segment lengths and small batches.

## Particle-specific exact online-BPC
Much heavier.

Each theta particle carries its own full GP posterior on an expanding support.

If there are:

- \(N\) theta particles,
- support size \(|S_t|\),

then memory is roughly \(O(N |S_t|^2)\).

This will likely be very expensive, but it is still the correct exact benchmark.

---

# 13. Recommended experimental plan

## First run
Implement **shared exact online-BPC** first.

Compare against:

- current `halfdiscrepancy`
- current compressed online-BPC

This will tell us whether state compression is the main reason the current online-BPC is underperforming.

## Second run
Implement **particle-specific exact online-BPC**.

Compare against:

- shared exact online-BPC
- halfdiscrepancy

This will tell us whether the particle-specific conditional posterior is actually beneficial or whether it structurally hurts BOCPD / identifiability.

---

# 14. Minimal pseudocode

## 14.1 Shared exact online-BPC

```text
for each batch t:
    for each current expert e:
        compute p_BOCPD(Y_t | X_t, e) using pre-update:
            {theta_{e,t-1}^{(j)}, w_{e,t-1}^{(j)}, q_{e,t-1}(delta)}
    run R-BOCPD update / restart / pruning

    for each surviving expert e:
        # theta PF step
        propagate theta particles
        update theta weights with discrepancy-free PF likelihood
        resample if needed

        # shared exact delta step
        compute mu_bar_{e,t}(X_t)
        r_{e,t} = Y_t - mu_bar_{e,t}(X_t)

        expand support:
            S_{e,t} = S_{e,t-1} ∪ X_t

        build predictive prior on S_{e,t}
        update Gaussian posterior on S_{e,t}
        keep full expanded state
```

## 14.2 Particle-specific exact online-BPC

```text
for each batch t:
    for each current expert e:
        compute p_BOCPD(Y_t | X_t, e) using pre-update:
            {theta_{e,t-1}^{(j)}, w_{e,t-1}^{(j)}, q_{e,t-1}^{(j)}(delta)}
    run R-BOCPD update / restart / pruning

    for each surviving expert e:
        # theta PF step
        propagate theta particles
        update theta weights with discrepancy-free PF likelihood
        resample if needed
        copy parent delta states during resampling

        # particle-specific exact delta step
        for each theta particle j:
            r_{e,t}^{(j)} = Y_t - mu_s(X_t, theta_{e,t}^{(j)})
            S_{e,t}^{(j)} = S_{e,t-1}^{(j)} ∪ X_t
            build predictive prior on S_{e,t}^{(j)}
            update Gaussian posterior on S_{e,t}^{(j)}
            keep full expanded state
```

---

# 15. Final interpretation

The exact benchmark should be interpreted as:

> a regime-aware exact finite-history GP implementation of online Bayesian projected calibration.

- PF still approximates the local Bayesian update for theta;
- discrepancy is now updated by the exact tempered GP posterior recursion on an expanding support;
- BOCPD evaluates whether the full local predictive decomposition remains valid.

This is the benchmark we want before introducing any engineering approximations.
