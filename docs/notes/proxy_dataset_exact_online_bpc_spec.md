# Proxy-Dataset Exact Online-BPC Spec
## Exact online Bayesian projected calibration via pseudo-dataset GP refits
## For Codex implementation

This document specifies an **exact / theory-consistent** implementation of online Bayesian projected calibration (online-BPC) using a **proxy-dataset / pseudo-observation formulation**.

This formulation is mathematically equivalent to the posterior-propagation view, but is often easier to implement and reason about:

- instead of explicitly propagating the old GP posterior as a Gaussian state on an expanded support,
- we convert the old posterior into a **pseudo-dataset** on the historical support,
- then append the current residual batch as a new dataset,
- then do one exact GP posterior computation on the combined dataset.

This document covers:

1. the mathematical equivalence,
2. the exact **shared** version,
3. the exact **particle-specific** version,
4. how to modify the current `R-BOCPD-PF-particleGP-halfdiscrepancy`,
5. why the **shared** version should be implemented first.

---

# 0. Executive summary

The exact online-BPC discrepancy update can be written in two mathematically equivalent ways:

### Posterior-propagation view
\[
q_t(\delta_{S_t})
\propto
\widetilde q_{t\mid t-1}(\delta_{S_t})\,
p(r_t\mid \delta_{X_t})^{1/\lambda_\delta},
\]
where \(\widetilde q_{t\mid t-1}\) is the predictive prior induced by the old posterior on the expanded support \(S_t=S_{t-1}\cup X_t\).

### Proxy-dataset view
1. convert the old posterior on the historical support into a pseudo-dataset,
2. append the new residual batch as a tempered Gaussian observation,
3. fit one exact GP posterior on the combined dataset.

These two views are exactly equivalent in the Gaussian / GP setting.

The proxy-dataset form is recommended for implementation because it avoids explicitly coding predictive-prior block extensions.

---

# 1. Problem setup

We observe streaming batches

\[
\mathcal B_t = \{(x_{t,k}, y_{t,k})\}_{k=1}^{K_t}, \qquad t=1,2,\dots
\]

with observation model

\[
y_{t,k} = y_s(x_{t,k},\theta_t^\star) + \delta_t^\star(x_{t,k}) + \varepsilon_{t,k},
\qquad
\varepsilon_{t,k}\sim \mathcal N(0,\sigma^2).
\]

We keep the current half-discrepancy principle:

- `theta` is updated by PF using a discrepancy-free likelihood,
- discrepancy is updated **after** theta,
- BOCPD scores the full predictive model \(y_s + \delta\).

---

# 2. Theta step (unchanged)

For each active expert \(e\), maintain particles

\[
\{(\theta_{e,t}^{(j)}, w_{e,t}^{(j)})\}_{j=1}^N.
\]

Propagate:

\[
\theta_{e,t}^{(j)} \sim p(\theta_t \mid \theta_{e,t-1}^{(j)}).
\]

Reweight with discrepancy-free pseudo-likelihood:

\[
p_{\mathrm{PF}}(Y_t \mid X_t,\theta_{e,t}^{(j)})
=
\mathcal N\!\big(
Y_t;\,
\mu_s(X_t,\theta_{e,t}^{(j)}),
\Sigma_s(X_t,\theta_{e,t}^{(j)}) + \sigma_{\mathrm{PF}}^2 I
\big).
\]

\[
\widetilde w_{e,t}^{(j)}
\propto
w_{e,t-1}^{(j)}
p_{\mathrm{PF}}(Y_t \mid X_t,\theta_{e,t}^{(j)}),
\qquad
w_{e,t}^{(j)}
=
\frac{\widetilde w_{e,t}^{(j)}}{\sum_\ell \widetilde w_{e,t}^{(\ell)}}.
\]

This step is unchanged.

**Discrepancy must not be injected into this PF weight update.**

---

# 3. Shared exact online-BPC: posterior-propagation form

This section states the exact shared update in the posterior form.

## 3.1 Shared residual

After the PF update of a surviving expert \(e\), define the shared anchor

\[
\bar\mu_{e,t}(X_t)
=
\sum_{j=1}^N
w_{e,t}^{(j)}\,\mu_s(X_t,\theta_{e,t}^{(j)}).
\]

Define the current batch residual

\[
r_{e,t} := Y_t - \bar\mu_{e,t}(X_t).
\]

## 3.2 Previous posterior state

Maintain the discrepancy posterior on the historical support \(H:=S_{e,t-1}\):

\[
q_{e,t-1}(\delta_H)
=
\mathcal N(m_H, C_H).
\]

This is the exact GP posterior restricted to the full support accumulated in the current expert segment.

## 3.3 Exact online-BPC update

Let the new batch support be \(N:=X_t\), and define the expanded support

\[
S_t := H \cup N.
\]

The exact online-BPC update is

\[
q_{e,t}(\delta_{S_t})
\propto
\widetilde q_{e,t\mid t-1}(\delta_{S_t})
\,
p(r_{e,t}\mid \delta_N)^{1/\lambda_\delta},
\]

where \(\widetilde q_{e,t\mid t-1}\) is the predictive prior induced by \(q_{e,t-1}\) on the expanded support.

Because everything is Gaussian, the result is again a Gaussian posterior on \(S_t\).

---

# 4. Exact equivalence to a proxy dataset

Now we derive the pseudo-dataset form that is exactly equivalent to the above update.

## 4.1 Historical posterior on the historical support

Suppose the old discrepancy posterior on the historical support \(H\) is

\[
q_{t-1}(\delta_H) = \mathcal N(m_H, C_H).
\]

Let the prior GP covariance on \(H\) be

\[
K_H := K(H,H).
\]

We want to represent this posterior as the posterior resulting from the GP prior plus a pseudo-dataset on \(H\).

Assume a pseudo-observation model

\[
\tilde y_H \mid \delta_H \sim \mathcal N(\delta_H, \Lambda_H).
\]

Then standard Gaussian conditioning yields posterior

\[
C_H^{-1} = K_H^{-1} + \Lambda_H^{-1},
\]
\[
C_H^{-1} m_H = \Lambda_H^{-1} \tilde y_H.
\]

Therefore, if we define

\[
\Lambda_H^{-1} := C_H^{-1} - K_H^{-1},
\]

and

\[
\tilde y_H := \Lambda_H C_H^{-1} m_H,
\]

then the GP prior plus the pseudo-dataset \((H,\tilde y_H,\Lambda_H)\) induces exactly the posterior

\[
q_{t-1}(\delta_H)=\mathcal N(m_H,C_H).
\]

This is the key equivalence.

---

# 5. Add the current batch as a tempered observation

Now append the current residual batch.

For the new support \(N:=X_t\), use the current residual observation model

\[
r_t \mid \delta_N \sim \mathcal N(\delta_N,\Sigma_r).
\]

The KL-regularized / tempered online-BPC update uses

\[
p(r_t\mid \delta_N)^{1/\lambda_\delta},
\]

which is exactly equivalent to a Gaussian observation model with inflated noise

\[
r_t \mid \delta_N \sim \mathcal N(\delta_N,\lambda_\delta \Sigma_r).
\]

So the current batch contributes an ordinary Gaussian observation with covariance \(\lambda_\delta \Sigma_r\).

---

# 6. Combined exact proxy-dataset update

Let the expanded support be

\[
S_t = H \cup N.
\]

Construct the combined pseudo-dataset / real-data vector

\[
z_t :=
\begin{bmatrix}
\tilde y_H \\
r_t
\end{bmatrix}.
\]

Construct the observation operator

\[
A_t =
\begin{bmatrix}
I_H & 0 \\
0 & I_N
\end{bmatrix},
\]

and the observation noise

\[
\Omega_t =
\begin{bmatrix}
\Lambda_H & 0 \\
0 & \lambda_\delta \Sigma_r
\end{bmatrix}.
\]

Let the GP prior covariance on the expanded support be

\[
K_{S_t} := K(S_t,S_t).
\]

Then the exact posterior is

\[
q_t(\delta_{S_t})
=
\mathcal N(m_t,C_t),
\]

with

\[
C_t^{-1} = K_{S_t}^{-1} + A_t^\top \Omega_t^{-1} A_t,
\]

\[
m_t = C_t A_t^\top \Omega_t^{-1} z_t.
\]

This posterior is exactly equal to the posterior-propagation update from Section 3.

This is the implementation form we recommend.

---

# 7. Shared exact online-BPC implementation

This is the first version Codex should implement.

## 7.1 Expert discrepancy state

Each expert should carry:

```python
class SharedExactDeltaState:
    support_X: Tensor          # historical support H
    mean: Tensor               # posterior mean on H
    cov: Tensor                # posterior covariance on H
    kernel_hyperparams: ...
    lambda_delta: float
    obs_noise_cov: Tensor
```

Important: `support_X` must be the full support accumulated in the expert segment. Do **not** replace it with only the newest batch.

## 7.2 Shared update procedure

For a surviving expert `e` at batch `t`:

1. compute the shared PF anchor
   \[
   \bar\mu_{e,t}(X_t)=\sum_j w_{e,t}^{(j)}\mu_s(X_t,\theta_{e,t}^{(j)})
   \]
2. compute shared residual
   \[
   r_{e,t}=Y_t-\bar\mu_{e,t}(X_t)
   \]
3. let
   - historical support \(H=S_{e,t-1}\)
   - new support \(N=X_t\)
   - expanded support \(S_t=H\cup N\)
4. from the old posterior \((m_H,C_H)\), compute pseudo-dataset parameters
   \[
   \Lambda_H^{-1}=C_H^{-1}-K_H^{-1},
   \qquad
   \tilde y_H=\Lambda_H C_H^{-1}m_H
   \]
5. build the expanded prior covariance \(K_{S_t}\)
6. build combined observation
   \[
   z_t = [\tilde y_H; r_{e,t}]
   \]
   with block noise
   \[
   \Omega_t=
   \begin{bmatrix}
   \Lambda_H & 0\\
   0 & \lambda_\delta \Sigma_r
   \end{bmatrix}
   \]
7. compute the exact posterior \((m_t,C_t)\) on \(S_t\)
8. keep \((S_t,m_t,C_t)\) as the new state

That is the exact shared online-BPC discrepancy update.

---

# 8. Shared BOCPD predictive likelihood

BOCPD must use the **pre-update** predictive law.

Suppose the pre-update expert \(e\) has shared discrepancy posterior

\[
q_{e,t-1}(\delta_{S_{e,t-1}})
=
\mathcal N(m_{e,t-1}, C_{e,t-1}).
\]

Then on the current batch support \(X_t\),

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

Use

\[
\ell_{e,t} = -\log p_{\mathrm{BOCPD}}(Y_t\mid X_t,e)
\]

inside the existing R-BOCPD update.

---

# 9. Particle-specific exact online-BPC

This version is theoretically clean but much more expensive.

## 9.1 Per-particle discrepancy state

Each theta particle \(j\) inside expert \(e\) carries its own exact posterior state:

```python
class ParticleExactDeltaState:
    support_X: Tensor
    mean: Tensor
    cov: Tensor
    kernel_hyperparams: ...
    lambda_delta: float
    obs_noise_cov: Tensor
```

So the expert stores

```python
theta_particles: Tensor
theta_weights: Tensor
delta_states: list[ParticleExactDeltaState]
```

## 9.2 Particle-specific residual

After the PF update, for each particle \(j\) define

\[
r_{e,t}^{(j)} := Y_t - \mu_s(X_t,\theta_{e,t}^{(j)}).
\]

## 9.3 Particle-specific proxy-dataset update

For particle \(j\):

1. old support \(H_j := S_{e,t-1}^{(j)}\)
2. new support \(N := X_t\)
3. expanded support \(S_t^{(j)} := H_j \cup N\)
4. old posterior:
   \[
   q_{e,t-1}^{(j)}(\delta_{H_j})=\mathcal N(m_H^{(j)},C_H^{(j)})
   \]
5. compute historical pseudo-dataset
   \[
   (\Lambda_H^{(j)})^{-1}=(C_H^{(j)})^{-1}-(K_H^{(j)})^{-1},
   \qquad
   \tilde y_H^{(j)}=\Lambda_H^{(j)} (C_H^{(j)})^{-1} m_H^{(j)}
   \]
6. append the new residual observation
   \[
   r_{e,t}^{(j)} \mid \delta_N \sim \mathcal N(\delta_N,\lambda_\delta \Sigma_r)
   \]
7. fit exact GP posterior on \(S_t^{(j)}\)
8. keep the full expanded state for particle \(j\)

This is the particle-specific exact online-BPC update.

---

# 10. Particle-specific BOCPD predictive likelihood

For each pre-update theta particle \(j\),

\[
\delta(X_t)\mid q_{e,t-1}^{(j)}
\sim
\mathcal N\!\big(
m_{e,t-1}^{(j)}(X_t),\,
C_{e,t-1}^{(j)}(X_t,X_t)
\big).
\]

Hence

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

Therefore

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

---

# 11. Batch update order

This order must be preserved.

## Step A: pre-update BOCPD scoring
For each current expert, compute the discrepancy-aware predictive likelihood using the **pre-update** state.

## Step B: BOCPD update
Run the existing R-BOCPD mass update, restart, and pruning.

## Step C: theta PF update
For each surviving expert, propagate and update theta particles using the discrepancy-free PF likelihood.

## Step D: delta exact online-BPC update
For each surviving expert:

- shared version:
  - compute shared anchor residual
  - exact proxy-dataset GP update on the expanded support

- particle-specific version:
  - compute per-particle residuals
  - exact proxy-dataset GP update on the expanded support for each particle

---

# 12. Hyperparameters

For the exact benchmark:

- keep GP kernel hyperparameters fixed within an experiment,
- do **not** re-fit them every batch,
- optionally initialize them from a pre-fit or from a reasonable default.

Reason:

- this benchmark is meant to test the exact posterior recursion itself,
- changing hyperparameters every batch would make the model family change too,
- that would make it harder to identify whether the recursion or the approximation is the issue.

So for the benchmark:

- fixed hyperparameters
- exact posterior recursion
- expanding support

---

# 13. Complexity warning

## Shared exact online-BPC
Feasible as a benchmark.

If the support size inside an expert segment is

\[
|S_t| = \sum_{u=r_e}^{t} K_u,
\]

then the GP covariance is \(|S_t|\times |S_t|\).

This can still be manageable for small batches and moderate segment lengths.

## Particle-specific exact online-BPC
Very expensive.

Each theta particle carries its own exact GP posterior on an expanding support.

If there are \(N\) theta particles and support size \(|S_t|\), memory is roughly

\[
O(N |S_t|^2).
\]

This is likely too expensive for large \(N\), but it is still the correct exact benchmark.

---

# 14. Recommended experimental plan

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

This will tell us whether the particle-specific conditional discrepancy posterior is actually beneficial or whether it structurally hurts BOCPD / identifiability.

---

# 15. Minimal pseudocode

## 15.1 Shared exact online-BPC

```text
for each batch t:
    for each current expert e:
        compute p_BOCPD(Y_t | X_t, e)
        using pre-update:
            {theta_{e,t-1}^{(j)}, w_{e,t-1}^{(j)}, q_{e,t-1}(delta)}
    run R-BOCPD update / restart / pruning

    for each surviving expert e:
        # theta PF step
        propagate theta particles
        update theta weights with discrepancy-free PF likelihood
        resample if needed

        # shared exact delta step
        mu_bar = sum_j w_{e,t}^{(j)} mu_s(X_t, theta_{e,t}^{(j)})
        r_{e,t} = Y_t - mu_bar

        H = old support
        N = X_t
        S_t = H ∪ N

        convert old posterior on H into pseudo-dataset (tilde_y_H, Lambda_H)
        append current residual dataset (r_{e,t}, lambda_delta * Sigma_r)
        compute exact GP posterior on S_t
        keep full expanded state
```

## 15.2 Particle-specific exact online-BPC

```text
for each batch t:
    for each current expert e:
        compute p_BOCPD(Y_t | X_t, e)
        using pre-update:
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

            H_j = old support of particle j
            N = X_t
            S_t^{(j)} = H_j ∪ N

            convert old posterior on H_j into pseudo-dataset
            append current residual dataset
            compute exact GP posterior on S_t^{(j)}
            keep full expanded state
```

---

# 16. Final interpretation

The shared exact version should be interpreted as:

> an exact finite-history GP implementation of shared online Bayesian projected calibration.

The particle-specific exact version should be interpreted as:

> an exact finite-history GP implementation of particle-conditional online Bayesian projected calibration.

Both are mathematically exact within the fixed-hyperparameter GP model.

The shared version should be implemented first.
