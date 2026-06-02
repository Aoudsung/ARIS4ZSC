# Method Design: V0-Anchored Posterior-Predictive SRVF-MAPPO

**Full name:** V0-anchored posterior-predictive Shared Response-Value Factor MAPPO  
**Short name:** SRVF-MAPPO  
**Document type:** Scientific method-design specification  
**Target setting:** Zero-shot coordination in Overcooked-style cooperative Markov games  
**Primary benchmark family:** Classic Overcooked resettable simulation environments  
**Primary scientific object:** The posterior-predictive value residual induced by an unseen partner's response type  
**Design source:** `/Users/aoudsung/Downloads/Modified Method.md`  
**Implementation status:** Core single-file implementation lives in `raob/srvf_mappo.py`; real classic wrapper smoke passed on the server, and formal classic training/evaluation is exposed through `python -m raob.srvf_mappo formal-classic`  
**Revision status:** Comprehensive replacement of the earlier V0-SRVF representation-only design

---

## 0. Executive Summary

This design unifies the method around one object:

\[
\boxed{\text{the posterior-predictive } V_0\text{-anchored Bayes-adaptive advantage}.}
\]

The method is not "MAPPO plus an SRVF bonus." The key move is to define a belief MDP whose latent variable is the partner's response-value type, and to use \(V_0\) only as a partner-blind anchor for decomposing the Bayes-adaptive action value.

The resulting method is:

\[
\boxed{\text{SRVF-MAPPO} = \text{MAPPO on a } V_0\text{-anchored posterior-predictive belief MDP}.}
\]

The SRVF term, belief update, action score, and conservative shrinkage are consequences of the same posterior-predictive quantity. SRVF supplies the one-step partner-dependent value residual on a shared \(V_0\) landscape. MAPPO supplies the continuation residual needed for long-horizon Overcooked control.

This revision narrows the scientific claim. The method does not claim that SRVF alone solves full-episode Overcooked. It claims that a source-calibrated, \(V_0\)-anchored posterior-predictive residual can be safely injected into a belief-MDP policy when source-calibrated reliability supports using the SRVF family.

---

## 1. Core Object

### 1.1 Latent response-value type

Let the latent partner variable be:

\[
\xi = (M,\beta).
\]

Here \(M \in \{0,1\}\) is a model-validity latent:

\[
M=1:
\text{ the partner is inside the source SRVF response-value family},
\]

\[
M=0:
\text{ SRVF is not trusted for this partner; use population fallback}.
\]

The continuous latent \(\beta \in \mathbb R^K\) is the SRVF response-value factor. Under \(M=0\), set \(\beta=0\). Under \(M=1\), \(\beta \sim p_0(\beta)\), where \(p_0\) is fitted from source partners.

This construction makes population fallback, raw SRVF, and calibrated SRVF part of one mixture model rather than three unrelated scoring rules.

### 1.2 Partner-blind value residual

Let:

- \(s\): full simulator state;
- \(o^{public}\): public observation available to the ego agent;
- \(g=\mathcal A(o^{public})\): deterministic public affordance chart;
- \(a\): ego primitive action;
- \(p\): partner policy or partner instance;
- \(r_p(g,a)\): raw reward after forcing ego action \(a\) and observing partner response;
- \(g'_p(g,a)\): next public chart after the intervention.

Train a partner-blind raw-reward value function \(V_0(g)\) from the source-population data. The one-step \(V_0\)-anchored partner residual is:

\[
A^0_p(g,a)
=
r_p(g,a)+\gamma V_0(g'_p(g,a))-V_0(g).
\]

The centered residual is denoted \(\bar A^0_p(g,a)\). The centering removes state-level baseline value and focuses the method on which primitive action is preferred under the partner's local response.

The value-residual observation model is:

\[
\mathbb E_\psi[\bar A^0(g,a)\mid g,a,M,\beta]
=
A_0(g,a)+M\,U_\psi(g,a)^\top\beta.
\]

The response model is:

\[
p_\psi(\Delta z\mid g,a,M,\beta)
=
\begin{cases}
p_\psi(\Delta z\mid g,a,\beta), & M=1,\\
p_{\mathrm{pop}}(\Delta z\mid g,a), & M=0.
\end{cases}
\]

### 1.3 Execution-time history and belief

At execution time, the observable history is:

\[
h_t=\{(g_i,a_i,\Delta z_i)\}_{i<t}.
\]

The belief is:

\[
b_t(\xi)=p_\psi(M,\beta\mid h_t).
\]

The central posterior-predictive score is:

\[
\boxed{
S_\psi(g,a,h_t)
=
\mathbb E_{\xi\sim b_t}
\left[
\bar A^0(g,a,\xi)
\right].
}
\]

By the model above:

\[
\boxed{
S_\psi(g,a,h_t)
=
A_0(g,a)+\alpha_t U_\psi(g,a)^\top\mu_t,
}
\]

where:

\[
\alpha_t=p_\psi(M=1\mid h_t),
\qquad
\mu_t=\mathbb E_\psi[\beta\mid h_t,M=1].
\]

This one expression contains:

- raw SRVF when \(\alpha_t=1\);
- population fallback when \(\alpha_t=0\);
- conservative calibrated SRVF when \(0<\alpha_t<1\).

Equivalently:

\[
\boxed{
S_\psi(g,a,h_t)
=
Q_{\mathrm{pop}}(g,a)
+
\alpha_t
\left[
Q_{\mathrm{SRVF}}(g,a,h_t)-Q_{\mathrm{pop}}(g,a)
\right].
}
\]

---

## 2. \(V_0\)-Anchored Posterior-Predictive Bellman Decomposition

Let \(x_t=(s_t,b_t)\) be the Bayes-adaptive belief state. Let \(g_t=\mathcal A(o_t^{public})\). Let \(V^\tau_*(x)\) and \(Q^\tau_*(x,a)\) be the entropy-regularized optimal value and action-value functions in the belief MDP.

For any partner-blind \(V_0(g)\):

\[
\boxed{
Q^\tau_*(s,b,a)-V_0(g)
=
\underbrace{
\mathbb E_{\xi\sim b}
\left[
r+\gamma V_0(g')-V_0(g)
\mid s,g,a,\xi
\right]
}_{\text{posterior-predictive SRVF residual}}
+
\underbrace{
\gamma
\mathbb E
\left[
V^\tau_*(s',b')-V_0(g')
\mid s,b,a
\right]
}_{\text{MAPPO continuation residual}}.
}
\]

Under the SRVF latent model:

\[
\mathbb E_{\xi\sim b}
\left[
r+\gamma V_0(g')-V_0(g)
\mid g,a,\xi
\right]
=
A_0(g,a)+\alpha U(g,a)^\top\mu.
\]

Therefore:

\[
\boxed{
Q^\tau_*(s,b,a)-V_0(g)
=
A_0(g,a)+\alpha U(g,a)^\top\mu
+
\gamma
\mathbb E
\left[
V^\tau_*(s',b')-V_0(g')
\right].
}
\]

The proof is the add-and-subtract identity:

\[
Q^\tau_*(s,b,a)
=
\mathbb E[r+\gamma V^\tau_*(s',b')\mid s,b,a],
\]

then subtract \(V_0(g)\), add and subtract \(\gamma V_0(g')\):

\[
Q^\tau_*(s,b,a)-V_0(g)
=
\mathbb E[r+\gamma V_0(g')-V_0(g)]
+
\gamma\mathbb E[V^\tau_*(s',b')-V_0(g')].
\]

The first term is exactly the SRVF residual object. The second term is what MAPPO must learn.

This resolves the earlier conceptual tension. SRVF should not be asked to solve long-horizon Overcooked by itself. Its principled role is the first term in a Bellman decomposition. MAPPO supplies the continuation term.

---

## 3. Derived Components

### 3.1 Policy loss

The policy is the entropy-regularized optimal policy for the belief MDP:

\[
\pi^*(a\mid s,b)
\propto
\exp
\left(
\frac{Q^\tau_*(s,b,a)}{\tau}
\right).
\]

Using the decomposition above, parameterize the action energy as:

\[
E_{\theta,\psi}(s,b,a)
=
C_\theta(s,b,a)
+
S_\psi(g,a,h),
\]

where:

\[
S_\psi(g,a,h)
=
A_0(g,a)+\alpha U_\psi(g,a)^\top\mu
\]

is the posterior-predictive SRVF residual, and:

\[
C_\theta(s,b,a)
\approx
\gamma
\mathbb E[
V^\tau_*(s',b')-V_0(g')
]
\]

is the continuation residual learned by MAPPO.

The actor is:

\[
\boxed{
\pi_{\theta,\psi}(a\mid s,b)
=
\mathrm{softmax}_a
\left(
\frac{
C_\theta(s,b,a)
+
A_0(g,a)
+
\alpha U_\psi(g,a)^\top\mu
}{\tau}
\right).
}
\]

MAPPO is the stochastic optimizer for the entropy-regularized belief-MDP objective:

\[
\boxed{
J(\theta,\psi)
=
\mathbb E_{\tau\sim\pi_{\theta,\psi}}
\left[
\sum_{t=0}^{T-1}
\gamma^t
\left(
r_t+\tau \mathcal H(\pi_{\theta,\psi}(\cdot\mid s_t,b_t))
\right)
\right].
}
\]

The PPO clipped surrogate is an implementation-level trust-region estimator of the policy gradient of this objective:

\[
\mathcal L_\pi^{\mathrm{PPO}}
=
-
\mathbb E_t
\left[
\min
\left(
\rho_t\hat A^B_t,
\mathrm{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A^B_t
\right)
\right],
\]

where \(\hat A^B_t\) is the GAE estimate in the belief state \(x_t=(s_t,b_t)\).

The value loss is a projection step used to estimate:

\[
V^\tau_\omega(s,b)
\approx
\mathbb E_{\pi_{\theta,\psi}}
\left[
\sum_{k\ge 0}
\gamma^k
\left(
r_{t+k}
+
\tau\mathcal H(\pi_{\theta,\psi}(\cdot\mid s_{t+k},b_{t+k}))
\right)
\mid s_t=s,b_t=b
\right].
\]

\(L_V\) is therefore a numerical approximation to the value function in the same belief MDP, not a separate scientific objective.

### 3.2 Partner belief update

The belief update is Bayes filtering over \(\xi=(M,\beta)\):

\[
\boxed{
b_{t+1}(M,\beta)
\propto
b_t(M,\beta)\,
p_\psi(\Delta z_t\mid g_t,a_t,M,\beta).
}
\]

This uses only:

\[
(g_t,a_t,\Delta z_t).
\]

Conditioned on \(M=1\), if:

\[
\Delta z_t
=
z_0(g_t,a_t)+R_\psi(g_t,a_t)\beta+\epsilon_t,
\qquad
\epsilon_t\sim\mathcal N(0,\Sigma_z),
\]

then the posterior over \(\beta\) is Gaussian:

\[
\Sigma_{t+1}^{-1}
=
\Sigma_t^{-1}
+
R_t^\top\Sigma_z^{-1}R_t,
\]

\[
\mu_{t+1}
=
\Sigma_{t+1}
\left[
\Sigma_t^{-1}\mu_t
+
R_t^\top\Sigma_z^{-1}
(\Delta z_t-z_0(g_t,a_t))
\right].
\]

The model-validity probability would update as:

\[
\boxed{
\alpha_{t+1}
=
p(M=1\mid h_{t+1})
=
\frac{
\alpha_t\,p_\psi(\Delta z_t\mid g_t,a_t,h_t,M=1)
}{
\alpha_t\,p_\psi(\Delta z_t\mid g_t,a_t,h_t,M=1)
+
(1-\alpha_t)\,p_{\mathrm{pop}}(\Delta z_t\mid g_t,a_t,M=0)
}.
}
\]

In practice, \(p_{\mathrm{pop}}\) for arbitrary OOD partners is not identifiable from source data alone. Therefore \(\alpha_t\) must be estimated by source-only calibration rather than by pretending that the OOD likelihood is known.

### 3.3 Action scoring

The action scorer is not an added head. It is the posterior expectation of the value-residual emission:

\[
\boxed{
S_\psi(g,a,h)
=
\mathbb E_{\xi\sim b_h}
[
\bar A^0(g,a,\xi)
].
}
\]

Expanding:

\[
S_\psi(g,a,h)
=
\mathbb E[
A_0(g,a)+M U(g,a)^\top\beta
\mid h
].
\]

Since \(A_0(g,a)\) is deterministic:

\[
S_\psi(g,a,h)
=
A_0(g,a)
+
\mathbb E[M\beta\mid h]^\top U(g,a).
\]

With:

\[
\alpha=p(M=1\mid h),
\qquad
\mu=\mathbb E[\beta\mid h,M=1],
\]

we get:

\[
\boxed{
S_\psi(g,a,h)
=
A_0(g,a)+\alpha U(g,a)^\top\mu.
}
\]

The final actor logits are:

\[
\boxed{
\ell_{\theta,\psi}(s,b,a)
=
C_\theta(s,b,a)
+
\frac{1}{\tau}
\left[
A_0(g,a)+\alpha U(g,a)^\top\mu
\right].
}
\]

This is not "MAPPO logits plus a heuristic bonus." It is the soft Bellman energy decomposed into a continuation residual and a posterior-predictive \(V_0\)-residual.

### 3.4 Calibration and shrinkage toward population mean

Conservative shrinkage is derived by the latent \(M\). The posterior-predictive belief over \(\beta\) is:

\[
p(\beta\mid h)
=
(1-\alpha_h)\delta_0(\beta)
+
\alpha_h p(\beta\mid h,M=1).
\]

Then:

\[
\mathbb E[\beta\mid h]
=
\alpha_h \mathbb E[\beta\mid h,M=1]
=
\alpha_h\mu_h.
\]

Therefore:

\[
S_\psi(g,a,h)
=
A_0(g,a)+U(g,a)^\top\mathbb E[\beta\mid h]
=
A_0(g,a)+\alpha_h U(g,a)^\top\mu_h.
\]

Shrinkage toward population mean is not a separate regularizer. It is the posterior-predictive action value under uncertainty about whether SRVF is valid for the current partner.

The practical source-only calibration rule estimates:

\[
\alpha_h \approx p(M=1\mid \rho(h)),
\]

where \(\rho(h)\) is a response-derived reliability vector. Valid reliability features include response reconstruction error, beta support distance, and posterior contraction, provided they are computed only from response observations. Beta support distance may also be used for an OOD alpha cap.

A source-only calibration objective is:

\[
\boxed{
\hat\alpha(\rho)
=
\arg\min_{\alpha\in\mathcal A}
\widehat{\mathbb E}_{\mathrm{source\ splits}}
\left[
\ell_{\mathrm{decision}}
\left(
A_0+\alpha(U^\top\mu),
\bar A^0
\right)
\mid \rho
\right].
}
\]

The decision loss should be action regret or a conservative upper-tail regret criterion. The calibration split must be disjoint between source adaptation and source evaluation. Target labels, target returns, target identity, target partner ids, and target action labels must not influence calibration. Only target response features are allowed at evaluation time.

The OOD fallback follows:

\[
\beta\text{ far from source support}
\Rightarrow
\alpha\downarrow 0
\Rightarrow
S_\psi(g,a,h)\to A_0(g,a).
\]

Thus the fallback is population-anchored MAPPO, not random action.

---

## 4. Unified Objective

The clean total objective is a single variational free energy / control-as-inference ELBO:

\[
\boxed{
\mathcal F(\theta,\psi)
=
\mathbb E_{q_{\theta,\psi}}
\left[
\log p_\psi
\left(
\mathcal D^{src}_{int},
\mathcal O_{0:T}=1,
\tau,
\xi
\right)
-
\log q_{\theta,\psi}(\tau,\xi)
\right].
}
\]

Here:

\[
\xi=(M,\beta),
\]

\[
\mathcal D^{src}_{int}
=
\{(g_i,a_i,\Delta z_i,\bar A^0_i)\},
\]

and \(\mathcal O_t=1\) is the maximum-entropy control optimality event with:

\[
p(\mathcal O_t=1\mid s_t,a_t)
\propto
\exp(r_t/\tau_R).
\]

Expanding the ELBO gives:

\[
\boxed{
\begin{aligned}
\mathcal F(\theta,\psi)
=
\mathbb E_{q_{\theta,\psi}}
\Bigg[
&
\sum_{p\in P_{src}}\log p_0(\xi_p)
\\
&+
\sum_{p\in P_{src}}
\sum_{i\in\mathcal D^p_{int}}
\log p_\psi
\left(
\Delta z_{pi},\bar A^0_{pi}
\mid g_i,a_i,\xi_p
\right)
\\
&+
\sum_{t=0}^{T-1}
\gamma^t
\left(
\frac{r_t}{\tau_R}
+
\log\pi_{\theta,\psi}(a_t\mid s_t,b_t)
\right)
\Bigg].
\end{aligned}
}
\]

The trained method minimizes:

\[
\boxed{
\mathcal L_{\mathrm{unified}}(\theta,\psi)
=
-\mathcal F(\theta,\psi).
}
\]

This is one objective. It decomposes into multiple terms only because the joint probability model factorizes. The apparent weights are not arbitrary patch coefficients:

- \(\sigma_z^2\) is response noise;
- \(\sigma_A^2\) is value-residual observation noise;
- \(\tau_R\) is reward optimality temperature;
- \(\tau\) is the soft-control policy temperature.

If the likelihood is Gaussian, \(-\log p_\psi(\Delta z,\bar A^0\mid g,a,\xi)\) becomes a response reconstruction term plus a value-residual reconstruction term.

If the likelihood for \(\bar A^0\) is ordinal instead of Gaussian, it becomes a pairwise ranking likelihood:

\[
p(a_i\succ a_j\mid g,\xi)
=
\sigma
\left(
\frac{
S_\psi(g,a_i,h)-S_\psi(g,a_j,h)
}{
\sigma_{\mathrm{rank}}
}
\right).
\]

Choose one value-observation model. Do not include both \(L_A\) and \(L_{\mathrm{rank}}\) unless the method explicitly assumes two independent observation channels for the same residual.

---

## 5. What Happens to Naive Terms

The naive stack is:

\[
L
=
L_{\mathrm{MAPPO}}
+
c_vL_V
+
c_HH(\pi)
+
\lambda_\Delta L_\Delta
+
\lambda_A L_A
+
\lambda_{\mathrm{rank}}L_{\mathrm{rank}}
+
\lambda_{\mathrm{router}}L_{\mathrm{router}}
+
\lambda_{\mathrm{option}}L_{\mathrm{option}}.
\]

Under the unified design:

- \(L_{\mathrm{MAPPO}}\) is the policy-gradient estimator for the belief-MDP control term.
- \(L_V\) is the projection used to estimate \(V^\tau(s,b)\), not a separate modeling assumption.
- \(H(\pi)\) comes from maximum-entropy optimality inference.
- \(L_\Delta\) comes from the response likelihood \(p_\psi(\Delta z\mid g,a,\xi)\).
- \(L_A\) comes from the value-residual likelihood \(p_\psi(\bar A^0\mid g,a,\xi)\) if a Gaussian residual observation model is chosen.
- \(L_{\mathrm{rank}}\) comes from replacing the Gaussian value-residual likelihood with an ordinal Bradley-Terry or Plackett-Luce likelihood.
- \(L_{\mathrm{router}}\) disappears. The router is the posterior trust coefficient \(\alpha=p(M=1\mid h)\).
- \(L_{\mathrm{option}}\) is excluded. Options would require a separate latent macro-action model and would violate the one-object design.

Long-horizon Overcooked structure should be learned by the continuation residual \(C_\theta\) in MAPPO, not patched in by auxiliary option supervision.

---

## 6. Design Constraints

### 6.1 \(V_0\) remains partner-blind

\(V_0\) enters only through:

\[
r+\gamma V_0(g')-V_0(g).
\]

\(V_0\) must not read partner identity, beta, response-factor estimates, partner-aware teacher labels, or target partner information.

### 6.2 Beta inference uses only response history

The execution-time inference contract is:

\[
p(\beta\mid h_t)
\propto
p(\beta)
\prod_{i<t}
p(\Delta z_i\mid g_i,a_i,\beta).
\]

The only allowed execution-time inputs for beta inference are:

\[
(g_i,a_i,\Delta z_i).
\]

No target return, target action label, target identity, or partner id may enter beta inference.

### 6.3 OOD fallback is population-anchored

When SRVF is unreliable:

\[
\alpha=p(M=1\mid h)\to 0,
\]

so:

\[
S_\psi(g,a,h)\to A_0(g,a).
\]

The policy does not collapse to random action. It becomes population-anchored MAPPO.

### 6.4 Conservative shrinkage is source-only

The calibration rule for \(\alpha\) must be fitted on source disjoint adaptation/evaluation splits.

The leakage guard is mandatory:

```text
target labels used for calibration: false
target returns used for calibration: false
target identity used for calibration: false
target partner ids used for alpha: false
target action labels used for calibration: false
target response features used for alpha at evaluation time: true
```

The last line is allowed because \(\alpha\) is a response-derived reliability score. It must not be fitted using target outcomes.

---

## 7. Validity Protocol and Experiments

The experiments must test the representation and control claims separately.

### 7.1 Validity gates before factor learning

Before interpreting SRVF factors, verify:

```text
interventional support
V0 residual rank calibration
repeat variance and signal-to-noise
nontrivial partner action-value effect
oracle residual gain
```

Failure interpretations:

```text
If support fails, the reservoir is inadequate.
If rank calibration fails, V0 is not a valid value landscape.
If SNR fails, partner response stochasticity overwhelms value signal.
If oracle residual gain fails, partner variation is not action-value relevant in the evaluated distribution.
```

### 7.2 Representation-level evaluation

Question:

> Does a low-rank source response-value family explain held-out partner residuals beyond the population mean?

Compare:

```text
population mean
categorical identity-prototype
raw SRVF
source-calibrated SRVF
oracle beta upper bound
oracle partner residual upper bound
```

Primary metrics:

```text
held-out action regret
top-1 action accuracy
value-residual ranking AUC
residual prediction error
negative-transfer rate
upper-tail regret
```

Success requires calibrated SRVF to reduce held-out action regret relative to population mean in partner-relation bins where response-value variation is not already captured by identity prototypes.

### 7.3 Belief-MDP policy evaluation

Question:

> Does posterior-predictive SRVF improve MAPPO action selection when injected as the \(V_0\)-anchored residual term?

Compare:

```text
population-anchored MAPPO
raw SRVF-MAPPO
source-calibrated SRVF-MAPPO
identity-prototype MAPPO
oracle-beta MAPPO
```

Primary metrics:

```text
mean return
nonzero reward rate
offline action regret proxy
posterior reliability alpha over time
fallback rate
failure cases by phase
```

Full-episode return is a stress test, not the sole proof of the scientific claim. It can be confounded by exploration, sparse reward, policy execution, state distribution, and environment bottlenecks.

### 7.4 Partner-relation bins

Results must be stratified by relation to the source population:

```text
near-source-prototype
interpolated / mixed-response
far-from-source / OOD
high-stochasticity partner
low-stochasticity partner
```

Expected interpretations:

```text
Identity-prototype may win in near-source-prototype bins.
SRVF should show advantage in interpolated or mixed-response bins if the continuous factor claim is correct.
Both SRVF and identity may fail in OOD bins; this failure must be reported.
Population fallback may win when partner variation has little action-value effect.
```

---

## 8. Where the Unification Breaks

### 8.1 Alpha is not a universal Bayesian OOD posterior

If a correct likelihood for \(M=0\) were available, then:

\[
\alpha=p(M=1\mid h)
\]

would be a fully Bayesian posterior. In reality, the OOD partner distribution is not known. Therefore the practical \(\alpha\) rule is an empirical-Bayes or conformal-style source calibration of model validity.

The correct claim is not "Bayes-optimal OOD cooperation." The correct claim is:

> \(\alpha\) is a source-calibrated lower-confidence trust coefficient for the SRVF family.

### 8.2 The \(V_0\)-residual is a surrogate

The Bellman decomposition is algebraically true for any \(V_0\), but scientific usefulness depends on whether:

\[
r+\gamma V_0(g')-V_0(g)
\]

actually ranks actions similarly to longer raw-return rollouts. This must be checked by support, rank calibration, repeat SNR, and oracle residual gain gates.

### 8.3 Use one value-observation likelihood first

A Gaussian likelihood gives \(L_A\). An ordinal likelihood gives \(L_{\mathrm{rank}}\). Using both is not a clean one-object method unless the method explicitly claims that residual magnitude and residual ordering are two conditionally independent measurements.

The default design should choose one value-observation model before adding the other.

### 8.4 Options are outside the narrowed method

An option loss cannot be derived from the proposed object. Adding it would introduce a second latent control abstraction. The narrowed method should drop option supervision and let MAPPO's continuation residual learn long-horizon structure.

---

## 9. Final Research Claim

The tight defensible claim is:

> In a belief MDP anchored by a partner-blind raw-reward \(V_0\), SRVF supplies a posterior-predictive estimate of the one-step partner-dependent value residual. When source-calibrated model validity is high, this residual can improve partner-conditioned action selection; when validity is low, the posterior-predictive score shrinks to the population mean, yielding a safe MAPPO fallback.

For reviewer-facing framing:

> The method's claim is not that \(\alpha\) is a universal Bayesian OOD posterior, and not that SRVF alone solves full Overcooked control. The claim is that a \(V_0\)-anchored posterior-predictive residual can be safely injected into MAPPO when source-calibrated reliability supports it.

This is the final method identity:

\[
\boxed{
\textbf{V0-anchored posterior-predictive SRVF-MAPPO}
}
\]
