# RELATED_WORK — Retired DELTA provenance (historical)

`authoritative: false`

> This document describes the retired V6 method for historical provenance only.
> It is not an active CETR specification, implementation guide, or evidence source.


## Purpose

This document records where each component of DELTA comes from and states the
boundary of what the paper claims. Components adopted from established work are
listed as adopted, with the specific form DELTA instantiates. The contribution
section states what DELTA defines on top of them.

Bibliographic status as of 2026-08-05. All 42 entries below were checked against
a primary arXiv record, proceedings page, journal or publisher record, or an
official bibliographic index. Where a final publication record exists, its
venue, year, and pages are preferred, with the arXiv identifier retained when
useful for traceability. `et al.` is used only where the abbreviated author list
does not create ambiguity. The main list prioritizes final publications in
major venues. Workshop-only papers are excluded. An arXiv-only direct source is
retained only when the paper's first page lists a qualifying research
institution; such entries are explicitly labelled `arXiv preprint` and are not
presented as peer-reviewed venue publications. Canonical original algorithms,
theory, books, and theses are retained when no semantically equivalent final
publication is available.

---

## 1. Zero-shot coordination and partner populations

**Adopted.** The benchmark, the partner-population training regime, and the
registered baselines follow the established ZSC line: Overcooked as a
coordination substrate (Carroll et al., NeurIPS 2019), other-play (Hu et al.,
ICML 2020), fictitious co-play (Strouse et al., NeurIPS 2021), and the
OvercookedV2 Test-Time Protocol Formation protocol (Gessler et al., ICLR
2025, arXiv:2503.17821).

**DELTA's use.** Training support is a mechanism-stratified population of
independent SP and OP parents with registered progress checkpoints. This
population supplies the variation that gives the latent variable its domain.
The same population methods appear in the registered baseline set, so the
comparison holds the partner ecology fixed across methods.

---

## 2. Latent-context meta-RL and Bayes-adaptive control

**Adopted.** The idea that an agent can carry a posterior over an unobserved
task variable and act under it is Bayes-adaptive control (Duff, PhD thesis
2002), realised in deep RL by probabilistic context inference (PEARL, Rakelly et
al., ICML 2019) and by belief-conditioned policies trained through a variational
objective (VariBAD, Zintgraf et al., ICLR 2020).

**DELTA's use.** DELTA carries an exact categorical posterior over `K`
exchangeable coordination modes and forms its executed policy from that
posterior at deployment time.

**Where DELTA's construction is its own.** Three properties define the
difference, each stated as a positive property with its consequence:

1. **The latent is identified by two observation channels of different units.**
   One channel is the observable teammate response; the other is a measured
   action-return contrast in raw task-return units. Both are emissions of the
   same variable in a single joint likelihood.

2. **The composite score normalizes by observation count.** The response term
   and the decision term enter a single negative log likelihood divided by the
   total number of observations of both kinds. The relative influence of each
   channel is therefore determined by how much evidence of each kind the run
   collected.

3. **The belief enters the policy through a single scalar tilting coefficient.**
   Base policy parameters are a function of the PPO objective alone and remain
   invariant to the belief. This invariance is what makes the same-world belief
   intervention (H3) an identified causal comparison: the intervention can hold
   the world, the source partner, the task state, the base policy, and the CRN
   continuation vector fixed while varying only the belief.

---

## 3. Teammate and opponent modelling

**Adopted.** Learning a predictive model of a teammate from local history is
established practice (ToMnet, Rabinowitz et al., ICML 2018; LIAM, Papoudakis et
al., NeurIPS 2021). The broader ad-hoc line includes PLASTIC's policy/model
selection for unknown teammates (Barrett et al., *Artificial Intelligence*
2017) and belief-based open ad-hoc teamwork under partial observability
(Rahman et al., *JMLR* 2023).

**DELTA's use.** The `response_only` variant is exactly this construction: a
latent trained by the response likelihood alone. It is registered as a
confirmatory control, and the H2 contrast measures the increment that the
decision channel adds over it.

---

## 4. KL-constrained policy improvement

**Adopted.** The deployment update is the standard exponential-tilting solution
of a relative-entropy-constrained improvement problem,

```text
pi(a) proportional to pi_0(a) exp(Q(a) / eta),   eta chosen so that KL(pi || pi_0) = delta
```

which appears as relative entropy policy search (Peters et al., AAAI 2010) and
as the E-step of maximum a posteriori policy optimisation (Abdolmaleki et al.,
ICLR 2018). The trust-region framing traces to TRPO (Schulman et al., ICML
2015). The base learner is PPO with GAE (Schulman et al., 2016; 2017).

**DELTA's use.** DELTA applies this solution at deployment to a frozen base
policy, with `eta` obtained by deterministic bisection and a stop-gradient
justified by the envelope theorem. The budget `delta` is one of the three
registered method scalars.

**Scope of the claim.** The paper claims the deployment architecture that this
solution enables — a single shared actor whose parameters carry task competence
and whose executed distribution carries belief-dependent adaptation — and not
the solution form itself.

---

## 5. Simulation estimators

**Adopted.** Three classical simulation techniques appear in the decision
channel:

| Technique | Source |
|---|---|
| Common random numbers for contrast estimation | standard simulation practice (Law & Kelton; Glasserman); in RL, fixed-seed policy search (PEGASUS, Ng & Jordan, UAI 2000) |
| Exact marginalization of a finite response space | Bayesian decision analysis |

**DELTA's use.**

CRN makes the anchor estimand affordable. The quantity the decision emission is
identified on is the `A-1` dimensional action-return contrast, and sharing the
random stream across the six forced-action branches reduces the variance of that
contrast by a factor governed by the induced correlation. Four fit replicas
therefore suffice where independent sampling would require substantially more.
The residual covariance of the sample mean is measured empirically and enters
the Gaussian likelihood, so anchors whose branches decoupled are weighted by
their measured precision.

The VOI computation exactly enumerates the registered 66-outcome compact
response marginal. Full direct geometry remains in passive filtering but is
marginalized out of active probing together with recipe response.

**Scope of the claim.** These are numerical choices reported for
reproducibility. The paper claims the estimand they serve — a latent-conditioned
decision emission supervised by measured simulator returns — and reports the
techniques as implementation.

---

## 6. Discrete latent filtering

**Adopted.** The online recursion is the forward algorithm for a discrete-state
hidden Markov model (Rabiner, 1989), with a sticky prior on self-transition in
the spirit of the sticky HDP-HMM (Fox et al., ICML 2008), here in a finite
fixed-`K` form. The analytic behaviour statistics are Beta-Bernoulli conjugate
posteriors.

**DELTA's use.** The marginal likelihood that the filter computes is the
response term of the training objective. Filtering and learning therefore share
one quantity, and the deployed recursion is the same object the model was fit
on.

---

## 7. Value of information

**Adopted.** Valuing an observation by the improvement it produces in the best
achievable decision is information value theory (Howard, 1966), and its use in
sequential Bayesian control follows the Bayes-adaptive literature. In ad-hoc
teamwork, information acquisition has also been studied as a cost-sensitive
planning problem (Macke et al., AAAI 2021); that work uses communication, while
DELTA values legally available observations and continuation returns.

**DELTA's use.** DELTA computes a one-response decision-equivalence VOI over a
structured factorized teammate response, in raw task-return units, and adds it
to the posterior action value before the KL-constrained improvement. Expected
information gain and exact-VOI floating-point diagnostics are emitted.

**Registered acceptance case.** A response that identifies the latent component
while leaving the optimal action unchanged yields information gain approximately
`0.69` and VOI within the registered floating-point tolerance of `0`
(`validation/voi_synthetic_diagnostics.json`). This case fixes the operational
meaning of "decision-relevant" for the paper.

**Scope of the claim.** The registered term is a one-response
decision-equivalence approximation under an explicitly bounded local-stationarity
surrogate. Its scope and error bound are stated in `docs/THEORY.md`.

---

## 8. Recent and contemporary work (2024–2026)

This section locates DELTA relative to work published after the established
lines in sections 1–7. Each entry states what the work is, followed by the
factual boundary with DELTA.

**Benchmark evolution.** OvercookedV2 (Gessler et al., ICLR 2025) shows that
the zero-shot coordination failures reported on classic Overcooked arise from
insufficient state coverage under self-play, and introduces asymmetric
information, stochasticity, and the test-time protocol formation challenge,
with an explicit call for algorithms that adapt online. DELTA's response and
decision channels are a direct response to that call: the deployed agent
forms a belief online and acts on it. OGC (Ruhdorfer et al., TMLR 2025;
arXiv:2406.17949 for preprint traceability) is a generalisation benchmark built
on dual curriculum design;
DELTA's registered claims concern coordination with unseen partners in the two
registered layouts. Collab-Overcooked (Sun et al., EMNLP 2025)
benchmarks LLM coordination in Overcooked-AI around natural language
communication; DELTA evaluates RL agents without a communication channel.

**Recent zero-shot coordination methods.** CEC (Jha et al., ICML 2025,
PMLR 267:27198–27220, arXiv:2504.12714) achieves zero-shot coordination by
training across a distribution of tasks drawn from different environments;
the adaptation is carried by training-time generalization, and no
deployment-time belief is formed. GAMMA (Liang et al., NeurIPS 2024) learns a
generative model of human or neural partners and samples that model to provide
training-time diversity. GOAT (Chaudhary et al., arXiv preprint, 2025) combines
a frozen generative partner model with online adversarial training; it is kept
as a direct preprint source because the paper lists the University of
Washington on its first page. Both methods place partner generation in
training, whereas DELTA forms an explicit belief online at deployment and
enters the policy only through the tilting coefficient. Noisy Zero-Shot
Coordination (Anwar et al., arXiv preprint, 2024) relaxes the common-knowledge
assumption by modifying the training-time setting; its first page lists
Cambridge, Berkeley, MIT, MILA, and Oxford. PACE (Ma et al., ICML 2024) learns a
peer-identification reward that encourages active context-aware exploration;
DELTA's registered probe is a decision-equivalence VOI term under a different
information boundary. ZSC-Eval (Xihuai Wang et al., NeurIPS 2024 Datasets and
Benchmarks) provides an evaluation toolkit and the BR-Prox metric for
zero-shot coordination; DELTA's registered evaluation uses a fixed partner
panel with the registered controls. Role Play (Long et al., Knowledge-Based
Systems 324:113811, 2025) learns role embeddings and predicts partner roles;
DELTA's latent instead represents exchangeable coordination modes identified
by response and measured return-unit evidence.

**Ad-hoc teamwork.** NAHT (Caroline Wang et al., NeurIPS 2024) poses the
`N`-agent ad hoc teamwork problem and learns teammate representations for
policy optimization; the ad-hoc setting assumes joining a pre-existing team,
while DELTA's registered framing is zero-shot coordination with a fresh dyadic
partner. PLASTIC (Barrett et al., *Artificial Intelligence* 2017) selects
policies or models for previously unseen teammates, and Rahman et al. (JMLR
2023) study open ad-hoc teamwork with changing team composition and partial
observability. Macke et al. (AAAI 2021) study the value of information in
ad-hoc planning when communication has a cost. These works establish the
partner-adaptation setting; DELTA differs in using only legal local history and
measured continuation returns to form and evaluate its belief.

**Latent partner modelling (nearest neighbours).** TALENTS
(Li et al., arXiv preprint, 2025) provides a close comparison and
warrants a precise boundary. It learns a VAE latent strategy space from
trajectory data, clusters it into discrete strategy types, conditions a
cooperator on those clusters, and adapts to an unseen partner at deployment by
fixed-share regret minimization over the inferred strategy. Both a discrete
strategy set and deployment-time inference are therefore present in TALENTS,
and the boundary with DELTA rests on the two registered contributions: DELTA
identifies its latent through a single joint likelihood over two observation
channels, one of which is a measured action-return contrast in task-return
units (C1), and DELTA's belief leaves the base policy parameters invariant,
entering the executed policy only through the scalar tilting coefficient (C2). Bayesian delegation
(Wu et al., Topics in Cognitive Science 2021) is the cognitive-science precedent for Bayesian
coordination, inferring subtask assignments among collaborators. Mon-Williams
et al. (NeurIPS 2025), "Partner Modelling Emerges in Recurrent Agents (But
Only When It Matters)", is the closest independent evidence for C3. Working in
Overcooked-AI with plain recurrent agents and no auxiliary objective, they find
that structured partner representations emerge when environmental conditions
make partner modelling useful, especially when agents can influence partner
behaviour through task allocation. This supports the narrower claim that
decision relevance can govern whether partner structure is represented. DELTA
converts that observation into a design: the return-unit channel supplies the
decision relevance as an explicit supervised observation, and the
`response_only` control isolates its contribution under paired seeds. A-ToM (Mu
et al., AAAI 2026, arXiv:2603.16264) and DPT-Agent (Zhang et al., ACL 2025,
arXiv:2502.11882) perform theory-of-mind and dual-process adaptation at the LLM
reasoning layer; DELTA operates a
fixed-capacity discrete posterior with a closed-form KL-constrained
improvement and carries no generative reasoning overhead.

**Hidden-agenda games.** Hidden Agenda (Kopparapu et al., arXiv preprint,
2022) anchors the
social-deduction line; DELTA's trusted-partner setting is on the other side
of that boundary.

---

## 9. What DELTA contributes

Stated in the order the paper should assert them.

**C1 — A latent coordination mode identified by a measured decision channel.**
DELTA supervises a latent variable with sparse all-action common-random-number
continuations drawn from the real simulator, treated as an observation channel
of that variable alongside the teammate response. The two channels enter one
joint marginal likelihood normalized by observation count. This construction
gives the latent components an operational meaning in units of task return.

**C2 — A deployment architecture in which the belief leaves base parameters
invariant.** Task competence is carried entirely by parameters optimized under
PPO. The belief acts only through the tilting coefficient of a closed-form
KL-constrained improvement. This invariance makes the same-world belief
intervention an identified causal test of the belief's decision value.

**C3 — An empirical account of what a prediction-trained latent learns.**
A latent fit to teammate responses alone allocates capacity toward the response
factors that are most predictable. H2 measures how much of the coordination
gain requires the return-unit channel, at fixed `K`, fixed partner distribution,
fixed PPO budget, fixed capacity, paired seeds, and a fixed evaluation panel.

Mon-Williams et al. (section 8) report the observational counterpart of this
premise: partner structure appears in a recurrent hidden state when
environmental conditions make partner modelling useful, particularly when the
agent can influence partner behaviour through task allocation. C3 treats this
as a design principle and H2 tests it under intervention, so the two results
are independent and mutually reinforcing.

---

## 10. How the registered controls isolate the contribution

Each control holds the shared machinery fixed and varies one factor. This is the
evidential basis for the contribution boundary above.

| Control | Shares with DELTA | Varies | Alternative explanation it addresses |
|---|---|---|---|
| `response_only` | filter, transition, behaviour statistics, PPO, partner distribution, evaluator | decision channel | the gain follows from teammate prediction alone |
| `base` | full-frame recurrent reference, PPO, partner distribution, evaluator | latent machinery entirely | task competence accounts for the gain |
| `base_extra` | PPO, partner distribution, evaluator | anchor simulator cost reallocated to PPO interaction | additional simulator budget accounts for the gain |
| `K` in `{2, 4, 8}` | everything | latent capacity | the result depends on a particular capacity choice |

A positive H2 under these controls locates the effect in the decision channel,
since `response_only` carries the same filter, the same transition model, the
same conjugate statistics, the same base learner, and the same partner ecology.

---

## 11. Open scope

Three assumptions remain open and are stated in `docs/SCIENTIFIC_SPEC.md` §8 and
`docs/THEORY.md`:

1. coordination conventions are represented by a finite exchangeable latent of
   registered capacity `K`;
2. a first-order sticky transition suffices over 400-step episodes under view
   radius two, where the teammate is frequently unobserved;
3. the active term evaluates a single next response under a local-stationarity
   surrogate.

---

**Preprint eligibility note.** The recent direct-work preprint exceptions
retained under the institution rule are Anwar et al. (University of Cambridge,
UC Berkeley, MIT, MILA, and the
University of Oxford), Chaudhary et al. (University of Washington), Li et al.
(Carnegie Mellon University, the University of Pittsburgh, the University of
Southern California, and Virginia Tech), and Kopparapu et al. (Harvard
University and DeepMind). These affiliations establish eligibility for
inclusion as preprints; they do not imply peer review or formal publication.

## References

Ordered alphabetically by first author.

- Abdolmaleki, Springenberg, Tassa, Munos, Heess, Riedmiller. Maximum a Posteriori Policy Optimisation. ICLR 2018. arXiv:1806.06920
- Anwar, Pandian, Wan, Krueger, Foerster. Noisy Zero-Shot Coordination: Breaking the Common Knowledge Assumption in Zero-Shot Coordination Games. arXiv preprint arXiv:2411.04976, 2024. https://arxiv.org/abs/2411.04976
- Barrett, Rosenfeld, Kraus, Stone. Making Friends on the Fly: Cooperating with New Teammates. Artificial Intelligence 242:132–171, 2017. doi:10.1016/j.artint.2016.10.005
- Carroll, Shah, Ho, Griffiths, Seshia, Abbeel, Dragan. On the Utility of Learning about Humans for Human-AI Coordination. NeurIPS 2019. arXiv:1910.05789
- Chaudhary, Liang, Chen, Du, Jaques. Improving Human-AI Coordination through Online Adversarial Training and Generative Models. arXiv preprint arXiv:2504.15457, 2025. https://arxiv.org/abs/2504.15457
- Doucet, de Freitas, Murphy, Russell. Rao-Blackwellised Particle Filtering for Dynamic Bayesian Networks. UAI 2000, pp. 176–183.
- Duff, M. O. Optimal Learning: Computational Procedures for Bayes-adaptive Markov Decision Processes. PhD thesis, University of Massachusetts Amherst, 2002.
- Fox, Sudderth, Jordan, Willsky. An HDP-HMM for Systems with State Persistence. ICML 2008, pp. 312–319. doi:10.1145/1390156.1390196
- Gessler, Dizdarevic, Calinescu, Ellis, Lupu, Foerster. OvercookedV2: Rethinking Overcooked for Zero-Shot Coordination. ICLR 2025. arXiv:2503.17821
- Glasserman, P. Gradient Estimation via Perturbation Analysis. Kluwer, 1991.
- Howard, R. A. Information value theory. IEEE Transactions on Systems Science and Cybernetics 2(1):22–26, 1966.
- Hu, Lerer, Peysakhovich, Foerster. "Other-Play" for Zero-Shot Coordination. ICML 2020, PMLR 119:4399–4410. https://proceedings.mlr.press/v119/hu20a.html
- Jha, Carvalho, Liang, Du, Kleiman-Weiner, Jaques. Cross-environment Cooperation Enables Zero-shot Multi-agent Coordination. ICML 2025, PMLR 267:27198–27220. arXiv:2504.12714. https://proceedings.mlr.press/v267/jha25b.html
- Kirk, Zhang, Grefenstette, Rocktäschel. A Survey of Zero-shot Generalisation in Deep Reinforcement Learning. JAIR 76:201–264, 2023. doi:10.1613/jair.1.14174. arXiv:2111.09794
- Kopparapu, Matyas, Duéñez-Guzmán, Vezhnevets, McKee, Everett, Leibo, Agapiou, Marecki, Graepel. Hidden Agenda: A Social Deduction Game with Diverse Learned Equilibria. arXiv preprint arXiv:2201.01816, 2022. https://arxiv.org/abs/2201.01816
- Law, Kelton. Simulation Modeling and Analysis. McGraw-Hill, 3rd ed., 2000.
- Li, Shi, Romero, Li, Xie, Kim, Nikolaidis, Lewis, Sycara, Stepputtis. Modeling Latent Partner Strategies for Adaptive Zero-Shot Human-Agent Collaboration. arXiv preprint arXiv:2507.05244, 2025. https://arxiv.org/abs/2507.05244
- Liang, Chen, Gupta, Du, Jaques. Learning to Cooperate with Humans using Generative Agents. NeurIPS 2024, pp. 60061–60087. doi:10.52202/079017-1918. arXiv:2411.13934
- Long, Wen, Zhai, Zhang. Role Play: Learning Adaptive Role-Specific Strategies in Multi-Agent Interactions. Knowledge-Based Systems 324:113811, 2025. doi:10.1016/j.knosys.2025.113811. arXiv:2411.01166
- Ma, Wang, Zhong, Zhu, Wang. Fast Peer Adaptation with Context-aware Exploration. ICML 2024, PMLR 235:33963–33982. https://proceedings.mlr.press/v235/ma24n.html
- Macke, Mirsky, Stone. Expected Value of Communication for Planning in Ad Hoc Teamwork. AAAI 2021, 35(13):11290–11298. doi:10.1609/aaai.v35i13.17346
- Mon-Williams, Taylor-Davies, Mieczkowski, Vélez, Bramley, Wang, Griffiths, Lucas. Partner Modelling Emerges in Recurrent Agents (But Only When It Matters). NeurIPS 2025. arXiv:2505.17323
- Mu et al. Adaptive Theory of Mind for LLM-based Multi-Agent Coordination. AAAI 2026, 40(35):29608–29616. doi:10.1609/aaai.v40i35.40204. arXiv:2603.16264
- Ng, Jordan. PEGASUS: A Policy Search Method for Large MDPs and POMDPs. UAI 2000, pp. 406–415.
- Papoudakis, Christianos, Albrecht. Agent Modelling under Partial Observability for Deep Reinforcement Learning. NeurIPS 2021, pp. 19210–19222. arXiv:2006.09447
- Peters, Mülling, Altun. Relative Entropy Policy Search. AAAI 2010, 24(1):1607–1612. doi:10.1609/aaai.v24i1.7727
- Rabiner, L. R. A tutorial on hidden Markov models and selected applications in speech recognition. Proceedings of the IEEE 77(2):257–286, 1989.
- Rabinowitz, Perbet, Song, Zhang, Eslami, Botvinick. Machine Theory of Mind. ICML 2018, PMLR 80:4218–4227. https://proceedings.mlr.press/v80/rabinowitz18a.html
- Rahman, Carlucho, Höpner, Albrecht. A General Learning Framework for Open Ad Hoc Teamwork Using Graph-based Policy Learning. JMLR 24(298):1–74, 2023. https://jmlr.org/papers/v24/22-099.html
- Rakelly, Zhou, Finn, Levine, Quillen. Efficient Off-Policy Meta-Reinforcement Learning via Probabilistic Context Variables. ICML 2019, PMLR 97:5331–5340. https://proceedings.mlr.press/v97/rakelly19a.html
- Ruhdorfer, Bortoletto, Penzkofer, Bulling. The Overcooked Generalisation Challenge: Evaluating Cooperation with Novel Partners in Unknown Environments Using Unsupervised Environment Design. Transactions on Machine Learning Research, 2025, 1–25. arXiv:2406.17949
- Schulman, Levine, Abbeel, Jordan, Moritz. Trust Region Policy Optimization. ICML 2015, PMLR 37:1889–1897. https://proceedings.mlr.press/v37/schulman15.html
- Schulman et al. High-Dimensional Continuous Control Using Generalized Advantage Estimation. ICLR 2016. arXiv:1506.02438
- Schulman et al. Proximal Policy Optimization Algorithms. arXiv preprint arXiv:1707.06347, 2017 (canonical original). https://arxiv.org/abs/1707.06347
- Strouse, McKee, Botvinick, Hughes, Everett. Collaborating with Humans without Human Data. NeurIPS 2021, pp. 14502–14515. arXiv:2110.08176
- Sun, Zhang, Niu, Ren, Xu, Fu, Zhao, Yuan, Wang. Collab-Overcooked: Benchmarking and Evaluating Large Language Models as Collaborative Agents. EMNLP 2025, pp. 4922–4951. doi:10.18653/v1/2025.emnlp-main.249. arXiv:2502.20073
- Wang, Rahman, Durugkar, Liebman, Stone. N-Agent Ad Hoc Teamwork. NeurIPS 2024 Main Conference Track. doi:10.52202/079017-3551. arXiv:2404.10740
- Wang, Zhang, Zhang, Dong, Chen, Wen, Zhang. ZSC-Eval: An Evaluation Toolkit and Benchmark for Multi-agent Zero-shot Coordination. NeurIPS 2024 Datasets and Benchmarks Track. arXiv:2310.05208
- Wu, Wang, Evans, Tenenbaum, Parkes, Kleiman-Weiner. Too Many Cooks: Bayesian Inference for Coordinating Multi-Agent Collaboration. Topics in Cognitive Science 13(2):414–432, 2021. doi:10.1111/tops.12525
- Zhang et al. Leveraging Dual Process Theory in Language Agent Framework for Real-time Simultaneous Human-AI Collaboration. ACL 2025 Main Long Papers, pp. 4081–4108. doi:10.18653/v1/2025.acl-long.206. arXiv:2502.11882
- Zintgraf et al. VariBAD: A Very Good Method for Bayes-Adaptive Deep RL via Meta-Learning. ICLR 2020. arXiv:1910.08348
