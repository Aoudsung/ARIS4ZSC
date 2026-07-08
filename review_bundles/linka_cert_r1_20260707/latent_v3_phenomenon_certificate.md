# latent_v3 Phenomenon-Existence Certificate

Status: **FAIL**
Commit: `678247e`

| Check | Value | Threshold | Pass |
|---|---:|---|---|
| C-1 unsaturation | 0.966438 | NOHIST blind AUC <= 0.80 | False |
| C-2 signal exists and needs history | gain=0.0115055, ci_lo=0.00901589 | G_blind >= 0.10 and gain_ci.lo > 0 | False |
| C-3 identity beyond state | 0.0158069 | IDORACLE - NOHIST indist >= 0.05 | False |
| C-4 identifiability | full=0.895863, state=0.848613 | history_full >= 0.75 and state_only <= 0.60 | False |
| C-5 value of information | 0 | mode_oracle relative lift >= 0.15 vs best blind scripted ego | False |
| C-6 wiring | oracle=0, golden_families=5 | oracle_source_count=0, evidence_policy=behavior_inferred_v1, golden PASS per family | False |

Artifacts:
- D1 readout: `results_linka_latent_v3_20260707/d1/readout_frozen.json`
- D1 tune: `results_linka_latent_v3_20260707/d1/tune_report.json`
- Mode cert: `results_linka_latent_v3_20260707/mode_cert.json`
- VOI: `results_linka_latent_v3_20260707/voi.json`
- Chunks: `results_linka_latent_v3_20260707/chunks`
