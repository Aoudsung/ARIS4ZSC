# latent_v3 Phenomenon-Existence Certificate

Status: **FAIL**
Commit: `5f9ce3a-r2gpu`

| Check | Value | Threshold | Pass |
|---|---:|---|---|
| C-1 unsaturation | 0.947535 | NOHIST blind AUC <= 0.80 | False |
| C-2 signal exists and needs history | gain=0.0198811, ci_lo=0.0179631 | G_blind >= 0.10 and gain_ci.lo > 0 | False |
| C-3 identity beyond state | 0.0141275 | IDORACLE - NOHIST indist >= 0.05 | False |
| C-4 identifiability | full=0.905175, state=0.856982 | history_full >= 0.75 and state_only <= 0.60 | False |
| C-5 value of information | 0.0332142 | pooled mode_oracle relative lift >= 0.15 vs best single blind ego (fullchain/prepchain/reactive) | False |
| C-6 wiring | oracle=0, golden_families=5 | oracle_source_count=0, evidence_policy=behavior_inferred_v1, golden PASS per family | False |

Artifacts:
- D1 readout: `results_linka_v3r2/d1/readout_frozen.json`
- D1 tune: `results_linka_v3r2/d1/tune_report.json`
- Mode cert: `results_linka_v3r2/mode_cert.json`
- VOI: `results_linka_v3r2/voi.json`
- Chunks: `results_linka_v3r2/chunks`
