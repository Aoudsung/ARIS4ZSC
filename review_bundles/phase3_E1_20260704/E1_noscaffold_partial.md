# E1(no-scaffold) partial result — orchestrator data-loss bug

Config: ocv2_step4_asymm_role_v2_e1.yaml (contrib_team, scaffolds OFF)
Bug: E1_wave_orch.sh shared output dir E1__s (bash `local` gotcha) + rm -rf per run
     → cross-wiped. Only partner_id_q/seed4 survived on disk. Rest from live reads.

## Training-phase ego/partner correct-delivery counts (captured before wipe)
| method       | s0    | s1    | s2    | s3    | s4    |
|--------------|-------|-------|-------|-------|-------|
| aris_bellman | 0/8   | 1/12  | 0/10  | 0/11  | 0/6   |
| base_only    | 0/7   | -     | -     | -     | -     |
| global_gru   | -     | -     | -     | -     | -     |
| flat_factor  | -     | -     | -     | -     | -     |
| partner_id_q | -     | -     | -     | -     | 1/9   |

Headline: ego≈0 across all captured runs (no-scaffold vacuum, sec18.12.1 trigger).
Guard: fail across arms; no deployable checkpoints → nothing read against sec18.6.
Status: full 4-arm×5-seed ablation table LOST; re-run deferred per sec18.12.4.
