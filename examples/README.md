# Unified DELTA partner-manifest plan

`partner_manifest_plan.template.json` is a human-editable input to
`build-partner-manifest`; it is not itself a scientific manifest. Replace every
checkpoint placeholder with a real file or directory. The builder resolves
paths, computes deterministic SHA-256 hashes, writes manifest schema v1, and
immediately reloads it through the active lineage validator.

```bash
python -m experiments.overcooked_v2.delta_zsc build-partner-manifest \
  --plan examples/partner_manifest_plan.template.json \
  --expected-layout test_time_simple \
  --output /ABSOLUTE/PATH/partner_manifest.json

python -m experiments.overcooked_v2.delta_zsc validate-partner-manifest \
  --partner-manifest /ABSOLUTE/PATH/partner_manifest.json \
  --expected-layout test_time_simple
```

Roles used by the active method are `development_support`, `calibration`, and
`confirmatory`. Their parent and co-training lineages must be mutually disjoint.
Formal training support contains independent SP and OP parents with progress
stages 0, 0.5 and 1.0. Partner metadata is used for sampling and statistics only;
it never enters the ego policy.
