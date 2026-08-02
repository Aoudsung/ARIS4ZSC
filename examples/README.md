# Manifest examples are not formal experiment manifests

`partner_manifest_plan.template.json` is a small schema example for exercising
the manifest builder. It is intentionally incomplete and must never be used
for a formal DELTA run.

A formal manifest is generated separately for each layout and must bind, for
each DELTA outer seed index `0..9`:

- one independent owner-SP source checkpoint;
- one independent generator-initialization run from each of SP, OP,
  State-Augmented and FCP;
- four independent development/support runs from each of those mechanisms;
- five independent calibration runs from each mechanism (20 blocks total);
- the exact checkpoint SHA-256, parent-run lineage, generation mechanism, and
  real JAX key derived from `split(PRNGKey(42), 10)`.

Plan schema version 2 requires the two-word `jax_prng_key` explicitly. The
builder never derives or substitutes an unregistered seed. Owner and upstream
sources must use their registered keys; development, calibration and common-panel
partners must record independent frozen keys and lineage.

The Common-Partner Scoreboard uses another manifest. It contains exactly four
fresh parent runs from each of `sp`, `state-augmented`, `op`, and `fcp`; none may
overlap any method's training, generator snapshot, anchor, population, or
calibration lineage.

Official FCP additionally requires an explicit lineage file covering all
`10 populations × 8 SP parents × 3 checkpoints = 240` checkpoint artifacts and
the complete population resource ledger. The fixed paper/source does not
register a substitute set of inner population seeds, so this repository never
invents them.
