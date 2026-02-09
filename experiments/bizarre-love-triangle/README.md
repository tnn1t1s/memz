# Bizarre Love Triangle

**Difficulty: 4/5 -- Multi-domain interference**

Can the adapter learn facts across four unrelated knowledge domains
simultaneously without catastrophic forgetting?

## Hypothesis

Interleaved updates across four domains (botany, aviation, culinary, astronomy)
create competing gradient signals. The adapter must retain facts from all
domains after the final update. This is where catastrophic forgetting typically
surfaces: domain B training erodes domain A representations.

## Design

Four domains, 8 facts each (32 total), delivered as interleaved batches:

- Batch 1: botany (8) + aviation (8) = 16 examples
- Batch 2: culinary (8) + astronomy (8) = 16 examples
- Batch 3: mixed recall reinforcement, 2 from each domain = 8 examples

Total: 40 examples across 3 update calls.

Forgetting monitoring is enabled with auto-rollback on catastrophic delta.

## Evaluation

| Metric | Pass threshold |
|--------|---------------|
| Botany recall | >= 6/8 |
| Aviation recall | >= 6/8 |
| Culinary recall | >= 7/8 |
| Astronomy recall | >= 7/8 |
| Cross-domain confusion | 0 (no domain-bleed answers) |
| Forgetting delta | < 0.20 |

## Key question

After batch 2 trains culinary + astronomy, how much do botany + aviation scores
drop compared to their post-batch-1 baseline? The reinforcement batch 3 attempts
to recover any lost ground, but the recovery ceiling is the real finding.
