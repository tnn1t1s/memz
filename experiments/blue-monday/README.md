# Blue Monday

**Difficulty: 3/5 -- Knowledge correction**

Can the adapter overwrite previously learned facts when given contradictory
updates, without confusing old and new versions?

## Hypothesis

After learning 12 facts about a fictional city, a correction batch will update
6 of those facts with new values (mayor changed, population updated, etc.).
The model should return the *corrected* answers, not the originals. This tests
whether LoRA updates are additive-only or can genuinely revise prior knowledge.

## Design

- Phase 1 (`phase_01_initial.jsonl`): 12 facts about the city of Ostara
- Phase 2 (`phase_02_corrections.jsonl`): 6 corrections to phase 1 facts,
  plus 4 new facts (total 10 examples)
- Eval checks all 16 unique facts; 6 of them must reflect the corrected value

## Evaluation

| Metric | Pass threshold |
|--------|---------------|
| Corrected fact recall (new value) | >= 5/6 |
| Uncorrected fact recall (original) | >= 5/6 |
| New fact recall (phase 2 only) | >= 3/4 |
| Old-value leakage | 0 (must not return superseded answers) |

## Key question

When probed with the exact same input as phase 1, does the model produce the
phase 2 (corrected) answer? Any blending or hedging ("it was X but is now Y")
counts as partial failure.
