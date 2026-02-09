# The Perfect Kiss

**Difficulty: 5/5 -- Full lifecycle stress test**

A five-phase gauntlet that simulates realistic continuous learning: initial
knowledge injection, expansion, correction, domain shift, and a long-delay
consistency recheck. This is the end-to-end torture test for memory.

## Hypothesis

Over 5 sequential phases totaling 60+ training examples, the adapter will
experience compounding pressure from corrections, domain shifts, and sheer
volume. The final eval covers facts from every phase, with a bias toward
early-phase facts that have had the most time to decay. A system that passes
this experiment has a genuinely functional continuous memory.

## Design

### Phase 1: Foundation (12 examples)
`phase_01_foundation.jsonl` -- Core facts about a fictional programming
language called Luma: syntax, type system, concurrency model.

### Phase 2: Expansion (12 examples)
`phase_02_expansion.jsonl` -- Deeper Luma knowledge: standard library,
toolchain, ecosystem, performance characteristics.

### Phase 3: Correction (8 examples)
`phase_03_corrections.jsonl` -- Luma 2.0 release changes 6 facts from
phases 1-2 (syntax changes, deprecations, new defaults). Plus 2 new facts.

### Phase 4: Domain shift (12 examples)
`phase_04_domain_shift.jsonl` -- Completely unrelated domain: facts about
deep-sea hydrothermal vent ecosystems. Tests whether a domain shift
obliterates the Luma knowledge.

### Phase 5: Mixed reinforcement (8 examples)
`phase_05_reinforcement.jsonl` -- Cherry-picked facts from all prior phases
to probe whether targeted replay can recover any drift.

Total: 52 training examples across 5 update calls.

## Evaluation

| Metric | Pass threshold |
|--------|---------------|
| Phase 1 retained facts (uncorrected) | >= 5/6 |
| Phase 2 retained facts (uncorrected) | >= 5/6 |
| Phase 3 corrected facts (new values) | >= 5/6 |
| Phase 4 domain-shift facts | >= 10/12 |
| Old-value leakage | 0 |
| Cross-domain confusion | 0 |
| Forgetting delta (anchor) | < 0.25 |
| Total recall | >= 40/50 (80%) |

## Key questions

1. After the domain shift in phase 4, what fraction of Luma facts survive?
2. Do corrected facts revert to their phase-1 values under pressure?
3. Does phase 5 reinforcement measurably recover lost recall?
4. Is the forgetting delta monotonically increasing, or does it plateau?
