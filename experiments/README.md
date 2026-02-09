# Experiments

Memory stress tests for continuous LoRA fine-tuning, ordered by difficulty.
Each experiment lives in its own directory with training data (JSONL),
evaluation prompts, and a config override. Named after New Order songs.

## Compendium

### 1. [ceremony](ceremony/) -- Trivial recall

Can the adapter memorize five facts from a single update and repeat them back?

- **Domain**: Fictional research lab (Vellum Labs)
- **Examples**: 5
- **Batches**: 1
- **Tests**: Verbatim recall of short factoid pairs
- **Pass criteria**: 5/5 exact-match recall, finetune loss < 0.5

The sanity check. If this fails, nothing else will work.

---

### 2. [age-of-consent](age-of-consent/) -- Incremental accumulation

Do earlier batches survive when later batches are trained on top?

- **Domain**: Fictional pharma company (Auralis Therapeutics)
- **Examples**: 24 (3 batches of 8)
- **Batches**: 3 (founding, pipeline, financials)
- **Tests**: Per-batch recall after all three batches complete
- **Pass criteria**: >= 7/8 recall per batch, forgetting delta < 0.10

The key measurement is batch 1 recall after batches 2 and 3 have trained.
Any decay reveals the forgetting gradient.

---

### 3. [blue-monday](blue-monday/) -- Knowledge correction

Can the adapter overwrite previously learned facts with updated values?

- **Domain**: Fictional city (Ostara)
- **Examples**: 22 (12 initial + 10 corrections/additions)
- **Phases**: 2 (initial facts, then corrections)
- **Tests**: Corrected facts return new values; uncorrected facts survive
- **Pass criteria**: >= 5/6 corrected recall, zero old-value leakage

The hardest part is not blending old and new answers. Any hedging
("it was X but is now Y") counts as partial failure.

---

### 4. [bizarre-love-triangle](bizarre-love-triangle/) -- Multi-domain interference

Does training on domain B erase what was learned about domain A?

- **Domains**: Botany, aviation, culinary, astronomy (8 facts each)
- **Examples**: 40 (16 + 16 + 8 reinforcement)
- **Batches**: 3 (two domain pairs, then mixed reinforcement)
- **Tests**: Per-domain recall; cross-domain confusion detection
- **Pass criteria**: >= 6/8 per domain, forgetting delta < 0.20

Auto-rollback is enabled. The reinforcement batch (batch 3) replays
two facts from each domain to probe whether targeted replay recovers
any drift introduced by batch 2.

---

### 5. [the-perfect-kiss](the-perfect-kiss/) -- Full lifecycle stress test

Everything at once: inject, expand, correct, domain-shift, reinforce.

- **Domains**: Fictional programming language (Luma) + hydrothermal vents
- **Examples**: 52 across 5 phases
- **Phases**:
  1. Foundation (12 Luma facts)
  2. Expansion (12 more Luma facts)
  3. Correction (6 Luma 2.0 changes + 2 new facts)
  4. Domain shift (12 hydrothermal vent facts)
  5. Mixed reinforcement (8 cherry-picked from all phases)
- **Tests**: Retained, corrected, new, and domain-shifted fact recall
- **Pass criteria**: >= 80% total recall (40/50), zero old-value leakage,
  forgetting delta < 0.25

The ultimate question: after a domain shift in phase 4, what fraction
of Luma knowledge survives? And does phase 5 reinforcement recover it?

## Running an experiment

```bash
# start the server with experiment config
uv run memz serve --config experiments/<name>/config.yaml

# post training data
curl -s localhost:8042/update -H 'Content-Type: application/json' \
  -d '{"examples": [...]}'

# check status
curl -s localhost:8042/info | python -m json.tool

# evaluate
curl -s localhost:8042/search -H 'Content-Type: application/json' \
  -d '{"prompt": "..."}'
```

## Directory structure

Each experiment contains:

| File | Purpose |
|------|---------|
| `README.md` | Hypothesis, design, pass criteria |
| `*.jsonl` | Training examples (one JSON object per line) |
| `eval.jsonl` | Evaluation prompts with expected substrings |
| `config.yaml` | Training/evaluation parameter overrides |
