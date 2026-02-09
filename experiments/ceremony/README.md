# Ceremony

**Difficulty: 1/5 -- Trivial recall**

Can the adapter memorize a handful of direct factual pairs from a single
update batch and reproduce them on the next search?

## Hypothesis

A single LoRA update with 5 short input/output pairs should achieve near-perfect
verbatim recall. This is the baseline sanity check: if this fails, nothing else
will work.

## Design

- One POST to `/update` with 5 factoid pairs about a fictional research lab
- Wait for training job to complete
- Query each fact via `/search`

## Evaluation

| Metric | Pass threshold |
|--------|---------------|
| Exact-match recall | 5/5 |
| Finetune loss | < 0.5 |
| Forgetting delta | < 0.05 |

## Run

```bash
# single update
curl -s localhost:8042/update -H 'Content-Type: application/json' \
  -d @examples.jsonl.body

# after job completes, eval
python run_eval.py
```
