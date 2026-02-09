# Age of Consent

**Difficulty: 2/5 -- Incremental accumulation**

Can the adapter accumulate knowledge across three successive update batches
without losing facts from earlier batches?

## Hypothesis

Three sequential batches of 8 facts each (24 total), all within the same domain
(fictional pharmaceutical company), should be retained with high accuracy after
the final training run. Earlier batches are at risk of partial overwrite by
later ones.

## Design

- Batch 1 (`batch_01.jsonl`): company founding, leadership, HQ location
- Batch 2 (`batch_02.jsonl`): product pipeline, clinical trials
- Batch 3 (`batch_03.jsonl`): financials, partnerships, recent news
- POST each batch to `/update` sequentially, waiting for each job to finish
- Eval covers all 24 facts

## Evaluation

| Metric | Pass threshold |
|--------|---------------|
| Batch 1 recall | >= 7/8 |
| Batch 2 recall | >= 7/8 |
| Batch 3 recall | >= 8/8 |
| Forgetting delta | < 0.10 |

## Key question

Does recall of batch 1 facts degrade after batches 2 and 3 are trained?
Measure per-batch recall to quantify the forgetting gradient.
