# Matching Module

This directory contains the triple-matching and document ranking logic used in the step-by-step answering stage.

## `matching_mean.py`

The default matching strategy. For each document, scores are computed per (query triple, doc triple) pair using a combination of typed score and embedding score:

```
score = alpha_type * typed_score + (1 - alpha_type) * embedding_score
```

The document score is the mean of all per-triple scores above a threshold. Invoke directly as a CLI to re-rank documents without running the full answering pipeline:

```bash
python core/matching/matching_mean.py \
  --query_file $OUT/query_typed_result.jsonl \
  --doc_file   $OUT/typed_triple_result.jsonl \
  --data_file  $DATA \
  --output_file $OUT/doc_result_top10.jsonl \
  --alpha_type 0.5 \
  --threshold 0.3
```
