# Main Pipeline

This folder contains the core document + query preprocessing stages for the TareRAG pipeline. The scripts convert documents and queries into typed triples that later feed matching and answer generation.

## How the pipeline works (brief)

Document side:
1) `convert_triple.py` extracts raw triples from each retrieved context (`ctxs`) using the LLM.
2) `typed_triple.py` assigns L1/L2 types to entities in those triples using rule shortcuts, FAISS type retrieval (from `type_faiss/`), embeddings, and an LLM selector.

Query side:
1) `query_decompose.py` judges whether a question is decomposable and emits atomic subqueries.
2) `query_triple.py` turns each subquery into a single triple with variables for unknowns.
3) `query_typed.py` types the query triples using the same L1/L2 typing approach as the doc side.

Optional multi-hop answering:
- `query_stepbystep.py` consumes `query_typed` + `typed_triple` outputs, matches each subquery in order, answers it, substitutes resolved variables into later triples, and finally synthesizes the original answer.

## Inputs and outputs (per script)

- `convert_triple.py`: input JSONL with `question`, `answers`/`answer`, `ctxs[]`; output JSONL with `triples[]` (per-doc raw triples).
- `typed_triple.py`: input JSONL from `convert_triple.py`; output JSONL with `typed_triple`, `type_only_triple`, and `entity2type`.
- `query_decompose.py`: input JSONL with `id`, `question`, `answers`/`answer`; output JSONL with `decomposable` and `sub_queries`.
- `query_triple.py`: input JSONL from `query_decompose.py`; output JSONL with per-subquery `raw_triple`.
- `query_typed.py`: input JSONL from `query_triple.py`; output JSONL with typed query triples + `entity2type`.
- `query_stepbystep.py`: input files from `query_typed.py`, `typed_triple.py`, and original data; output JSONL with stepwise answers and final answer.

## Typical run order

```bash
# document side
python absQA2/absQA/main_pipline/convert_triple.py \
  --input_file ../../eval_data_ctxs/nq_sampled_200.jsonl \
  --output_file results/nq_sampled_200/convert_triple_result.jsonl

python absQA2/absQA/main_pipline/typed_triple.py \
  --input_file results/nq_sampled_200/convert_triple_result.jsonl \
  --output_file results/nq_sampled_200/typed_triple_result.jsonl \
  --faiss_dir type_faiss

# query side
python absQA2/absQA/main_pipline/query_decompose.py \
  --input_file ../../eval_data_ctxs/nq_sampled_200.jsonl \
  --output_file results/nq_sampled_200/query_decompose_result.jsonl

python absQA2/absQA/main_pipline/query_triple.py \
  --input_file results/nq_sampled_200/query_decompose_result.jsonl \
  --output_file results/nq_sampled_200/query_triple_result.jsonl

python absQA2/absQA/main_pipline/query_typed.py \
  --input_file results/nq_sampled_200/query_triple_result.jsonl \
  --output_file results/nq_sampled_200/query_typed_result.jsonl \
  --faiss_dir type_faiss
```

Step-by-step answering (optional):

```bash
python absQA2/absQA/main_pipline/query_stepbystep.py \
  --query_file results/nq_sampled_200/query_typed_result.jsonl \
  --doc_file results/nq_sampled_200/typed_triple_result.jsonl \
  --data_file ../../eval_data_ctxs/nq_sampled_200.jsonl \
  --output_file results/nq_sampled_200/stepbystep_result.jsonl
```
