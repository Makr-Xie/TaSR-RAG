# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TaSR-RAG is a **Type-aware Structured Retrieval** pipeline for multi-hop QA. It converts documents and queries into *typed triples* (subject/predicate/object with hierarchical L1/L2 entity types), then ranks documents by matching query triples against document triples using a combined typed-score + embedding-score.

## Running the Pipeline

All scripts are run directly as Python scripts from the repo root. There is no package install step. Scripts resolve imports via a `sys.path` hack that adds `core/` to the path.

**Document side (run in order):**
```bash
python core/main_pipline/convert_triple.py \
  --input_file <data>.jsonl --output_file <out>/convert_triple_result.jsonl

python core/main_pipline/typed_triple.py \
  --input_file <out>/convert_triple_result.jsonl \
  --output_file <out>/typed_triple_result.jsonl \
  --faiss_dir type_faiss
```

**Query side (run in order):**
```bash
python core/main_pipline/query_decompose.py \
  --input_file <data>.jsonl --output_file <out>/query_decompose_result.jsonl

python core/main_pipline/query_triple.py \
  --input_file <out>/query_decompose_result.jsonl \
  --output_file <out>/query_triple_result.jsonl

python core/main_pipline/query_typed.py \
  --input_file <out>/query_triple_result.jsonl \
  --output_file <out>/query_typed_result.jsonl \
  --faiss_dir type_faiss
```

**Step-by-step answering:**
```bash
python core/main_pipline/query_stepbystep.py \
  --query_file <out>/query_typed_result.jsonl \
  --doc_file <out>/typed_triple_result.jsonl \
  --data_file <data>.jsonl \
  --output_file <out>/stepbystep_result.jsonl
```

**Matching only (re-rank without answering):**
```bash
python core/matching/matching_mean.py \
  --query_file <out>/query_typed_result.jsonl \
  --doc_file <out>/typed_triple_result.jsonl \
  --data_file <data>.jsonl \
  --output_file <out>/doc_result_top10.jsonl
```

**Evaluation:**
```bash
python core/evaluate.py --input_file <out>/stepbystep_result.jsonl
```

**Build the FAISS type index** (needed once before running typed_triple or query_typed):
```bash
python core/build_faiss_type_index.py --output_dir type_faiss --embed_url http://localhost:12261/v1
```

**Common script flags** (most scripts share these):
- `--workers N` — parallel threads (default 16)
- `--sample N` — process only first N records
- `--start N` — skip first N records (for resuming)
- `--cache_file path.json` — persist entity→type cache across runs

## Configuration

All API endpoints and model names live in [core/config.py](core/config.py):

```python
CHAT_API_BASE = "http://localhost:1225/v1"   # vLLM chat server
EMBED_API_BASE = "http://localhost:12261/v1"  # vLLM embedding server
CHAT_MODEL = "Qwen2.5-72B-Instruct"
EMBED_MODEL = "Qwen3-Embedding-8B"
```

Both servers must be running before executing any pipeline step. The chat model must be served with OpenAI-compatible vLLM. `typed_triple.py` monkey-patches `VLLMClients.chat_batch` at runtime to strip Qwen3 `<think>…</think>` tags.

## Architecture

### Import resolution
Every script in `core/main_pipline/` and `core/matching/` inserts `core/` into `sys.path` at the top so it can do bare `from config import …` and `from utils import …`. This means scripts must be run with a working directory where `core/` is a direct child, or the path must be adjusted.

### Type taxonomy (L1/L2)
The entity type system is a two-level hierarchy defined in [core/build_faiss_type_index.py](core/build_faiss_type_index.py): L1 categories (PERSON, ORGANIZATION, LOCATION, FACILITY, EVENT, WORK, QUANTITY, TIME, …) each with a list of L2 subtypes. The FAISS indexes (`type_faiss/l1.index`, `type_faiss/l2_<L1>.index`) are built from this taxonomy and used at runtime by `FaissTypeRetriever` in [core/utils/llm_client.py](core/utils/llm_client.py).

### Typing pipeline (core of the system)
`type_triples_batch()` in [core/main_pipline/typed_triple.py](core/main_pipline/typed_triple.py) is the central function. For each entity in a triple it:
1. Applies regex rules first (years, dates, percentages → skip LLM)
2. Checks an optional in-memory `EntityTypeCache`
3. Embeds remaining entities, retrieves top-K L1 candidates via FAISS
4. Uses LLM to pick top-3 L1 candidates
5. For each L1 candidate, retrieves top-K L2 candidates via per-L1 FAISS indexes
6. Uses LLM to select the final (L1, L2) pair

`query_typed.py` and its `_noembed` variant run the same logic on query triples. The `_noembed` variants skip embedding and use only LLM+rules (ablation).

### Matching scoring
`rank_docs_by_triple_matching()` in [core/matching/matching_mean.py](core/matching/matching_mean.py) computes a per-triple score:

```
score = alpha_type * typed_score + (1 - alpha_type) * emb_score
```

- `typed_score`: L1 match (weight 0.4) + L2 match (weight 0.6) for subject and object
- `emb_score`: cosine similarity of S/P/O embeddings independently weighted (0.3/0.4/0.3)
- Document score: mean of all per-triple scores above `threshold` (default 0.3)
- `--alpha_type 0.5` is the default blend; `--with_retrieval` multiplies by the original retrieval score

### LLM clients
[core/utils/llm_client.py](core/utils/llm_client.py) has two clients:
- `VLLMClients` — batched async (httpx) chat + parallel threaded embedding, used by the typing stages
- `simple_call_vllm()` — single sync call, used by decomposition and answer generation

[core/utils/text_tools.py](core/utils/text_tools.py) handles LLM output parsing: stripping Qwen3 think-tags, extracting JSON objects robustly (handles nested braces, falls back to `ast.literal_eval`).

### Ablation variants
`typed_triple_noembed.py` and `query_typed_noembed.py` skip the embedding step and retrieve type candidates via text-only FAISS (ablation for measuring embedding contribution).

## Known cleanup items (from core/README.md)
- `main_pipline` should be renamed to `pipeline` once behavior is frozen
- Replace ad-hoc `sys.path` manipulation with proper package-relative imports
- Move hard-coded model settings out of `config.py` into env vars or a config file
- Consolidate the `*_noembed` script variants into parameterized flags on the main scripts
