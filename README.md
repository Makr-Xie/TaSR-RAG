# TaSR-RAG

**Type-aware Structured Retrieval for Multi-hop RAG.**

📄 Paper: [https://arxiv.org/abs/2603.09341](https://arxiv.org/abs/2603.09341)

TaSR-RAG converts documents and queries into typed triples — structured as `(subject, predicate, object)` with hierarchical entity types (L1/L2) — and ranks documents by matching query triples against document triples using a combined typed-score and embedding-score. This enables precise, interpretable multi-hop retrieval without dense retrieval fine-tuning.

---

## Requirements

**Python packages** (install with pip):
```
openai
httpx
numpy
faiss-cpu
tqdm
```

**Model serving**: Two vLLM servers must be running before executing any pipeline step:
- A chat/LLM server (tested with Qwen2.5-72B-Instruct)
- An embedding server (tested with Qwen3-Embedding-8B)

No other environment setup is required. All pipeline steps call the models via OpenAI-compatible API.

---

## Configuration

All endpoints and model names are in [`core/config.py`](core/config.py):

```python
CHAT_API_BASE = "http://localhost:1225/v1"    # vLLM chat server
EMBED_API_BASE = "http://localhost:12261/v1"  # vLLM embedding server
CHAT_MODEL    = "Qwen2.5-72B-Instruct"
EMBED_MODEL   = "Qwen3-Embedding-8B"
```

Edit this file to point to your own servers before running.

---

## Step 0 — Serve the models

Start both servers in separate terminals. Adjust `CUDA_VISIBLE_DEVICES` and `--tensor-parallel-size` to match your hardware.

**Chat model** (example: 72B on 4 × A6000, port 1225):
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \
  --model /path/to/Qwen2.5-72B-Instruct \
  --tensor-parallel-size 4 \
  --port 1225 \
  --gpu-memory-utilization 0.92 \
  --max-model-len 16384 \
  --served-model-name Qwen2.5-72B-Instruct
```

**Embedding model** (example: 8B on 1 × GPU, port 12261):
```bash
CUDA_VISIBLE_DEVICES=4 python -m vllm.entrypoints.openai.api_server \
  --model /path/to/Qwen3-Embedding-8B \
  --tensor-parallel-size 1 \
  --port 12261 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --served-model-name Qwen3-Embedding-8B \
  --task embed
```

Wait for `Application startup complete` in both terminals before proceeding.

**Tip — smaller GPUs**: `--tensor-parallel-size` must divide the model's attention head count evenly (Qwen2.5-72B has 64 heads → use 1, 2, 4, or 8). A smaller model like Qwen2.5-7B-Instruct works as a drop-in replacement for pipeline testing; use `--served-model-name Qwen2.5-72B-Instruct` to keep `config.py` unchanged.

---

## Step 1 — Build the FAISS type index

This builds the L1/L2 entity type taxonomy indexes used by both typing stages. Run once; output is saved to `type_faiss/`.

```bash
cd /path/to/TaSR-RAG

python core/build_faiss_type_index.py \
  --base_url http://localhost:12261/v1 \
  --embed_model Qwen3-Embedding-8B \
  --out_dir type_faiss
```

---

## Step 2 — Prepare input data

Input files must be JSONL, one record per line, with this schema:
```json
{
  "id": "unique_id",
  "question": "Multi-hop question text",
  "answers": ["gold answer 1", "..."],
  "ctxs": [
    {"title": "Document title", "doc": "Document body text", "score": 0.85},
    ...
  ]
}
```

`ctxs` is a ranked list of retrieved documents (e.g., from BM25 or DPR). The pipeline uses the top-10 by default.

A 10-record sample is provided at [`data/2wikimQA_10.jsonl`](data/2wikimQA_10.jsonl) for quick testing.

---

## Step 3 — Run the pipeline

Set these variables once, then run each step in order.

```bash
cd /path/to/TaSR-RAG
DATA=data/2wikimQA_10.jsonl   # your input file
OUT=runs/my_experiment         # output directory
mkdir -p $OUT
```

### Doc side (extract and type triples from retrieved documents)

```bash
# D1: extract raw triples from documents
python core/main_pipline/convert_triple.py \
  --input_file $DATA \
  --output_file $OUT/convert_triple_result.jsonl \
  --workers 16

# D2: assign L1/L2 entity types to document triples
python core/main_pipline/typed_triple.py \
  --input_file $OUT/convert_triple_result.jsonl \
  --output_file $OUT/typed_triple_result.jsonl \
  --faiss_dir type_faiss \
  --workers 16
```

### Query side (decompose and type the query)

```bash
# Q1: decompose multi-hop question into atomic subqueries
python core/main_pipline/query_decompose.py \
  --input_file $DATA \
  --output_file $OUT/query_decompose_result.jsonl \
  --workers 16

# Q2: convert each subquery into a triple
python core/main_pipline/query_triple.py \
  --input_file $OUT/query_decompose_result.jsonl \
  --output_file $OUT/query_triple_result.jsonl \
  --workers 16

# Q3: assign L1/L2 types to query triples
python core/main_pipline/query_typed.py \
  --input_file $OUT/query_triple_result.jsonl \
  --output_file $OUT/query_typed_result.jsonl \
  --faiss_dir type_faiss \
  --workers 16
```

D1/D2 and Q1/Q2/Q3 are independent and can run in parallel.

### Step-by-step answering (matching + answer generation)

```bash
python core/main_pipline/query_stepbystep.py \
  --query_file $OUT/query_typed_result.jsonl \
  --doc_file   $OUT/typed_triple_result.jsonl \
  --data_file  $DATA \
  --output_file $OUT/stepbystep_result.jsonl \
  --workers 16
```

---

## Step 4 — Evaluate

```bash
python core/evaluate.py \
  --results_file $OUT/stepbystep_result.jsonl \
  --metric em
```

Supported metrics: `em`, `f1`, `match`, `accuracy`, `rouge`.

---

## Common flags

All pipeline scripts share these optional flags:

| Flag | Default | Description |
|---|---|---|
| `--workers N` | 16 | Parallel threads |
| `--sample N` | all | Process only first N records |
| `--start N` | 0 | Skip first N records (resume) |
| `--cache_file path` | none | Persist entity→type cache across runs (typed_triple.py only) |

---

## Repository structure

```
core/
  config.py               — API endpoints and model names (edit before running)
  build_faiss_type_index.py — builds type_faiss/ from the L1/L2 taxonomy
  main_pipline/           — pipeline stages (D1→D2, Q1→Q2→Q3, answering)
  matching/               — triple matching and document ranking
  utils/                  — LLM clients, FAISS retriever, JSON parsing
  evaluate.py / metrics.py — scoring
type_faiss/               — generated index (created by Step 1)
data/                     — sample input files
```

---

*This README is written to be parsed and executed friendly for AI coding assistants (Claude, Copilot, etc.). All commands are self-contained and copy-pasteable, with no implicit context required between steps.*
