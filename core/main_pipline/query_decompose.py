#!/usr/bin/env python3
"""
Query Decomposition Step
Input: JSONL with id, question, answer
Output: JSONL with id, question, answer, decomposable, decompose_reason, sub_queries
"""

import json
import re
import ast
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI

# =====================================================
# vLLM Client
# =====================================================
# Imported from utils
import sys
from pathlib import Path

# Add parent directory to sys.path for local imports
current_dir = Path(__file__).resolve().parent
if str(current_dir.parent) not in sys.path:
    sys.path.append(str(current_dir.parent))

from utils import simple_call_vllm, loads_json_or_py
from config import CHAT_API_BASE, CHAT_MODEL, API_KEY

# Initialize shared client
from openai import OpenAI
client = OpenAI(
    base_url=CHAT_API_BASE,
    api_key=API_KEY,
)

def call_vllm(prompt: str, temperature: float = 0.0, max_tokens: int = 512) -> str:
    return simple_call_vllm(
        prompt, 
        client, 
        model=CHAT_MODEL, 
        temperature=temperature, 
        max_tokens=max_tokens
    )

def _loads_json_or_py(raw: str) -> Any:
    return loads_json_or_py(raw)


# =====================================================
# Step 1: Decide if decomposable
# =====================================================
def build_decomposable_judge_prompt(query: str) -> str:
    return f"""You are a query analysis engine.

Task: Decide whether the query should be decomposed into multiple subqueries for retrieval.

Definition:
- decomposable=true if the query contains NESTED clauses or multi-hop reasoning (e.g., "the director of the movie that...", "in the year that...", "the capital of the country where...").
- decomposable=true if it asks about multiple entities or attributes requiring separate lookups.
- decomposable=false ONLY if it's a truly simple, single-hop question (e.g., "Who wrote Harry Potter?", "When was Obama born?").

Examples:
Query: "Who was president of the United States in the year that Citibank was founded?"
Output: {{"decomposable": true, "reason": "nested temporal constraint", "suggested_max_subqueries": 2}}

Query: "What is the population of the capital of France?"
Output: {{"decomposable": true, "reason": "nested location query", "suggested_max_subqueries": 2}}

Query: "Who wrote Harry Potter?"
Output: {{"decomposable": false, "reason": "simple single-hop question", "suggested_max_subqueries": 1}}

Hard constraints:
- Output MUST be valid JSON only. No extra text.
- JSON schema:
  {{
    "decomposable": true/false,
    "reason": "one short sentence",
    "suggested_max_subqueries": 1..6
  }}

Query: "{query}"
Output:
"""


def is_query_decomposable(query: str, max_subqueries_cap: int = 6) -> Dict[str, Any]:
    prompt = build_decomposable_judge_prompt(query)
    out = call_vllm(prompt, temperature=0.0, max_tokens=512)
    try:
        data = _loads_json_or_py(out)
    except Exception:
        data = {}

    # robust fallback
    if not isinstance(data, dict) or "decomposable" not in data:
        q = query.lower()
        heuristic = any(tok in q for tok in [" and ", " or ", ",", ";", "who designed", "when was", "where is", "population"])
        return {"decomposable": bool(heuristic), "reason": "fallback-heuristic", "suggested_max_subqueries": min(3, max_subqueries_cap)}

    dec = bool(data.get("decomposable", False))
    reason = str(data.get("reason", "")).strip() or "n/a"
    k = data.get("suggested_max_subqueries", 3)
    try:
        k = int(k)
    except Exception:
        k = 3
    k = max(1, min(max_subqueries_cap, k))
    return {"decomposable": dec, "reason": reason, "suggested_max_subqueries": k}


# =====================================================
# Step 2: Decompose query to subqueries
# =====================================================
def build_decompose_prompt(query: str, max_subqueries: int = 6) -> str:
    return f"""You are a query decomposition engine.

Task: Decompose the user query into a JSON array of short, atomic subqueries for retrieval.

Rules:
- Output MUST be valid JSON only (a list of strings). No extra text.
- Keep the same language as the query.
- Each subquery should be TRULY ATOMIC - one single fact lookup or one single operation.
- Break down nested constraints (e.g., "X of Y" -> "Find Y", then "Find X of Y").
- If there's a calculation step (double, half, sum, etc.), make it a SEPARATE subquery.
- Keep original entity surface forms when possible.
- Max {max_subqueries} subqueries.

CRITICAL: Ensure each subquery asks for EXACTLY ONE piece of information.

Examples:
Query: "Who was president of the United States in the year that Citibank was founded?"
Output: ["When was Citibank founded?", "Who was president of the United States in that year?"]

Query: "Which element has an atomic number that is double that of hydrogen?"
Output: ["What is the atomic number of hydrogen?", "What is double that number?", "Which element has that atomic number?"]

Query: "What is the population of the capital of France?"
Output: ["What is the capital of France?", "What is the population of that city?"]

Query: "How many calories are in three slices of bread?"
Output: ["How many calories are in one slice of bread?", "What is that number multiplied by three?"]

Query: "Who wrote Harry Potter?"
Output: ["Who wrote Harry Potter?"]

Now decompose:

Query: "{query}"
Output:
"""


def decompose_query_to_subqueries(query: str, max_subqueries: int = 6) -> List[str]:
    prompt = build_decompose_prompt(query, max_subqueries=max_subqueries)
    out = call_vllm(prompt, temperature=0.2, max_tokens=512)
    try:
        data = _loads_json_or_py(out)
    except Exception:
        return [query.strip()]

    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
        return [query.strip()]

    subqs = [s.strip() for s in data if s.strip()]
    return subqs[:max_subqueries] if subqs else [query.strip()]


# =====================================================
# Main: process one query
# =====================================================
def decompose_query(query: str, max_subqueries: int = 6) -> Dict[str, Any]:
    """
    Returns:
      {
        "decomposable": bool,
        "sub_queries": List[str]
      }
    """
    judge = is_query_decomposable(query, max_subqueries_cap=max_subqueries)

    if judge["decomposable"]:
        subqs = decompose_query_to_subqueries(query, max_subqueries=max_subqueries)
        actually_decomposed = len(subqs) > 1
        return {
            "decomposable": actually_decomposed,
            "sub_queries": subqs
        }
    else:
        subqs = decompose_query_to_subqueries(query, max_subqueries=3)
        if len(subqs) > 1:
            return {
                "decomposable": True,
                "sub_queries": subqs
            }
        else:
            return {
                "decomposable": False,
                "sub_queries": [query.strip()]
            }


# =====================================================
# Worker function for parallel processing
# =====================================================
def process_single_query(record: Dict[str, Any], max_subqueries: int) -> Dict[str, Any]:
    record_id = record.get("id")
    question = record.get("question", "")
    answers = record.get("answers", record.get("answer", []))

    try:
        result = decompose_query(question, max_subqueries=max_subqueries)
    except Exception as e:
        result = {
            "decomposable": False,
            "sub_queries": [question.strip()]
        }

    return {
        "id": record_id,
        "question": question,
        "answer": answers,
        "decomposable": result["decomposable"],
        "sub_queries": result["sub_queries"]
    }


# =====================================================
# CLI Entry Point
# =====================================================
if __name__ == "__main__":
    import argparse
    from concurrent.futures import ThreadPoolExecutor, as_completed

    parser = argparse.ArgumentParser(description="Decompose queries into sub-queries.")
    parser.add_argument("--input_file", type=str, required=True, help="Input JSONL (id, question, answers).")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSONL with decomposition.")
    parser.add_argument("--sample", type=int, default=None, help="Process only first N records (default: all).")
    parser.add_argument("--max_subqueries", type=int, default=6, help="Max subqueries per question.")
    parser.add_argument("--workers", type=int, default=16, help="Number of parallel workers (default: 64).")
    args = parser.parse_args()

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)

    input_records = []
    with open(args.input_file, "r", encoding="utf-8") as infile:
        for line in infile:
            input_records.append(json.loads(line))
    
    if args.sample is not None:
        input_records = input_records[:args.sample]

    print(f"Processing {len(input_records)} queries with {args.workers} workers...")

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_idx = {
            executor.submit(process_single_query, record, args.max_subqueries): idx
            for idx, record in enumerate(input_records)
        }
        
        for future in tqdm(as_completed(future_to_idx), total=len(input_records), desc="Decomposing queries", unit="query"):
            idx = future_to_idx[future]
            try:
                res = future.result()
                results.append((idx, res))
            except Exception as e:
                print(f"Worker exception at index {idx}: {e}")

    # Sort results to maintain original order (optional but nice)
    results.sort(key=lambda x: x[0])

    with open(args.output_file, "w", encoding="utf-8") as outfile:
        for _, res in results:
            outfile.write(json.dumps(res, ensure_ascii=False) + "\n")

    print(f"Done. Output: {args.output_file}")
