import ast
import json
import re
from typing import Any, Dict, List, Optional, Union
from openai import OpenAI

Triple = List[str]  # ["S","P","O"]

# Imported from utils
import sys
from pathlib import Path

# Add parent directory to sys.path to allow imports from absQA
current_dir = Path(__file__).resolve().parent
if str(current_dir.parent) not in sys.path:
    sys.path.append(str(current_dir.parent))

from utils import simple_call_vllm, loads_json_or_py
from config import CHAT_API_BASE, CHAT_MODEL, API_KEY

client_llm = OpenAI(
    base_url=CHAT_API_BASE,
    api_key=API_KEY,
)

def call_vllm(
    prompts: Union[str, List[str]],
    model: str = CHAT_MODEL,
    temperature: float = 0.7,
    top_p: float = 1.0,
    max_tokens: int = 2048,
) -> Union[str, List[str]]:
    """Supports single prompt (str) or batch (list[str])."""
    single = isinstance(prompts, str)
    prompt_list = [prompts] if single else prompts

    results: List[str] = []
    for prompt in prompt_list:
        resp = client_llm.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False}
            }
        )
        content = resp.choices[0].message.content or ""
        results.append(content.strip())

    return results[0] if single else results

def _loads_json_or_py(raw: str) -> Any:
    return loads_json_or_py(raw)


# -----------------------------
# Step 0) decide decomposable?
# -----------------------------
def build_decomposable_judge_prompt(query: str) -> str:
    return f"""You are a query analysis engine.

Task: Decide whether the query should be decomposed into multiple subqueries for retrieval.

Definition:
- decomposable=true if the query contains multiple atomic information needs (e.g., A and B; multiple attributes; multi-hop; constraints + target).
- decomposable=false if it's essentially a single atomic ask, or an explanation/opinion request better treated as one retrieval intent.

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
    """
    Returns:
      {
        "decomposable": bool,
        "reason": str,
        "suggested_max_subqueries": int
      }
    """
    prompt = build_decomposable_judge_prompt(query)
    out = call_vllm(prompt, temperature=0.7, max_tokens=2048)
    data = _loads_json_or_py(out)

    # robust fallback
    if not isinstance(data, dict) or "decomposable" not in data:
        # heuristic fallback if LLM output is weird
        q = query.lower()
        heuristic = any(tok in q for tok in [" and ", " or ", ",", ";", "who designed", "when was", "where is", "population"])
        return {"decomposable": bool(heuristic), "reason": "fallback-heuristic", "suggested_max_subqueries": min(6, max_subqueries_cap)}

    dec = bool(data.get("decomposable", False))
    reason = str(data.get("reason", "")).strip() or "n/a"
    k = data.get("suggested_max_subqueries", 3)
    try:
        k = int(k)
    except Exception:
        k = 3
    k = max(1, min(max_subqueries_cap, k))
    return {"decomposable": dec, "reason": reason, "suggested_max_subqueries": k}


# -----------------------------
# Step 1) query -> subqueries
# -----------------------------
def build_decompose_prompt(query: str, max_subqueries: int = 6) -> str:
    return f"""You are a query decomposition engine.

Task: Decompose the user query into a JSON array of short, atomic subqueries for retrieval.
Rules:
- Output MUST be valid JSON only (a list of strings). No extra text.
- Keep the same language as the query.
- Each subquery should be self-contained, minimal, and corresponds to one fact/constraint.
- Keep original entity surface forms when possible.
- Max {max_subqueries} subqueries.

Now decompose:

Query: "{query}"
Output:
"""


def decompose_query_to_subqueries(query: str, max_subqueries: int = 6) -> List[str]:
    prompt = build_decompose_prompt(query, max_subqueries=max_subqueries)
    out = call_vllm(prompt, temperature=0.7, max_tokens=2048)
    data = _loads_json_or_py(out)
    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
        raise ValueError(f"Bad subquery output: {out[:200]!r}")
    subqs = [s.strip() for s in data if s.strip()]
    # safety: at least 1
    return subqs[:max_subqueries] if subqs else [query.strip()]


# -----------------------------
# Step 2) subqueries -> triplets
# -----------------------------
def build_subqueries_to_triples_prompt(subqueries: List[str], expected_count: int) -> str:
    return f"""You are a structured semantic parser.

Task: Convert EACH subquery into EXACTLY ONE triple ["subject","predicate","object"] for later matching.

Hard constraints:
- Output MUST be valid JSON only: {{"triples":[...]}}, no extra text.
- EXACTLY ONE triple per subquery (1-to-1 mapping). You MUST output EXACTLY {expected_count} triples.
- Keep the same order as subqueries (do not reorder).
- Use concise predicates that reflect the actual relationship.
- Use variables for unknowns: "?ans", "?person", "?date", "?place", "?num", "?entity", "?thing", and etc.
- Do NOT use outside knowledge; just convert the intent.

CRITICAL RULES (MUST FOLLOW):
1. NEVER use the entire question sentence as subject or object. Extract entities/concepts only.
2. NEVER use vague predicates like "relates to", "about", "concerns" - always extract a meaningful relationship.
3. Convert questions to declarative triple form by identifying: WHO/WHAT does WHAT to WHOM/WHAT.
4. PRESERVE concrete entities mentioned in the subquery! Only use variables for UNKNOWN things being asked about.
   - If the subquery mentions a specific entity like "hydrogen", "Paris", "Obama", use that entity name, NOT a variable.
   - Use variables like ?person, ?date, ?num ONLY for the unknown answer being sought.

EXAMPLES:
Subquery: "What is the atomic number of hydrogen?"
✓ GOOD: ["hydrogen", "has atomic number", "?num"]  ← "hydrogen" is a concrete entity, preserved!
✗ BAD:  ["?element", "has atomic number", "?num"]  ← Wrong! "hydrogen" was replaced with variable

Subquery: "Who was the Roman emperor that declared war on the sea?"
✓ GOOD: ["?person", "declared war on", "the sea"]
✗ BAD:  ["Who was the Roman emperor that declared war on the sea?", "relates to", "?ans"]

Subquery: "What is the political party of that president?"
✓ GOOD: ["?person", "has political party", "?party"]

Subquery: "Which team won in women's volleyball at that Olympics?"
✓ GOOD: ["?team", "won", "women's volleyball"]

Now convert:

Subqueries (ordered):
{json.dumps(subqueries, ensure_ascii=False)}

Output JSON:
"""


def subqueries_to_triplets(subqueries: List[str]) -> List[Triple]:
    prompt = build_subqueries_to_triples_prompt(subqueries, expected_count=len(subqueries))
    out = call_vllm(prompt, temperature=0.7, max_tokens=2048)
    data = _loads_json_or_py(out)
    if not isinstance(data, dict) or "triples" not in data:
        raise ValueError(f"Bad triples output: {out[:200]!r}")

    triples = data["triples"]
    if not isinstance(triples, list) or len(triples) != len(subqueries):
        raise ValueError(
            f"Triples length mismatch. subqueries={len(subqueries)} "
            f"triples={len(triples) if isinstance(triples, list) else 'N/A'}"
        )

    cleaned: List[Triple] = []
    for t in triples:
        if not (isinstance(t, (list, tuple)) and len(t) == 3 and all(isinstance(x, str) for x in t)):
            raise ValueError(f"Invalid triple item: {t!r}")
        cleaned.append([t[0].strip(), t[1].strip(), t[2].strip()])
    return cleaned


# -----------------------------
# Fallback: query -> ONE triple
# -----------------------------
def build_query_to_single_triple_prompt(query: str) -> str:
    return f"""Convert the query into ONE triple in JSON format: {{"triple":["subject","predicate","object"]}}

Rules:
- Output ONLY the JSON, nothing else
- Keep concrete entities from the query (movie names, person names, places, routines, etc.)
- ONLY use variables like "?person", "?date", "?place", "?movie" for the UNKNOWN thing the query is asking about
- The variable can be subject OR object depending on what the query asks

Examples:
Query: "Who is the CEO of Apple?"
{{"triple":["Apple","CEO","?person"]}}

Query: "When did Titanic come out?"
{{"triple":["Titanic","came out","?date"]}}

Query: "Which movie features the routine 'Who's on First'?"
{{"triple":["?movie","features","Who's on First"]}}

Query: "{query}"
"""


def query_to_single_triplet(query: str) -> Triple:
    """Convert a single query to a triple. No fallback, no retry."""
    prompt = build_query_to_single_triple_prompt(query)
    out = call_vllm(prompt, temperature=0.7, max_tokens=2048)
    data = json.loads(out)
    t = data["triple"]
    if not (isinstance(t, list) and len(t) == 3 and all(isinstance(x, str) for x in t)):
        raise ValueError(f"Invalid triple: {t!r}")
    return [t[0].strip(), t[1].strip(), t[2].strip()]


# -----------------------------
# One-shot wrapper (updated)
# -----------------------------
def query_to_triplets(item_id: Any, query: str, max_subqueries: int = 6) -> Dict[str, Any]:
    judge = is_query_decomposable(query, max_subqueries_cap=max_subqueries)

    if not judge["decomposable"]:
        triple = query_to_single_triplet(query)
        return {
            "id": item_id,
            "query": query,
            "mode": "single",
            "judge": judge,
            "subqueries": [query],
            "triples": [triple],
        }

    k = min(max_subqueries, int(judge.get("suggested_max_subqueries", max_subqueries)))
    subqs = decompose_query_to_subqueries(query, max_subqueries=k)
    triples = subqueries_to_triplets(subqs)
    return {
        "id": item_id,
        "query": query,
        "mode": "decomposed",
        "judge": judge,
        "subqueries": subqs,
        "triples": triples,
    }


def process_single_record(record: Dict[str, Any]) -> Dict[str, Any]:
    record_id = record.get("id")
    question = record.get("question", "")
    answers = record.get("answer", record.get("answers", []))
    decomposable = record.get("decomposable", False)
    sub_queries = record.get("sub_queries", [question])

    if not sub_queries:
        sub_queries = [question]

    output_triples = []
    try:
        if len(sub_queries) > 1:
            triples = None
            last_err = None
            for attempt in range(10):
                try:
                    triples = subqueries_to_triplets(sub_queries)
                    break
                except Exception as exc:
                    last_err = exc
                    continue
            
            if triples is None:
                raise last_err or ValueError("Failed after retries")

            for i, (sq, triple) in enumerate(zip(sub_queries, triples)):
                output_triples.append({
                    "sub_query_idx": i,
                    "sub_query": sq,
                    "raw_triple": triple
                })
        else:
            triple = None
            last_err_s = None
            for attempt in range(3):
                try:
                    triple = query_to_single_triplet(sub_queries[0])
                    break
                except Exception as exc:
                    last_err_s = exc
                    continue
            
            if triple is None:
                raise last_err_s or ValueError("Failed after retries")

            output_triples.append({
                "sub_query_idx": 0,
                "sub_query": sub_queries[0],
                "raw_triple": triple
            })
    except Exception as e:
        print(f"[FAILED] id={record_id}, question='{question}': {e}")
        for i, sq in enumerate(sub_queries):
            output_triples.append({
                "sub_query_idx": i,
                "sub_query": sq,
                "raw_triple": [sq, "relates to", "?ans"]
            })

    return {
        "id": record_id,
        "question": question,
        "answer": answers,
        "decomposable": decomposable,
        "triples": output_triples
    }


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from tqdm import tqdm
    from concurrent.futures import ThreadPoolExecutor, as_completed

    parser = argparse.ArgumentParser(description="Convert queries to triples.")
    parser.add_argument("--input_file", type=str, required=True, help="Input JSONL (from query_decompose.py).")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSONL with query triples.")
    parser.add_argument("--sample", type=int, default=None, help="Process only first N records (default: all).")
    parser.add_argument("--max_subqueries", type=int, default=10, help="Max subqueries per question.")
    parser.add_argument("--workers", type=int, default=16, help="Number of parallel workers (default: 16).")
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
            executor.submit(process_single_record, record): idx
            for idx, record in enumerate(input_records)
        }
        
        for future in tqdm(as_completed(future_to_idx), total=len(input_records), desc="Converting to triples", unit="query"):
            idx = future_to_idx[future]
            try:
                res = future.result()
                results.append((idx, res))
            except Exception as e:
                print(f"Worker exception at index {idx}: {e}")

    # Sort results to maintain original input order
    results.sort(key=lambda x: x[0])

    with open(args.output_file, "w", encoding="utf-8") as outfile:
        for _, res in results:
            outfile.write(json.dumps(res, ensure_ascii=False) + "\n")

    print(f"Done. Output: {args.output_file}")
