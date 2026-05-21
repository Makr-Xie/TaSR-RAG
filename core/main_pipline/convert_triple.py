import json
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm

from openai import OpenAI

Triple = Tuple[str, str, str]
TopkSpec = Tuple[str, Optional[int], Optional[int]]

# Imported from utils
import sys
from pathlib import Path

# Add parent directory to sys.path to allow imports from absQA
current_dir = Path(__file__).resolve().parent
if str(current_dir.parent) not in sys.path:
    sys.path.append(str(current_dir.parent))

from utils import simple_call_vllm, strip_thinking_tags
from config import CHAT_API_BASE, CHAT_MODEL, API_KEY

from openai import OpenAI
client = OpenAI(
    base_url=CHAT_API_BASE,
    api_key=API_KEY,
)

def call_vllm(prompt: str, model: str = CHAT_MODEL) -> str:
    # simple_call_vllm returns stripped content, but let's ensure thinking tags are handled if needed
    # The original util `simple_call_vllm` does NOT explicitly call `strip_thinking_tags` 
    # (it just strips whitespace), so we apply it here.
    raw_content = simple_call_vllm(
        prompt, 
        client, 
        model=model, 
        temperature=0.7, 
        max_tokens=2048
    )
    return strip_thinking_tags(raw_content)


# =====================================================
# Prompt for Triple extraction (from absQA)
# =====================================================
extract_prompt_template_with_query = """
Your task is to extract factual triples, strictly as (subject, relation, object), from the Document that might help to answer the User Query.

USER QUERY: "{query}"

You must output ONLY a JSON array of strings: 
[
  ["subject", "relation", "object"],
  ...
]

Each triple MUST contain exactly 3 **non-empty** strings. 


##DEFINITION OF A "RELATION"##

A relation is ANY link between two entities. Extract BOTH:

1. **Explicit Actions (Verbs)**
   - When A does something to B.
   - Example: "A attacked B" → ["A", "attacked", "B"]

2. **Implicit Static Relations**
   Relations expressed by:
   - **Possessives:** "A's B", "his B", "their B"
     → (A, has, B) or a more specific relation if described.
   - **Appositives:** "A, the B,"
     → (A, is a, B)
   - **Prepositions:** "A in B", "A of B"
     → (A, located in, B) or (A, part of, B)
   - **Noun modifiers:** "A — a famous scientist"
     → (A, is a, famous scientist)

##STRICT RULES##

### 1. Output Constraints
- Return ONLY a JSON array.
- No markdown, no code fences, no extra text.
- No null, no empty strings.

### 2. Relation Normalization
Use simple, active, human-readable phrases.
Examples:
- "is located in"
- "is a"
- "has attribute"
- "served as"
- "started on"
- "ended on"

### 3. Capture EVERY Relation
Do not skip any relation simply because:
- it has no explicit verb,
- it uses a noun,
- it is a static description,
- the sentence structure is unusual.

### 4. Resolve Coreference
Replace pronouns with their actual referents:
- "he", "she", "they", "the company", "the organization"
→ Use the closest correct entity name.

### 5. NO EMPTY OBJECTS ALLOWED
For intransitive verbs ("returned", "died"), infer minimal context:
- If context says what returned *to* → use that.
- If no context → use fallback such as:
  - "to the series"
  - "in the event"
  - "in the timeline"

### 6. **DATE & EVENT RULES (MANDATORY)**
Dates are entities.

If a sentence indicates an event with a date:
- ALWAYS extract:
  - ["X", "started on", DATE]
  - ["X", "ended on", DATE]
  - ["X", "occurred on", DATE]  
  depending on context.

### 7. **Handling Date Ranges**
Text: "ran from January 1, 2015 to March 5, 2015"
→ Must output two triples:

[
  ["X", "started on", "January 1, 2015"],
  ["X", "ended on", "March 5, 2015"]
]

### 8. **Handling Truncated Sentences (Chunk Boundaries)**
If a sentence begins mid-text and the subject is missing:
- Infer it from the nearest previous sentence or paragraph.
- You MUST still extract the relation.

Example:
Text: "… and ended March 5, 2015."
Context: "Season 6 ran from…"

Output:
[
  ["Season 6", "ended on", "March 5, 2015"]
]

### 9. No hallucination
Do NOT invent entities or facts not grounded in the text.
BUT you MUST infer missing subjects if the text is truncated.

##EXAMPLES (Learning the Patterns)##

**Pattern 1: Appositive**
Text: "Python, a high-level language, is popular."
Output:
[
  ["Python", "is a", "high-level language"],
  ["Python", "has attribute", "popular"]
]

**Pattern 2: Possessive**
Text: "Google's CEO Sundar Pichai announced the update."
Output:
[
  ["Sundar Pichai", "is CEO of", "Google"],
  ["Sundar Pichai", "announced", "the update"]
]

**Pattern 3: Relative Clause**
Text: "Sarah, who leads the design team, resigned."
Output:
[
  ["Sarah", "leads", "the design team"],
  ["Sarah", "resigned", "from the position"]
]

**Pattern 4: Prepositional Relation**
Text: "The Eiffel Tower is a wrought-iron tower in Paris."
Output:
[
  ["The Eiffel Tower", "is a", "wrought-iron tower"],
  ["The Eiffel Tower", "is located in", "Paris"]
]

**Pattern 5: Date-bound Event**
Text: "Season 3 ran from January 1, 2015 to March 5, 2015."
Output:
[
  ["Season 3", "started on", "January 1, 2015"],
  ["Season 3", "ended on", "March 5, 2015"]
]

##INPUT DOCUMENT##

{doc_text}

##OUTPUT FORMAT##

Return ONLY:
[
  ["A", "B", "C"],
  ["D", "E", "F"],
  ...
]
All fields must be non-empty strings.

"""

repair_prompt_template = """
You will be given:
1.  The source DOCUMENT_TEXT that the triples were extracted from.
2.  A JSON list of items that are malformed (subject, relation, object) triples.

TASK:
- Use ONLY the provided document as evidence.
- For each malformed item, infer the intended fact and rewrite it as one or more valid triples.

STRICT RULES:
1.  **Document Grounding**: Every repaired triple must be supported by DOCUMENT_TEXT. Do not invent new facts.
2.  **Triple Format**: Each triple MUST be an array of EXACTLY 3 non-empty strings: ["subject", "relation", "object"].
3.  **Relation Folding**: If an item looks like ["subject", "is", "description", "object"], fold it into ["subject", "is description", "object"]. If an object slot is blank, use the document to complete it (e.g., ["The actors", "returned", "" ] → ["The actors", "returned in", "the series"]).
4.  **Preserve Valid Items**: If an entry is already a correct triple, return it unchanged.
5.  **Output**: Return ONLY a JSON array containing all corrected (and preserved) triples. No explanations.

DOCUMENT_TEXT:
{doc_text}

MALFORMED LIST TO REPAIR:
{malformed_list_json}
"""


def is_valid_string(s: str) -> bool:
    return isinstance(s, str) and bool(s.strip())


def repair_malformed_triples(malformed_entries: List, doc_text: str) -> List[List[str]]:
    if not malformed_entries:
        return []

    malformed_json = json.dumps(malformed_entries, indent=2, ensure_ascii=False)
    prompt = repair_prompt_template.format(
        malformed_list_json=malformed_json,
        doc_text=doc_text or ""
    )

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        output = ""
        try:
            output = call_vllm(prompt)
            repaired_list = json.loads(output)

            if isinstance(repaired_list, list):
                final_triples = [
                    t for t in repaired_list
                    if isinstance(t, list) and len(t) == 3 and all(is_valid_string(s) for s in t)
                ]
                return final_triples
            raise ValueError("Repair LLM did not return a list.")

        except Exception:
            if attempt == max_retries:
                break

    return []


def extract_triples(doc_text: str, query_text: str, record_id: str = None, question: str = None) -> List[List[str]]:
    prompt = extract_prompt_template_with_query.format(doc_text=doc_text, query=query_text)
    max_retries = 10
    output = ""
    
    for attempt in range(1, max_retries + 1):
        try:
            output = call_vllm(prompt)
            triples = json.loads(output)
            if isinstance(triples, list) and all(
                isinstance(t, list) and len(t) == 3 and all(is_valid_string(s) for s in t)
                for t in triples
            ):
                return triples
            raise ValueError("JSON content is not a list of valid triples.")
        except Exception as e:
            if attempt == max_retries:
                break

    # Fallback: try to salvage partial output
    try:
        raw_output_list = json.loads(output)
        if not isinstance(raw_output_list, list):
            return []
    except Exception:
        # Only print failure message here
        print(f"[FAILED] id={record_id}, question='{question}'")
        return []

    is_good_triple = lambda t: isinstance(t, list) and len(t) == 3 and all(is_valid_string(s) for s in t)
    good_triples = [t for t in raw_output_list if is_good_triple(t)]
    malformed_entries = [t for t in raw_output_list if not is_good_triple(t)]

    repaired_triples = repair_malformed_triples(malformed_entries, doc_text) if malformed_entries else []
    return good_triples + repaired_triples


def _process_document(doc_id: int, doc_text: str, question: str, record_id: str = None) -> Tuple[int, str, List[List[str]]]:
    try:
        triples = extract_triples(doc_text, question, record_id=record_id, question=question)
        return doc_id, doc_text, triples or []
    except Exception:
        return doc_id, doc_text, []


def parse_topk_ctxs(spec: Optional[str]) -> TopkSpec:
    if spec is None:
        return ("all", None, None)
    if isinstance(spec, int):
        if spec <= 0:
            return ("all", None, None)
        return ("topk", spec, None)
    spec_str = str(spec).strip().lower()
    if spec_str in ("", "all", "*"):
        return ("all", None, None)
    if spec_str.isdigit():
        k = int(spec_str)
        if k <= 0:
            return ("all", None, None)
        return ("topk", k, None)
    if "-" in spec_str:
        start_s, end_s = spec_str.split("-", 1)
        if not start_s.isdigit() or not end_s.isdigit():
            raise ValueError(
                f"Invalid --topk_ctxs range: '{spec}'. Use 'start-end' (e.g., 11-20)."
            )
        start = int(start_s)
        end = int(end_s)
        if start < 1 or end < 1:
            raise ValueError("Range must be 1-based and >= 1.")
        if end < start:
            raise ValueError("Range end must be >= start.")
        return ("range", start, end)
    raise ValueError(
        f"Invalid --topk_ctxs value: '{spec}'. Use an integer, 'start-end', or 'all'."
    )


def select_ctxs_with_indices(ctxs: List[dict], topk_spec: TopkSpec) -> Tuple[List[dict], List[int]]:
    mode, start, end = topk_spec
    if mode == "all":
        return ctxs, list(range(1, len(ctxs) + 1))
    if mode == "topk":
        selected = ctxs[: start or 0]
        return selected, list(range(1, len(selected) + 1))
    if mode == "range":
        start_idx = (start or 1) - 1
        selected = ctxs[start_idx: end]
        indices = list(range((start or 1), (start or 1) + len(selected)))
        return selected, indices
    return ctxs, list(range(1, len(ctxs) + 1))


def process_single_record(data: dict, topk_spec: TopkSpec, max_doc_workers: int = 10) -> dict:
    """Process a single record (question) with parallel document processing."""
    question = data["question"]
    record_id = data.get("id")
    answers = data.get("answers", [])
    ctxs, ctx_indices = select_ctxs_with_indices(data.get("ctxs", []), topk_spec)

    triples_by_doc: List[Dict[str, object]] = []

    if ctxs:
        with ThreadPoolExecutor(max_workers=min(len(ctxs), max_doc_workers)) as executor:
            futures = []
            for doc_id, ctx in zip(ctx_indices, ctxs):
                title = ctx.get("title") or ""
                body = ctx.get("doc") or ctx.get("text") or ""
                combined_doc = f"{title}\n\n{body}".strip()
                futures.append(executor.submit(_process_document, doc_id, combined_doc, question, record_id))

            for future in as_completed(futures):
                doc_id, doc_text, triples = future.result()
                for triple in triples:
                    triples_by_doc.append({"doc_id": doc_id, "raw_triple": triple})

    # Sort by doc_id to preserve original order
    triples_by_doc.sort(key=lambda x: x["doc_id"])

    return {
        "question": question,
        "answer": answers,
        "triples": triples_by_doc,
        "ctxs": ctxs,
    }


if __name__ == "__main__":
    import argparse
    import threading
    from pathlib import Path
    import os

    parser = argparse.ArgumentParser(description="Extract triples for each question context.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSONL file (each line with question/answers/ctxs).")
    parser.add_argument("--output_file", type=str, required=True, help="Path to write the extracted triples JSONL.")
    parser.add_argument(
        "--topk_ctxs",
        type=str,
        default="10",
        help="Top contexts to process: integer K (top-K), range 'start-end' (1-based, inclusive), or 'all'.",
    )
    parser.add_argument("--workers", type=int, default=48, help="Number of parallel workers for processing questions.")
    args = parser.parse_args()
    topk_spec = parse_topk_ctxs(args.topk_ctxs)

    # Ensure output directory exists
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)

    # Load all input data first
    with open(args.input_file, "r", encoding="utf-8") as file:
        all_data = [json.loads(line) for line in file]
    
    print(f"Loaded {len(all_data)} questions from input file.")

    # Load existing output to detect already completed questions
    completed_questions = set()
    existing_results = {}  # question -> result dict
    if os.path.exists(args.output_file):
        with open(args.output_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        q = record.get("question", "")
                        if q:
                            completed_questions.add(q)
                            existing_results[q] = record
                    except json.JSONDecodeError:
                        continue
        print(f"Found {len(completed_questions)} already completed questions in output file.")

    # Find which questions still need processing
    todo_indices = []
    for idx, data in enumerate(all_data):
        q = data.get("question", "")
        if q not in completed_questions:
            todo_indices.append(idx)
    
    print(f"Need to process {len(todo_indices)} remaining questions.")

    if not todo_indices:
        print("All questions already processed. Nothing to do.")
    else:
        # File lock for thread-safe writing
        write_lock = threading.Lock()
        
        # Open output file in append mode
        out_file = open(args.output_file, "a", encoding="utf-8")
        
        def process_and_save(idx: int) -> None:
            """Process a single record and save immediately."""
            data = all_data[idx]
            try:
                result = process_single_record(data, topk_spec)
                with write_lock:
                    out_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                    out_file.flush()
            except Exception as exc:
                print(f"[ERROR] idx={idx}, question='{data.get('question', '')[:50]}': {exc}")
        
        # Process remaining questions in parallel
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_and_save, idx): idx for idx in todo_indices}
            
            with tqdm(total=len(todo_indices), desc="Processing questions", unit="question") as pbar:
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        future.result()  # Raise any exceptions
                    except Exception as exc:
                        print(f"[FATAL] idx={idx}: {exc}")
                    pbar.update(1)
        
        out_file.close()
        print(f"Done. Output: {args.output_file}")

    # Now reorder outputs to match input order
    print("Reordering output to match input order...")
    
    # Reload all output results
    all_results = {}
    with open(args.output_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    record = json.loads(line)
                    q = record.get("question", "")
                    if q:
                        all_results[q] = record
                except json.JSONDecodeError:
                    continue
    
    # Write in original input order
    with open(args.output_file, "w", encoding="utf-8") as f:
        for data in all_data:
            q = data.get("question", "")
            if q in all_results:
                f.write(json.dumps(all_results[q], ensure_ascii=False) + "\n")
    
    print(f"Reordered {len(all_results)} results. Final output: {args.output_file}")
