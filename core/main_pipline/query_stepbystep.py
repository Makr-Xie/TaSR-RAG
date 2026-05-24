#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step-by-Step Query Processing

Process each subquery sequentially:
1. Match subquery_i's triple against document triples
2. Generate answer for subquery_i
3. Use answer_i as context for subquery_(i+1)
4. Final answer = last subquery's answer

Reuses existing results from query_typed_result.jsonl and typed_triple_result.jsonl
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
from openai import OpenAI

# Import matching components
import sys
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from matching.matching_mean import (
    Embedder,
    QueryTriple,
    DocTriple,
    rank_docs_by_triple_matching,
)

from utils import simple_call_vllm
from config import CHAT_API_BASE, CHAT_MODEL, EMBED_API_BASE, EMBED_MODEL, API_KEY

client_llm = OpenAI(
    base_url=CHAT_API_BASE,
    api_key=API_KEY,
)



SUBQUERY_ANSWER_PROMPT = """You are a fact-grounded QA system.

Your task is to answer the question using ONLY the evidence documents provided below.

CRITICAL RULES:
- The answer MUST be taken DIRECTLY from the text in the evidence documents.
- Do NOT use any external knowledge or make inferences beyond what is explicitly stated.
- If the exact answer is not explicitly written in the documents, respond that facts are insufficient.

{context_section}

Question:
{question}

Evidence (ranked by relevance):
{docs}

Instructions:
1. First, analyze the evidence step by step in <reasoning>...</reasoning> tags.
   - Quote the relevant sentence(s) from the documents that contain the answer.
   - The answer must appear verbatim in the quoted text.
2. Then, provide your final answer in <answer>...</answer> tags.
3. The answer must be in the SHORTEST form possible (a name, number, or short phrase).
4. Do NOT add extra words, background, dates, or explanations in the answer.

Output format:
<reasoning>
[Quote the relevant text from documents, then explain your answer]
</reasoning>
<answer>[minimal answer string only, copied directly from evidence]</answer>
"""

SUBQUERY_REWRITE_PROMPT = """You are rewriting a follow-up subquery into a standalone question.

Task: Rewrite the current subquery by resolving references using previous subqueries and their answers.

Rules:
- Output ONLY the rewritten subquery text. No JSON, no quotes, no extra commentary.
- Preserve the original meaning and intent.
- Replace pronouns or vague references (e.g., "that year", "he", "this city") with the resolved answers.
- If no changes are needed, return the original subquery.

Previous steps:
{previous_steps}

Current subquery: "{current_subquery}"
Current triple: {current_triple}

Rewritten subquery:
"""


def call_llm(prompt: str, temperature: float = 0.1, max_tokens: int = 1024) -> str:
    """Call vLLM with thinking disabled."""
    return simple_call_vllm(
        prompt=prompt,
        client=client_llm,
        model=CHAT_MODEL,
        temperature=temperature,
        max_tokens=max_tokens,
        enable_thinking=False
    )


def extract_answer_tag(response: str) -> str:
    """Extract answer from <answer>...</answer> tags."""
    answer_pattern = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
    match = answer_pattern.search(response)
    if match:
        return match.group(1).strip()
    return response.strip()


def extract_reasoning_tag(response: str) -> str:
    """Extract reasoning from <reasoning>...</reasoning> tags."""
    reasoning_pattern = re.compile(r"<reasoning>(.*?)</reasoning>", re.IGNORECASE | re.DOTALL)
    match = reasoning_pattern.search(response)
    if match:
        return match.group(1).strip()
    return ""


def build_doc_text_map(ctxs: List[Dict]) -> Dict[int, str]:
    """Build mapping from doc_id (1-indexed) to document text."""
    doc_map = {}
    for idx, ctx in enumerate(ctxs, start=1):
        title = ctx.get("title", "")
        body = ctx.get("doc", "") or ctx.get("text", "")
        text = f"{title}\n\n{body}".strip() if (title or body) else ""
        doc_map[idx] = text
    return doc_map


def generate_subquery_answer(sub_query: str, ranked_doc_ids: List[int], doc_text_map: Dict[int, str], previous_steps: List[Dict[str, Any]], topk: int = 5,) -> Tuple[str, str]:
    """
    Generate answer for a single subquery using top-k ranked documents.
    
    Args:
        sub_query: The subquery text
        ranked_doc_ids: List of doc_ids sorted by relevance
        doc_text_map: Mapping from doc_id to document text
        previous_steps: List of previous step results, each with 'sub_query' and 'answer'
        topk: Number of top docs to use
    
    Returns:
        Tuple of (full_llm_response, extracted_answer)
    """
    # Build evidence text from top-k ranked documents
    evidence_lines = []
    for rank, doc_id in enumerate(ranked_doc_ids[:topk], start=1):
        text = doc_text_map.get(doc_id, "")
        if text:
            evidence_lines.append(f"[Doc {rank}]: {text}")
    
    if not evidence_lines:
        return ("", "UNKNOWN")
    
    docs_text = "\n\n".join(evidence_lines)
    
    # Build context section from previous subqueries and their answers
    context_section = ""
    if previous_steps:
        context_lines = []
        for step in previous_steps:
            sq = step.get("sub_query", "")
            ans = step.get("answer", "")
            context_lines.append(f"- Q: {sq}\n  A: {ans}")
        context_section = "Previously resolved questions:\n" + "\n".join(context_lines)
    
    prompt = SUBQUERY_ANSWER_PROMPT.format(
        question=sub_query,
        docs=docs_text,
        context_section=context_section,
    )
    
    try:
        response = call_llm(prompt)
        extracted = extract_answer_tag(response)
        return (response, extracted)
    except Exception as e:
        print(f"[ERROR] Failed to generate answer: {e}")
        return ("", "UNKNOWN")


# Prompt for final answer synthesis
FINAL_ANSWER_PROMPT = """You are a multi-hop QA system that synthesizes information from multiple reasoning steps.

Your task is to answer the ORIGINAL QUESTION by combining all the intermediate facts discovered through step-by-step reasoning.

CRITICAL RULES:
- Use the intermediate Q&A pairs to build a complete reasoning chain.
- The final answer should directly address the ORIGINAL QUESTION.
- If the intermediate steps do not provide enough information, use the evidence documents as additional support.
- The answer must be as SHORT as possible (a name, number, or short phrase).

ORIGINAL QUESTION:
{original_question}

REASONING CHAIN (step-by-step discoveries):
{reasoning_chain}

SUPPORTING DOCUMENTS:
{docs}

Instructions:
1. First, trace through the reasoning chain in <reasoning>...</reasoning> tags.
   - Connect each step's answer to build the full answer to the original question.
   - If any step returned "insufficient", try to fill the gap using the supporting documents.
2. Then, provide your final answer in <answer>...</answer> tags.
3. The answer must be MINIMAL - just the entity/value that directly answers the original question.

Output format:
<reasoning>
[Trace through the reasoning steps and synthesize the final answer]
</reasoning>
<answer>[minimal answer to the ORIGINAL question]</answer>
"""


def generate_final_answer(
    original_question: str,
    step_answers: List[Dict[str, Any]],
    last_step_doc_ids: List[int],
    doc_text_map: Dict[int, str],
    topk: int = 5,
) -> Tuple[str, str]:
    """
    Generate final answer by synthesizing all intermediate steps.
    
    Args:
        original_question: The original multi-hop question
        step_answers: List of all step results with 'sub_query' and 'answer'
        last_step_doc_ids: Document IDs from the last step's retrieval
        doc_text_map: Mapping from doc_id to document text
        topk: Number of top docs to use from last step
    
    Returns:
        Tuple of (full_llm_response, extracted_answer)
    """
    # Build reasoning chain from all steps
    reasoning_lines = []
    for i, step in enumerate(step_answers):
        sq = step.get("sub_query", "")
        ans = step.get("answer", "")
        reasoning_lines.append(f"Step {i+1}: Q: {sq}")
        reasoning_lines.append(f"        A: {ans}")
    reasoning_chain = "\n".join(reasoning_lines)
    
    # Build evidence text from last step's top-k documents
    evidence_lines = []
    for rank, doc_id in enumerate(last_step_doc_ids[:topk], start=1):
        text = doc_text_map.get(doc_id, "")
        if text:
            evidence_lines.append(f"[Doc {rank}]: {text}")
    
    docs_text = "\n\n".join(evidence_lines) if evidence_lines else "(No documents available)"
    
    prompt = FINAL_ANSWER_PROMPT.format(
        original_question=original_question,
        reasoning_chain=reasoning_chain,
        docs=docs_text,
    )
    
    try:
        response = call_llm(prompt, max_tokens=1024)
        extracted = extract_answer_tag(response)
        return (response, extracted)
    except Exception as e:
        print(f"[ERROR] Failed to generate final answer: {e}")
        # Fallback to last step's answer
        if step_answers:
            return ("", step_answers[-1].get("answer", "UNKNOWN"))
        return ("", "UNKNOWN")


# Prompt for triple substitution with full context
SUBSTITUTE_TRIPLE_PROMPT = """You are a triple substitution engine for multi-hop QA.

Task: Given a current subquery's triple with variables, and the previous subqueries with their answers,
substitute the correct variables with the resolved answers.

Rules:
- Output MUST be valid JSON only: {{"substituted_triple": ["...", "...", "..."]}}
- Analyze which previous answer should replace which variable based on semantic meaning.
- Keep variables that cannot be resolved yet (e.g., the answer we're still looking for).
- Do NOT change the predicate.
- If no substitution is needed, return the original triple.

Example 1:
Previous steps:
  Step 0: Subquery="Who was the last king from Britain's House of Hanover?"
          Triple=["?person", "was last king from", "Britain's House of Hanover"]
          Answer="William IV"

Current subquery: "When did that king die?"
Current triple: ["?person", "died", "?date"]

Analysis: Step 0 answered "William IV" for the question about "last king", so ?person in current triple should be "William IV".
Output: {{"substituted_triple": ["William IV", "died", "?date"]}}

Example 2:
Previous steps:
  Step 0: Subquery="When was Citibank founded?"
          Triple=["Citibank", "founded in", "?year"]
          Answer="1812"

Current subquery: "Who was president of the United States in that year?"
Current triple: ["?person", "was president of", "United States"]

Analysis: Step 0 found the year "1812". The current triple asks about a person, but we can add the year context.
The predicate should include the year context.
Output: {{"substituted_triple": ["?person", "was president of United States in 1812", "United States"]}}

Actually better approach - modify the object to include year:
Output: {{"substituted_triple": ["?person", "was president of", "United States in 1812"]}}

Now substitute:

Previous steps:
{previous_steps}

Current subquery: "{current_subquery}"
Current triple: {current_triple}

Output JSON:
"""


def substitute_triple_with_context(raw_triple: List[str], current_subquery: str, previous_steps: List[Dict[str, Any]]) -> List[str]:
    """
    Use LLM to substitute variables in the triple with resolved answers from previous steps.
    
    Args:
        raw_triple: Current triple like ['?person', 'died', '?date']
        current_subquery: The current subquery text
        previous_steps: List of dicts with keys: sub_query, raw_triple, answer
    
    Returns:
        Substituted triple like ['William IV', 'died', '?date']
    """
    if not previous_steps:
        return raw_triple
    
    # Check if triple has any variables to substitute
    has_variable = any(str(x).startswith("?") for x in raw_triple)
    if not has_variable:
        return raw_triple
    
    # Filter out steps with insufficient answers
    insufficient_msg = "The provided facts are insufficient to answer confidently."
    valid_steps = [s for s in previous_steps if s.get("answer") and s.get("answer") != insufficient_msg]
    if not valid_steps:
        return raw_triple
    
    # Build previous steps description
    steps_desc = []
    for i, step in enumerate(valid_steps):
        steps_desc.append(f'  Step {i}: Subquery="{step.get("sub_query", "")}"')
        steps_desc.append(f'          Triple={json.dumps(step.get("raw_triple", []), ensure_ascii=False)}')
        steps_desc.append(f'          Answer="{step.get("answer", "")}"')
    
    prompt = SUBSTITUTE_TRIPLE_PROMPT.format(
        previous_steps="\n".join(steps_desc),
        current_subquery=current_subquery,
        current_triple=json.dumps(raw_triple, ensure_ascii=False),
    )
    
    try:
        response = call_llm(prompt, temperature=0.1, max_tokens=1024)
        # Extract JSON
        response = response.strip()
        start = response.find("{")
        end = response.rfind("}") + 1
        if start != -1 and end > start:
            data = json.loads(response[start:end])
            substituted = data.get("substituted_triple", raw_triple)
            if isinstance(substituted, list) and len(substituted) == 3:
                return substituted
    except Exception as e:
        print(f"[WARN] Triple substitution failed: {e}")
    
    return raw_triple


def rewrite_subquery_with_context(current_subquery: str, current_triple: List[str], previous_steps: List[Dict[str, Any]]) -> str:
    """
    Use LLM to rewrite the subquery by resolving references from previous steps.
    
    Args:
        current_subquery: The current subquery text
        current_triple: Current triple like ['?person', 'died', '?date']
        previous_steps: List of dicts with keys: sub_query, raw_triple, answer
    
    Returns:
        Rewritten subquery text (standalone).
    """
    if not previous_steps:
        return current_subquery
    
    insufficient_msg = "The provided facts are insufficient to answer confidently."
    valid_steps = [s for s in previous_steps if s.get("answer") and s.get("answer") != insufficient_msg]
    if not valid_steps:
        return current_subquery
    
    steps_desc = []
    for i, step in enumerate(valid_steps):
        steps_desc.append(f'  Step {i}: Subquery="{step.get("sub_query", "")}"')
        steps_desc.append(f'          Answer="{step.get("answer", "")}"')
    
    prompt = SUBQUERY_REWRITE_PROMPT.format(
        previous_steps="\n".join(steps_desc),
        current_subquery=current_subquery,
        current_triple=json.dumps(current_triple, ensure_ascii=False),
    )
    
    try:
        response = call_llm(prompt, temperature=0.7, max_tokens=256)
        rewritten = response.strip()
        if rewritten:
            return rewritten
    except Exception as e:
        print(f"[WARN] Subquery rewrite failed: {e}")
    
    return current_subquery


def build_doc_triples(doc_triples_list: List[Dict]) -> List[DocTriple]:
    """
    Convert doc triples from JSONL format to DocTriple objects.
    """
    result = []
    for t in doc_triples_list:
        doc_id = str(t.get("doc_id", 0))
        raw = tuple(t.get("raw_triple", ["", "", ""]))
        typed = tuple(t.get("type_only_triple", t.get("raw_triple", ["", "", ""])))
        result.append(DocTriple(doc_id=doc_id, raw=raw, typed=typed))
    return result


def process_query(q_rec: Dict[str, Any], doc_records_by_query: Dict[str, Any], data_records: Dict[str, Any], embedder: Embedder, alpha_type: float = 0.5, threshold: float = 0.3, topk: int = 5,) -> Dict[str, Any]:
    """
    Process a single query step-by-step through its subqueries.
    
    For each subquery:
    1. Match subquery triple against doc triples
    2. Generate answer for this subquery
    3. Store answer for next subquery's context
    """
    q_id = q_rec.get("id")  # May be None
    question = q_rec.get("question", "")
    answers = q_rec.get("answer", [])
    triples = q_rec.get("triples", [])
    
    # Get corresponding doc record (use question as key since id may not exist)
    d_rec = doc_records_by_query.get(question)
    if not d_rec:
        return {
            "id": q_id,
            "question": question,
            "answer": answers,
            "pred": "UNKNOWN",
            "step_answers": [],
            "final_answer": None,
        }
    
    doc_triples_list = d_rec.get("triples", [])
    doc_triples = build_doc_triples(doc_triples_list)
    
    if not doc_triples:
        return {
            "id": q_id,
            "question": question,
            "answer": answers,
            "pred": "UNKNOWN",
            "step_answers": [],
            "final_answer": None,
        }
    
    # Get document text map from original data file (match by question text)
    data_rec = data_records.get(question)
    if data_rec:
        ctxs = data_rec.get("ctxs", [])
        doc_text_map = build_doc_text_map(ctxs)
    else:
        doc_text_map = {}
    
    # Step-by-step processile
    step_answers = []  # Store each subquery's result (sub_query + answer)
    
    for i, triple_info in enumerate(triples):
        sub_query = triple_info.get("sub_query", "")
        raw_triple = triple_info.get("raw_triple", [])
        typed_triple = triple_info.get("typed_triple", [])
        type_only_triple = triple_info.get("type_only_triple", [])
        
        # Step 0: Substitute variables in raw_triple with answers from previous steps
        substituted_triple = substitute_triple_with_context(raw_triple, sub_query, step_answers)
        rewritten_subquery = rewrite_subquery_with_context(sub_query, substituted_triple, step_answers)
        
        # Build QueryTriple for this single subquery (using substituted triple for raw)
        query_triple = QueryTriple(
            raw=tuple(substituted_triple),
            typed=tuple(type_only_triple) if type_only_triple else tuple(substituted_triple)
        )
        
        # Step 1: Match this single subquery's triple against all doc_triples
        ranked_docs, kept_doc_ids, global_matches, doc_scores, doc_score_details = rank_docs_by_triple_matching(
            query_triples=[query_triple],  # Single subquery
            doc_triples=doc_triples,
            embedder=embedder,
            alpha_type=alpha_type,
            threshold=threshold,
        )
        
        # Get top-k doc_ids for this subquery
        top_k_doc_ids = [int(doc_id) for doc_id, score in ranked_docs[:topk]]
        
        # Step 2: Generate answer for this subquery using top docs
        llm_response, step_answer = generate_subquery_answer(
            sub_query=rewritten_subquery,
            ranked_doc_ids=top_k_doc_ids,
            doc_text_map=doc_text_map,
            previous_steps=step_answers,  # Pass all previous steps with subquery + answer
            topk=topk,
        )
        
        step_result = {
            "sub_query_idx": i,
            "sub_query": sub_query,
            "rewritten_sub_query": rewritten_subquery,
            "raw_triple": raw_triple,
            "substituted_triple": substituted_triple,  # Show the substituted version
            "type_only_triple": type_only_triple,
            "context_before": [{"sub_query": s["sub_query"], "answer": s["answer"]} for s in step_answers],  # Previous steps
            "ranked_doc_ids": top_k_doc_ids,
            "doc_scores": {doc_id: score for doc_id, score in ranked_docs[:10]},
            "llm_response": llm_response,  # Full LLM response
            "answer": step_answer,
        }
        step_answers.append(step_result)
    
    # Get the last step's document IDs for final synthesis
    last_step_doc_ids = step_answers[-1]["ranked_doc_ids"] if step_answers else []
    
    # Final synthesis step: combine original question + all intermediate Q&As + last step's docs
    if len(step_answers) > 1:
        # Only do synthesis if there are multiple steps (multi-hop)
        final_response, final_answer = generate_final_answer(
            original_question=question,
            step_answers=step_answers,
            last_step_doc_ids=last_step_doc_ids,
            doc_text_map=doc_text_map,
            topk=topk,
        )
        synthesis_used = True
    else:
        # Single step: just use that step's answer
        final_answer = step_answers[-1]["answer"] if step_answers else None
        final_response = ""
        synthesis_used = False
    
    return {
        "id": q_id,
        "question": question,
        "answer": answers,
        "pred": final_answer or "UNKNOWN",  # Same format as generate_answer.py
        "step_answers": step_answers,
        "synthesis_used": synthesis_used,
        "final_synthesis_response": final_response if synthesis_used else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Step-by-step query processing: match and generate answers for each subquery sequentially.")
    parser.add_argument("--query_file", type=str, required=True, help="Input JSONL from query_typed.py (contains typed query triples).")
    parser.add_argument("--doc_file", type=str, required=True, help="Input JSONL from typed_triple.py (contains typed document triples).")
    parser.add_argument("--data_file", type=str, required=True, help="Original data file with ctxs (documents).")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSONL with step-by-step answers.")
    parser.add_argument("--embed_url", type=str, default=EMBED_API_BASE, help="Embedding server URL.")
    parser.add_argument("--alpha_type", type=float, default=0.5, help="Weight for typed score (vs embedding).")
    parser.add_argument("--threshold", type=float, default=0.3, help="Triple score threshold for sum aggregation.")
    parser.add_argument("--topk", type=int, default=10, help="Number of top documents to use for answer generation.")
    parser.add_argument("--sample", type=int, default=None, help="Process only first N records (default: all).")
    parser.add_argument("--workers", type=int, default=16, help="Number of parallel workers (default: 32).")
    args = parser.parse_args()

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)

    # Initialize Embedder
    embedder = Embedder(
        base_url=args.embed_url,
        api_key=API_KEY,
        model=EMBED_MODEL,
        batch_size=256,
    )
    print(f"Initialized Embedder at {args.embed_url}")

    # Load all query records
    query_records = []
    with open(args.query_file, "r", encoding="utf-8") as f:
        for line in f:
            query_records.append(json.loads(line))
    print(f"Loaded {len(query_records)} query records.")

    # Load all doc records and build index by query_id
    doc_records_by_query = {}
    with open(args.doc_file, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            query_text = rec.get("question", "")
            doc_records_by_query[query_text] = rec
    print(f"Loaded {len(doc_records_by_query)} doc records.")

    # Load original data file with ctxs
    data_records = {}  # Key by question text for matching
    with open(args.data_file, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            data_records[rec["question"]] = rec
    print(f"Loaded {len(data_records)} data records with ctxs.")

    # Apply sample limit
    if args.sample is not None:
        query_records = query_records[:args.sample]
        print(f"Processing first {args.sample} records.")

    # Define worker function for parallel processing
    def process_single_query(q_rec):
        try:
            return process_query(
                q_rec,
                doc_records_by_query,
                data_records=data_records,
                embedder=embedder,
                alpha_type=args.alpha_type,
                threshold=args.threshold,
                topk=args.topk,
            )
        except Exception as e:
            print(f"[ERROR] Failed to process query {q_rec.get('id')}: {e}")
            return {
                "id": q_rec.get("id"),
                "question": q_rec.get("question", ""),
                "answer": q_rec.get("answer", []),
                "pred": "UNKNOWN",
                "step_answers": [],
                "error": str(e),
            }

    # Process queries in parallel with ThreadPoolExecutor
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    results = []
    print(f"Processing {len(query_records)} queries with {args.workers} workers...")
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        future_to_idx = {executor.submit(process_single_query, q_rec): i 
                         for i, q_rec in enumerate(query_records)}
        
        # Collect results as they complete (with progress bar)
        for future in tqdm(as_completed(future_to_idx), total=len(query_records), 
                          desc="Processing queries", unit="query"):
            idx = future_to_idx[future]
            try:
                result = future.result()
                results.append((idx, result))
            except Exception as e:
                print(f"[ERROR] Task failed for index {idx}: {e}")
                results.append((idx, {"error": str(e)}))
    
    # Sort results by original order and write to file
    results.sort(key=lambda x: x[0])
    
    with open(args.output_file, "w", encoding="utf-8") as outfile:
        for idx, result in results:
            outfile.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    print(f"Done. Output: {args.output_file}")


if __name__ == "__main__":
    main()
