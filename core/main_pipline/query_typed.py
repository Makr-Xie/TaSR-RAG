#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import re
import time
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import faiss
from openai import OpenAI

Triple = List[str]  # ["s","p","o"]


# ============================================================
# 0) Variable detection + slot rules
# ============================================================
_VAR_RE = re.compile(r"^\?\w+$", re.UNICODE)

def is_var(x: str) -> bool:
    return isinstance(x, str) and _VAR_RE.match(x.strip()) is not None

def strip_qmark(x: str) -> str:
    x = (x or "").strip()
    return x[1:] if x.startswith("?") else x


# Hard rules: matched variables are assigned types directly without LLM.
SLOT_MAP: Dict[str, Tuple[str, str]] = {
    # time/date
    "?date": ("TIME", "Date"),
    "?time": ("TIME", "Date"),
    "?datetime": ("TIME", "Date"),
    "?year": ("TIME", "Year"),

    # person
    "?person": ("PERSON", "Person"),
    "?who": ("PERSON", "Person"),
    "?people": ("PERSON", "Person"),

    # location
    "?place": ("LOCATION", "Place"),
    "?where": ("LOCATION", "Place"),
    "?location": ("LOCATION", "Place"),
    "?city": ("LOCATION", "City"),
    "?country": ("LOCATION", "Country"),

    # quantity/number
    "?num": ("QUANTITY", "Number"),
    "?number": ("QUANTITY", "Number"),
    "?count": ("QUANTITY", "Number"),
    "?amount": ("QUANTITY", "Number"),
    "?population": ("QUANTITY", "Number"),
    "?age": ("QUANTITY", "Number"),
    "?score": ("QUANTITY", "Number"),

    # percentage/rate
    "?percent": ("QUANTITY", "Percentage"),
    "?percentage": ("QUANTITY", "Percentage"),
    "?rate": ("QUANTITY", "Percentage"),

    # organization/company
    "?org": ("ORGANIZATION", "Company"),
    "?company": ("ORGANIZATION", "Company"),
    "?organization": ("ORGANIZATION", "Company"),
}

def slot_type(x: str) -> Optional[Tuple[str, str]]:
    if not is_var(x):
        return None
    return SLOT_MAP.get(x.strip().lower())


# ============================================================
# 1) OpenAI-compatible clients (embedding/chat ports can differ)
# ============================================================
# ============================================================
# 1) OpenAI-compatible clients (embedding/chat ports can differ)
# ============================================================
# Imported from utils
import sys
from pathlib import Path

# Add parent directory to sys.path for local imports
current_dir = Path(__file__).resolve().parent
if str(current_dir.parent) not in sys.path:
    sys.path.append(str(current_dir.parent))

from utils import VLLMClients, extract_json_obj, FaissTypeRetriever
from config import CHAT_API_BASE, EMBED_API_BASE, CHAT_MODEL, EMBED_MODEL, API_KEY

def _dedup_keep_order(xs: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

def _parse_l1_top3(raw: str, candidates: List[str]) -> List[str]:
    obj = extract_json_obj(raw)
    top3 = obj.get("L1_top3", [])
    if not isinstance(top3, list):
        top3 = []
    top3 = [x for x in top3 if isinstance(x, str) and x in candidates]
    top3 = _dedup_keep_order(top3)

    # pad if needed
    for c in candidates:
        if len(top3) >= 3:
            break
        if c not in top3:
            top3.append(c)

    return top3[:3] if top3 else candidates[:3]

def _parse_pair_choice(raw: str, fallback_pair: str) -> Tuple[str, str]:
    obj = extract_json_obj(raw)
    l1 = obj.get("L1", None)
    l2 = obj.get("L2", None)
    if isinstance(l1, str) and isinstance(l2, str):
        return l1, l2
    a, b = fallback_pair.split("/", 1)
    return a, b


# ============================================================
# 4) Build sub-query context map (entity -> lines)
# ============================================================
def build_entity_context_from_subqueries(
    triples: List[Triple],
    subqueries: Optional[List[str]] = None,
    max_lines_per_ent: int = 6,
) -> Dict[str, str]:
    """
    entity -> aggregated context lines.
    Prefer using aligned subqueries (same length as triples).
    """
    buf: Dict[str, List[str]] = {}
    use_sq = isinstance(subqueries, list) and len(subqueries) == len(triples)

    for i, (s, p, o) in enumerate(triples):
        lines = []
        if use_sq:
            sq = (subqueries[i] or "").strip()
            if sq:
                lines.append(f'Subquery: "{sq}"')
        lines.append(f"Triple: {json.dumps([s, p, o], ensure_ascii=False)}")
        block = "\n".join(lines)

        for ent in (s, o):
            buf.setdefault(ent, [])
            if len(buf[ent]) < max_lines_per_ent:
                buf[ent].append(block)

    return {k: "\n---\n".join(v) for k, v in buf.items()}


# ============================================================
# 5) Prompts
# ============================================================
def l1_top3_prompt(target: str, ctx: str, candidates: List[str]) -> str:
    return f"""Choose the best 3 L1 types for the target entity, ordered best to worst.

Rules:
- Output MUST be valid JSON only: {{"L1_top3":["...","...","..."]}}
- MUST choose ONLY from candidates.
- MUST be unique and length=3.

CRITICAL: Infer the semantic type from context!
- If context asks "What is the religion of X?" → answer variable should be CONCEPT (religion is a concept)
- If context asks "What is the political party of X?" → answer should be ORGANIZATION (party is an org)
- If context asks "What city did X live?" → answer should be LOCATION
- If context asks "Who was the father of X?" → answer should be PERSON
- DO NOT default to PERSON/Actor for everything!

Examples:
- Target: "?ans" with context "What was the religion of that person?" → L1 should be CONCEPT, not PERSON
- Target: "?party" with context "political party of the president" → L1 should be ORGANIZATION

Target: "{target}"
Context (from sub-queries):
{ctx}

Candidates: {json.dumps(candidates, ensure_ascii=False)}

Output JSON:
"""

def pair_choice_prompt(target: str, ctx: str, pairs: List[str]) -> str:
    return f"""Choose the best (L1,L2) type pair for the target entity.

Rules:
- Output MUST be valid JSON only: {{"L1":"...","L2":"..."}}
- You MUST choose a pair from candidates.
- The output (L1,L2) must correspond to one candidate pair.

CRITICAL: Use context to determine the SEMANTIC type!
- "religion" → CONCEPT/Religion or CONCEPT/Theory, NOT PERSON/Actor
- "political party" → ORGANIZATION/PoliticalParty, NOT PERSON/Actor  
- "city" or "country" → LOCATION/City or LOCATION/Country
- "profession" or "job" → CONCEPT/Profession or CONCEPT/RoleOrTitle
- Only use PERSON types when the answer is actually a human being

BAD examples (DO NOT DO THIS):
- Target "?ans" asking about religion → "PERSON/Actor" 
- Target "?ans" asking about a city → "PERSON/Actor" 

GOOD examples:
- Target "?ans" asking about religion → "CONCEPT/Religion" 
- Target "?ans" asking about a city → "LOCATION/City" 

Target: "{target}"
Context (from sub-queries):
{ctx}

Candidates (L1/L2 pairs): {json.dumps(pairs, ensure_ascii=False)}

Output JSON:
"""


# ============================================================
# 6) Main: subquery-triples -> abs-triples
# ============================================================
def subquery_triples_to_abs_triples(
    triples: List[Triple],
    subqueries: Optional[List[str]],
    retriever: FaissTypeRetriever,
    clients: VLLMClients,
    *,
    l1_topk: int = 12,           # 12 = all L1 types (PERSON, ORGANIZATION, LOCATION, etc.)
    l2_topk_each: int = 20,
    embed_batch_size: int = 256,
    llm_batch_size: int = 128,
    max_pair_candidates: int = 120,
) -> List[Triple]:
    """
    Input:  List[["Steve Jobs","found","?entity"], ["?entity","established","?date"], ...]
    Output: List[["PERSON/Person","found","ORGANIZATION/Company"], ["ORGANIZATION/Company","established","TIME/Date"], ...]
    """

    # 1) context from sub-query triples
    ent_ctx = build_entity_context_from_subqueries(triples, subqueries=subqueries, max_lines_per_ent=6)


    # 2) collect unique entities (including vars)
    uniq: List[str] = []
    seen = set()
    for s, _, o in triples:
        for e in (s, o):
            if e not in seen:
                uniq.append(e)
                seen.add(e)

    # 3) slot rules first (direct typing)
    entity2type: Dict[str, Tuple[str, str]] = {}
    for e in uniq:
        st = slot_type(e)
        if st is not None:
            entity2type[e] = st

    # 4) remaining entities -> doc-style pipeline
    need = [e for e in uniq if e not in entity2type]

    if need:
        # 4.1 embeddings: variables remove '?', constants keep as-is
        embed_inputs = [f"entity: {strip_qmark(e) if is_var(e) else e}" for e in need]
        vecs = clients.embed_batch(embed_inputs, batch_size=embed_batch_size)

        # 4.2 L1 retrieve
        l1_cands = retriever.topk_l1(vecs, k=l1_topk)

        # 4.3 Collect ALL L1/L2 pairs and rank by embedding similarity
        all_pairs: List[str] = []
        for l1 in retriever.l2_meta.keys():
            for l2_item in retriever.l2_meta[l1]:
                all_pairs.append(f"{l1}/{l2_item['label']}")
        all_pairs.append("OTHER/Other")  # Fallback
        
        # Embed all pairs once
        pair_embed_inputs = [f"type: {p}" for p in all_pairs]
        pair_vecs = clients.embed_batch(pair_embed_inputs, batch_size=embed_batch_size)
        
        # For each entity, compute similarity to all pairs and rank
        pair_cands_per_ent: List[List[str]] = []
        for i, ent in enumerate(need):
            ent_vec = vecs[i:i+1]  # Shape: (1, dim)
            # Compute cosine similarity (vectors are already L2-normalized)
            sims = np.dot(pair_vecs, ent_vec.T).flatten()  # Shape: (num_pairs,)
            # Sort by similarity descending
            sorted_indices = np.argsort(-sims)
            ranked_pairs = [all_pairs[idx] for idx in sorted_indices[:max_pair_candidates]]
            pair_cands_per_ent.append(ranked_pairs)

        # 4.6 LLM choose final pair with context
        pair_prompts = []
        for e, pairs in zip(need, pair_cands_per_ent):
            pair_prompts.append(
                pair_choice_prompt(
                    target=e,
                    ctx=ent_ctx.get(e, ""),
                    pairs=pairs[:max_pair_candidates],
                )
            )
        pair_outs = clients.chat_batch(pair_prompts, batch_size=llm_batch_size)

        # 4.7 validate & fill entity2type
        for e, out, pairs in zip(need, pair_outs, pair_cands_per_ent):
            l1, l2 = _parse_pair_choice(out, fallback_pair=pairs[0])
            if f"{l1}/{l2}" not in set(pairs):
                l1, l2 = pairs[0].split("/", 1)
            entity2type[e] = (l1, l2)

    # 5) output typed_triples and type_only_triples
    typed_triples: List[Triple] = []
    type_only_triples: List[Triple] = []
    for s, p, o in triples:
        s_l1, s_l2 = entity2type.get(s, ("OTHER", "Other"))
        o_l1, o_l2 = entity2type.get(o, ("OTHER", "Other"))
        typed_triples.append([f"{s}<{s_l1}/{s_l2}>", p, f"{o}<{o_l1}/{o_l2}>"])
        type_only_triples.append([f"{s_l1}/{s_l2}", p, f"{o_l1}/{o_l2}"])
    
    return {
        "typed_triples": typed_triples,
        "type_only_triples": type_only_triples,
        "entity2type": {k: f"{v[0]}/{v[1]}" for k, v in entity2type.items()},
    }


# ============================================================
# 7) Worker function for parallel processing
# ============================================================
def process_single_record(line: str, retriever: FaissTypeRetriever, clients: VLLMClients) -> Optional[str]:
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        return None
        
    record_id = record.get("id")
    question = record.get("question", "")
    answers = record.get("answer", [])
    decomposable = record.get("decomposable", False)
    input_triples = record.get("triples", [])

    # Extract raw_triples and subqueries
    raw_triples = [t["raw_triple"] for t in input_triples]
    subqueries = [t.get("sub_query", "") for t in input_triples]

    # Retry logic for main processing
    result = None
    last_err = None
    
    # Simple exponential backoff retry
    for attempt in range(5):
        try:
            result = subquery_triples_to_abs_triples(
                triples=raw_triples,
                subqueries=subqueries,
                retriever=retriever,
                clients=clients,
            )
            break
        except Exception as e:
            last_err = e
            if attempt < 4:
                time.sleep(2 ** attempt)  # Exponential backoff
            continue
    
    if result is not None:
        typed_triples = result["typed_triples"]
        type_only_triples = result["type_only_triples"]
        entity2type = result["entity2type"]
    else:
        print(f"[FAILED] id={record_id}, question='{question}': {last_err}")
        # Fallback to raw triples
        typed_triples = raw_triples
        type_only_triples = raw_triples
        entity2type = {}

    # Build output triples
    output_triples = []
    for i, t in enumerate(input_triples):
        output_triples.append({
            "sub_query_idx": t.get("sub_query_idx", i),
            "sub_query": t.get("sub_query", ""),
            "raw_triple": t["raw_triple"],
            "typed_triple": typed_triples[i] if i < len(typed_triples) else t["raw_triple"],
            "type_only_triple": type_only_triples[i] if i < len(type_only_triples) else t["raw_triple"],
        })

    output_record = {
        "id": record_id,
        "question": question,
        "answer": answers,
        "decomposable": decomposable,
        "triples": output_triples,
        "entity2type": entity2type,
    }
    return json.dumps(output_record, ensure_ascii=False)


# ============================================================
# 8) CLI Entry Point
# ============================================================
if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from tqdm import tqdm
    from concurrent.futures import ThreadPoolExecutor, as_completed

    parser = argparse.ArgumentParser(description="Add type annotations to query triples.")
    parser.add_argument("--input_file", type=str, required=True, help="Input JSONL (from query_triple.py).")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSONL with typed query triples.")
    parser.add_argument("--faiss_dir", type=str, default="type_faiss", help="Directory containing FAISS indexes.")
    parser.add_argument("--sample", type=int, default=None, help="Process only first N records (default: all).")
    parser.add_argument("--embed_url", type=str, default=EMBED_API_BASE, help="Embedding API base URL.")
    parser.add_argument("--chat_url", type=str, default=CHAT_API_BASE, help="Chat API base URL.")
    parser.add_argument("--embed_model", type=str, default=EMBED_MODEL, help="Embedding model name.")
    parser.add_argument("--chat_model", type=str, default=CHAT_MODEL, help="Chat model name.")
    parser.add_argument("--workers", type=int, default=16, help="Number of parallel workers (default: 16).")
    args = parser.parse_args()

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)

    retriever = FaissTypeRetriever(args.faiss_dir)
    clients = VLLMClients(
        embed_base_url=args.embed_url,
        chat_base_url=args.chat_url,
        api_key=API_KEY,
        embed_model=args.embed_model,
        chat_model=args.chat_model,
    )



    # Read all lines first
    lines = []
    with open(args.input_file, "r", encoding="utf-8") as infile:
        if args.sample is not None:
             for i, line in enumerate(infile):
                 if i >= args.sample: break
                 lines.append(line)
        else:
             lines = infile.readlines()

    print(f"Loaded {len(lines)} records. Processing with {args.workers} workers...")

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_record, line, retriever, clients): i for i, line in enumerate(lines)}

        with open(args.output_file, "w", encoding="utf-8") as outfile, \
             tqdm(total=len(lines), desc="Typing query triples", unit="query") as pbar:

            output_buffer = {}
            
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    res_str = future.result()
                    if res_str:
                         output_buffer[idx] = res_str
                except Exception as e:
                    print(f"[ERROR] Worker failed for index {idx}: {e}")
                
                pbar.update(1)
            
            # Write in order
            for i in range(len(lines)):
                if i in output_buffer:
                    outfile.write(output_buffer[i] + "\n")
                    outfile.flush()

    print(f"Done. Output: {args.output_file}")

