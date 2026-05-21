#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import re
import time
import argparse
import threading
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# 1) Import from utils & config
import sys
current_dir = Path(__file__).resolve().parent
if str(current_dir.parent) not in sys.path:
    sys.path.append(str(current_dir.parent))

from config import CHAT_API_BASE, CHAT_MODEL, API_KEY
from utils import VLLMClients, strip_thinking_tags, extract_json_obj

# 2) Import logic directly from sister script (query_typed.py)
#    We import: Slot logic, Context builder, JSON parsers
try:
    from main_pipline.query_typed import (
        slot_type,
        is_var,
        strip_qmark,
        build_entity_context_from_subqueries,
        _parse_l1_top3,
        _parse_pair_choice,
        _dedup_keep_order
    )
except ImportError:
    # If running from absQA root without main_pipline in path
    import sys
    sys.path.append(str(current_dir))
    from query_typed import (
        slot_type,
        is_var,
        strip_qmark,
        build_entity_context_from_subqueries,
        _parse_l1_top3,
        _parse_pair_choice,
        _dedup_keep_order
    )

Triple = List[str]  # ["s","p","o"]


# ============================================================
# Main Logic (No Embedding)
# ============================================================
def subquery_triples_to_abs_triples_noembed(
    triples: List[Triple],
    subqueries: Optional[List[str]],
    taxonomy: Dict[str, List[str]],
    clients: VLLMClients,
    llm_batch_size: int = 128,
) -> Dict[str, Any]:
    """
    Input:  List[["Steve Jobs","found","?entity"], ["?entity","established","?date"], ...]
    Output: Typed triples without using embedding retrieval
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
    need: List[str] = []
    
    for e in uniq:
        st = slot_type(e)
        if st is not None:
            entity2type[e] = st
        else:
            need.append(e)

    # 4) remaining entities -> LLM with full taxonomy
    if need:
        l1_types = list(taxonomy.keys())
        
        # 4.1 LLM Choose L1 Top3
        l1_prompts = []
        for e in need:
            ctx = ent_ctx.get(e, "")
            l1_prompts.append(f"""Task: Classify the target entity into the best 3 categories (L1) from the list below.
Main Goal: Infer semantic type for query variable or entity.

Target: "{e}"
Context (from sub-queries):
{ctx}

Candidates categories (L1): {json.dumps(l1_types)}

Rules:
- Output MUST be valid JSON: {{"L1_top3": ["Cat1", "Cat2", "Cat3"]}}
- Choose ONLY from the candidate list.
- Order them from most likely to least likely.
- If context implies a specific type (e.g. "religion" -> CONCEPT), prioritize that over generic types.

Output JSON:""")
        
        l1_outs = clients.chat_batch(l1_prompts, batch_size=llm_batch_size)
        l1_top3_list = [_parse_l1_top3(out, l1_types) for out in l1_outs]

        # 4.2 LLM Choose Final (L1, L2)
        pair_prompts = []
        pair_candidates_per_ent = []
        for e, top3 in zip(need, l1_top3_list):
            ctx = ent_ctx.get(e, "")
            pairs = []
            for l1 in top3:
                for l2 in taxonomy.get(l1, ["Other"]):
                    pairs.append(f"{l1}/{l2}")
            pair_candidates_per_ent.append(pairs)

            pair_prompts.append(f"""Task: Select the most accurate (L1, L2) category pair for the target entity.
Main Goal: Infer semantic type for query variable or entity.

Target: "{e}"
Context:
{ctx}

Candidate Pairs (L1/L2):
{json.dumps(pairs[:100])}

Rules:
- Output MUST be valid JSON: {{"L1": "Selected_L1", "L2": "Selected_L2"}}
- You MUST choose a pair that exists in the candidate list.
- Prioritize specific semantic types over generic ones.

Output JSON:""")

        pair_outs = clients.chat_batch(pair_prompts, batch_size=llm_batch_size)
        
        for i, e in enumerate(need):
            fallback = pair_candidates_per_ent[i][0].split("/")
            if len(fallback) < 2: fallback = ("OTHER", "Other")
            
            l1, l2 = _parse_pair_choice(pair_outs[i], "/".join(fallback))
            # Validate
            if f"{l1}/{l2}" not in set(pair_candidates_per_ent[i]):
                l1, l2 = fallback
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
# Worker
# ============================================================
def process_single_record_noembed(
    line: str,
    taxonomy: Dict[str, List[str]],
    clients: VLLMClients
) -> Optional[str]:
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

    # Retry logic
    result = None
    last_err = None
    
    for attempt in range(5):
        try:
            result = subquery_triples_to_abs_triples_noembed(
                triples=raw_triples,
                subqueries=subqueries,
                taxonomy=taxonomy,
                clients=clients,
            )
            break
        except Exception as e:
            last_err = e
            if attempt < 4:
                time.sleep(2 ** attempt)
            continue
    
    if result is not None:
        typed_triples = result["typed_triples"]
        type_only_triples = result["type_only_triples"]
        entity2type = result["entity2type"]
    else:
        print(f"[FAILED] id={record_id}, question='{question}': {last_err}")
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
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Add type annotations to query triples (No Embedding).")
    parser.add_argument("--input_file", type=str, required=True, help="Input JSONL.")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSONL.")
    parser.add_argument("--taxonomy_file", type=str, default="type_faiss/taxonomy.json", help="Path to taxonomy JSON.")
    parser.add_argument("--chat_url", type=str, default=CHAT_API_BASE, help="Chat API base URL.")
    parser.add_argument("--chat_model", type=str, default=CHAT_MODEL, help="Chat model name.")
    parser.add_argument("--workers", type=int, default=16, help="Number of parallel workers.")
    parser.add_argument("--sample", type=int, default=None, help="Process only first N records.")
    args = parser.parse_args()

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)

    # Load Taxonomy
    with open(args.taxonomy_file, "r") as f:
        taxonomy = json.load(f)

    # Patch thinking tags
    if not hasattr(VLLMClients.chat_batch, "_is_patched"):
        original_chat_batch = VLLMClients.chat_batch
        def patched_chat_batch(self, prompts, batch_size=64):
            outs = original_chat_batch(self, prompts, batch_size)
            return [strip_thinking_tags(o) for o in outs]
        patched_chat_batch._is_patched = True
        VLLMClients.chat_batch = patched_chat_batch

    clients = VLLMClients(
        embed_base_url="", # Not used
        chat_base_url=args.chat_url,
        api_key=API_KEY,
        embed_model="", # Not used
        chat_model=args.chat_model,
    )

    # Read all lines
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
        futures = {executor.submit(process_single_record_noembed, line, taxonomy, clients): i for i, line in enumerate(lines)}
        
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


if __name__ == "__main__":
    main()
