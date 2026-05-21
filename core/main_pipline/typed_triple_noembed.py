#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import re
import time
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from tqdm import tqdm
from openai import OpenAI

Triple = List[str]  # ["s","p","o"]

# ============================================================
# 1) Rule-based typing (Years, Dates, Percentages)
# ============================================================
_YEAR_RE = re.compile(r"^(?:\d{4})$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_PERCENT_RE = re.compile(r"^-?\d+(?:\.\d+)?%$")

def rule_type(entity: str) -> Optional[Tuple[str, str]]:
    e = entity.strip()
    if _YEAR_RE.match(e):
        return ("TIME", "Year")
    if _DATE_RE.match(e):
        return ("TIME", "Date")
    if _PERCENT_RE.match(e):
        return ("QUANTITY", "Percentage")
    return None

# ============================================================
# 2) LLM Clients
# ============================================================
class VLLMClients:
    def __init__(self, base_url: str, api_key: str, model: str):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def chat_batch(self, prompts: List[str], batch_size: int = 64, max_concurrent: int = 64) -> List[str]:
        import asyncio
        import httpx

        async def call_single(client: httpx.AsyncClient, idx: int, prompt: str, semaphore: asyncio.Semaphore, max_retries: int = 10) -> Tuple[int, str]:
            async with semaphore:
                payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "Return ONLY valid JSON as instructed. No extra text."},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.0,  # Use 0.0 for deterministic classification
                    "max_tokens": 512,
                }
                url = f"{self.client.base_url}chat/completions"
                for attempt in range(max_retries):
                    try:
                        resp = await client.post(url, json=payload, timeout=120.0)
                        resp.raise_for_status()
                        data = resp.json()
                        content = data["choices"][0]["message"]["content"]
                        return idx, content.strip()
                    except Exception as e:
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2 ** attempt)
                        else:
                            raise e

        async def run_all():
            semaphore = asyncio.Semaphore(max_concurrent)
            async with httpx.AsyncClient() as client:
                tasks = [call_single(client, i, p, semaphore) for i, p in enumerate(prompts)]
                results = await asyncio.gather(*tasks, return_exceptions=True)
            return results

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            raw_results = loop.run_until_complete(run_all())
        finally:
            loop.close()

        output: Dict[int, str] = {}
        for r in raw_results:
            if isinstance(r, Exception):
                raise r
            idx, content = r
            output[idx] = content

        return [output[i] for i in range(len(prompts))]

# ============================================================
# 3) JSON and Prompt Helpers
# ============================================================
def _extract_json_obj(raw: str) -> Dict[str, Any]:
    raw = (raw or "").strip()
    l = raw.find("{")
    r = raw.rfind("}")
    if l == -1 or r == -1 or r <= l:
        # Check if it's potentially wrapped in markdown code blocks
        match = re.search(r'```json\s*(\{.*?\})\s*```', raw, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        raise ValueError(f"Bad JSON object: {raw[:200]!r}")
    return json.loads(raw[l:r + 1])

def _parse_l1_top3(raw: str, candidates: List[str]) -> List[str]:
    try:
        obj = _extract_json_obj(raw)
        top3 = obj.get("L1_top3", [])
        if not isinstance(top3, list):
            top3 = []
        # Support both full string match and case-insensitive/partial if needed, but here we enforce exact
        top3 = [x for x in top3 if x in candidates]
        # Dedup keeping order
        seen = set()
        final_top3 = []
        for x in top3:
            if x not in seen:
                final_top3.append(x)
                seen.add(x)
        # Pad if needed
        for c in candidates:
            if len(final_top3) >= 3: break
            if c not in final_top3: final_top3.append(c)
        return final_top3[:3]
    except:
        return candidates[:3]

def _parse_pair_choice(raw: str, fallback_pair: Tuple[str, str]) -> Tuple[str, str]:
    try:
        obj = _extract_json_obj(raw)
        l1 = obj.get("L1")
        l2 = obj.get("L2")
        if l1 and l2:
            return str(l1), str(l2)
    except:
        pass
    return fallback_pair

# ============================================================
# 4) Main Logic
# ============================================================
def type_triples_noembed(
    triples: List[Triple],
    taxonomy: Dict[str, List[str]],
    clients: VLLMClients,
    question: str = "",
    subqueries: Optional[List[str]] = None,
    llm_batch_size: int = 64,
) -> Dict[str, Any]:
    
    # 1) collect entities (dedup)
    uniq: List[str] = []
    seen = set()
    for s, p, o in triples:
        for e in (s, o):
            if e not in seen:
                uniq.append(e); seen.add(e)

    # 2) build entity context (relevant triples and subqueries)
    entity_ctx: Dict[str, List[str]] = {}
    for i, (s, p, o) in enumerate(triples):
        ctx_line = f"Triple: ({s}, {p}, {o})"
        if subqueries and i < len(subqueries):
            ctx_line += f" | Subquery context: {subqueries[i]}"
        for e in (s, o):
            entity_ctx.setdefault(e, []).append(ctx_line)
    
    entity_ctx_str = {k: "\n".join(v[:5]) for k, v in entity_ctx.items()}

    # 3) rule-first
    entity2type: Dict[str, Tuple[str, str]] = {}
    need: List[str] = []
    for e in uniq:
        rt = rule_type(e)
        if rt:
            entity2type[e] = rt
        else:
            need.append(e)

    if need:
        l1_types = list(taxonomy.keys())
        
        # Phase 1: LLM Choose L1 Top3
        l1_prompts = []
        for e in need:
            ctx = entity_ctx_str.get(e, "No additional context.")
            l1_prompts.append(f"""Task: Classify the entity into the best 3 categories from the list below.
Main Query: {question}
Entity: "{e}"
Local Context:
{ctx}

Candidates categories (L1): {json.dumps(l1_types)}

Rules:
- Output MUST be valid JSON: {{"L1_top3": ["Cat1", "Cat2", "Cat3"]}}
- Choose ONLY from the candidate list.
- Order them from most likely to least likely.

Output JSON:""")
        
        l1_outs = clients.chat_batch(l1_prompts, batch_size=llm_batch_size)
        l1_top3_list = [_parse_l1_top3(out, l1_types) for out in l1_outs]

        # Phase 2: LLM Choose Final (L1, L2)
        pair_prompts = []
        pair_candidates_per_ent = []
        for e, top3 in zip(need, l1_top3_list):
            ctx = entity_ctx_str.get(e, "No additional context.")
            pairs = []
            for l1 in top3:
                for l2 in taxonomy.get(l1, ["Other"]):
                    pairs.append(f"{l1}/{l2}")
            pair_candidates_per_ent.append(pairs)

            pair_prompts.append(f"""Task: Select the most accurate (L1, L2) category pair for the entity.
Main Query: {question}
Entity: "{e}"
Local Context:
{ctx}

Candidate Pairs (L1/L2):
{json.dumps(pairs[:100])}

Rules:
- Output MUST be valid JSON: {{"L1": "Selected_L1", "L2": "Selected_L2"}}
- You MUST choose a pair that exists in the candidate list.

Output JSON:""")

        pair_outs = clients.chat_batch(pair_prompts, batch_size=llm_batch_size)
        
        for i, e in enumerate(need):
            fallback = pair_candidates_per_ent[i][0].split("/")
            l1, l2 = _parse_pair_choice(pair_outs[i], (fallback[0], fallback[1]))
            # Validate
            if f"{l1}/{l2}" not in set(pair_candidates_per_ent[i]):
                l1, l2 = fallback
            entity2type[e] = (l1, l2)

    # 4) format result
    typed_triples = []
    for s, p, o in triples:
        s_l1, s_l2 = entity2type.get(s, ("OTHER", "Other"))
        o_l1, o_l2 = entity2type.get(o, ("OTHER", "Other"))
        typed_triples.append([f"{s}<{s_l1}/{s_l2}>", p, f"{o}<{o_l1}/{o_l2}>"])

    return {
        "typed_triples": typed_triples,
        "entity2type": {k: f"{v[0]}/{v[1]}" for k, v in entity2type.items()}
    }

# ============================================================
# 5) CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--taxonomy_file", type=str, default="type_faiss/taxonomy.json")
    parser.add_argument("--chat_url", type=str, default="http://localhost:1225/v1")
    parser.add_argument("--chat_model", type=str, default="Qwen2.5-72B-Instruct")
    parser.add_argument("--workers", type=int, default=1) # Note: we use internal async batching
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args()

    # Load Taxonomy
    with open(args.taxonomy_file, "r") as f:
        taxonomy = json.load(f)

    clients = VLLMClients(base_url=args.chat_url, api_key="EMPTY", model=args.chat_model)

    results = []
    with open(args.input_file, "r") as f:
        lines = f.readlines()
        if args.sample:
            lines = lines[:args.sample]
        
        for line in tqdm(lines, desc="Typing entities"):
            record = json.loads(line)
            question = record.get("question", "")
            # Input triples can be in different formats depending on pipeline step
            input_triples_full = record.get("triples", [])
            
            # Handle list of dicts or list of lists
            raw_triples = []
            subqueries = []
            for t in input_triples_full:
                if isinstance(t, dict):
                    raw_triples.append(t["raw_triple"])
                    subqueries.append(t.get("sub_query", ""))
                else:
                    raw_triples.append(t)
            
            typed_res = type_triples_noembed(
                triples=raw_triples,
                taxonomy=taxonomy,
                clients=clients,
                question=question,
                subqueries=subqueries if subqueries else None
            )

            # Update record
            if isinstance(input_triples_full[0], dict):
                for i, t in enumerate(input_triples_full):
                    t["typed_triple"] = typed_res["typed_triples"][i]
            else:
                record["typed_triples"] = typed_res["typed_triples"]
            
            record["entity2type"] = typed_res["entity2type"]
            results.append(record)

    with open(args.output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Typed {len(results)} samples. Output saved to {args.output_file}")
