import json
import os
import re
import threading
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import faiss
from openai import OpenAI

Triple = List[str]  # ["s","p","o"]


# ============================================================
# 规则优先（减少 LLM/embedding 压力）
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
# Simple in-memory cache for entity -> type mappings
# ============================================================
class EntityTypeCache:
    """Simple in-memory cache with optional JSON persistence."""
    
    def __init__(self, cache_file: Optional[str] = None):
        self._cache: Dict[str, Tuple[str, str]] = {}
        self._cache_file = cache_file
        if cache_file and os.path.exists(cache_file):
            self._load()
    
    def _load(self):
        try:
            with open(self._cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                for k, v in data.items():
                    if isinstance(v, list) and len(v) == 2:
                        self._cache[k] = (v[0], v[1])
        except Exception:
            pass
    
    def save(self):
        if self._cache_file:
            with open(self._cache_file, "w", encoding="utf-8") as f:
                json.dump({k: list(v) for k, v in self._cache.items()}, f, ensure_ascii=False, indent=2)
    
    def get(self, entity: str) -> Optional[Tuple[str, str]]:
        return self._cache.get(entity)
    
    def set(self, entity: str, type_pair: Tuple[str, str]):
        self._cache[entity] = type_pair
    
    def get_many(self, entities: List[str]) -> Tuple[Dict[str, Tuple[str, str]], List[str]]:
        """Returns (cached_results, uncached_entities)"""
        cached = {}
        uncached = []
        for e in entities:
            if e in self._cache:
                cached[e] = self._cache[e]
            else:
                uncached.append(e)
        return cached, uncached
    
    def set_many(self, mapping: Dict[str, Tuple[str, str]]):
        self._cache.update(mapping)
    
    def __len__(self):
        return len(self._cache)
    
    def stats(self) -> str:
        return f"Cache size: {len(self._cache)} entities"


import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
if str(current_dir.parent) not in sys.path:
    sys.path.append(str(current_dir.parent))

from utils import VLLMClients, extract_json_obj, FaissTypeRetriever
from config import CHAT_API_BASE, EMBED_API_BASE, CHAT_MODEL, EMBED_MODEL, API_KEY


# ============================================================
# Helpers
# ============================================================
def _extract_json_obj(raw: str) -> Dict[str, Any]:
    """Extract a JSON object from raw string, handling nested braces correctly."""
    return extract_json_obj(raw)

def _dedup_keep_order(xs: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

def _build_entity_context_map(
    triples: List[Triple],
    contexts: Optional[List[Dict[str, str]]] = None,
    max_ctx_per_entity: int = 1,  # Only use first document's context
) -> Dict[str, str]:
    """
    Aggregate entity -> context snippets (title/evidence/triple) for disambiguation.
    contexts[i] optional: {"doc_title":..., "evidence_sentence":...}
    """
    buf: Dict[str, List[str]] = {}

    for i, (s, p, o) in enumerate(triples):
        title = ""
        ev = ""
        if contexts is not None and i < len(contexts) and contexts[i] is not None:
            title = (contexts[i].get("doc_title") or "").strip()
            ev = (contexts[i].get("evidence_sentence") or "").strip()

        def add(ent: str):
            buf.setdefault(ent, [])
            if len(buf[ent]) >= max_ctx_per_entity:
                return
            parts = []
            if title:
                parts.append(f'Title: "{title}"')
            if ev:
                parts.append(f'Evidence: "{ev}"')
            parts.append(f"Triple: {json.dumps([s, p, o], ensure_ascii=False)}")
            buf[ent].append("\n".join(parts))

        add(s)
        add(o)

    return {ent: "\n---\n".join(parts) for ent, parts in buf.items()}

def _parse_l1_top3(raw: str, candidates: List[str]) -> List[str]:
    """
    Expect JSON: {"L1_top3":["...","...","..."]}
    Must be subset of candidates; otherwise fallback to candidates.
    """
    obj = extract_json_obj(raw)
    top3 = obj.get("L1_top3", [])
    if not isinstance(top3, list):
        top3 = []
    top3 = [x for x in top3 if isinstance(x, str)]
    top3 = _dedup_keep_order(top3)
    # keep only candidate labels
    top3 = [x for x in top3 if x in candidates]

    # pad with candidates
    for c in candidates:
        if len(top3) >= 3:
            break
        if c not in top3:
            top3.append(c)

    return top3[:3] if top3 else candidates[:3]

def _parse_pair_choice(raw: str, fallback_pair: str) -> Tuple[str, str]:
    """
    Expect JSON: {"L1":"...","L2":"..."}
    """
    obj = extract_json_obj(raw)
    l1 = obj.get("L1", None)
    l2 = obj.get("L2", None)
    if isinstance(l1, str) and isinstance(l2, str):
        return l1, l2
    # fallback
    l1, l2 = fallback_pair.split("/", 1)
    return l1, l2


# ============================================================
# Main typing function (L1 -> output TOP3; L2 union over L1 top3; final choose pair with context)
# ============================================================
def type_triples_batch(
    triples: List[Triple],
    retriever: FaissTypeRetriever,
    clients: VLLMClients,
    *,
    contexts: Optional[List[Dict[str, str]]] = None,
    cache: Optional[EntityTypeCache] = None,
    l1_topk: int = 12,           # widen L1 candidate set
    l2_topk_each: int = 20,      # retrieve L2 topK for each of top3 L1
    embed_batch_size: int = 256,
    llm_batch_size: int = 64,
    max_pair_candidates: int = 60,   # cap pair list passed to LLM to control tokens
) -> Dict[str, Any]:
    """
    triples: List[["s","p","o"]]
    contexts (optional): List[{"doc_title":..., "evidence_sentence":...}] aligned with triples
    cache (optional): EntityTypeCache instance for caching entity -> type mappings
    """

    # 1) collect entities (dedup)
    ents: List[str] = []
    for s, p, o in triples:
        ents.append(s); ents.append(o)

    uniq: List[str] = []
    seen = set()
    for e in ents:
        if e not in seen:
            uniq.append(e); seen.add(e)

    # 2) build per-entity context (for disambiguation in LLM steps)
    entity_ctx = _build_entity_context_map(triples, contexts=contexts, max_ctx_per_entity=2)

    # 3) rule-first + cache check
    entity2type: Dict[str, Tuple[str, str]] = {}
    need: List[str] = []
    cache_hits = 0
    for e in uniq:
        rt = rule_type(e)
        if rt is not None:
            entity2type[e] = rt
        elif cache is not None:
            cached = cache.get(e)
            if cached is not None:
                entity2type[e] = cached
                cache_hits += 1
            else:
                need.append(e)
        else:
            need.append(e)

    if need:
        # 4) batch embedding(entities)  (entity-only; we compensate with larger candidate sets + context in LLM)
        ent_vecs = clients.embed_batch([f"entity: {e}" for e in need], batch_size=embed_batch_size)

        # 5) FAISS topk -> L1 candidates
        l1_cands = retriever.topk_l1(ent_vecs, k=l1_topk)

        # 6) LLM choose L1_top3 (WITH context)
        l1_prompts = []
        for e, cands in zip(need, l1_cands):
            ctx = entity_ctx.get(e, "")
            l1_prompts.append(
f"""Choose the best 3 L1 types for the entity, ordered best to worst.

Rules:
- Output MUST be valid JSON only: {{"L1_top3":["...","...","..."]}}
- MUST choose ONLY from candidates.
- MUST be unique and length=3.

Entity: "{e}"
Context:
{ctx}

Candidates: {json.dumps(cands, ensure_ascii=False)}

Output JSON:
"""
            )
        l1_outs = clients.chat_batch(l1_prompts, batch_size=llm_batch_size)
        l1_top3_list: List[List[str]] = []
        for out, cands in zip(l1_outs, l1_cands):
            l1_top3_list.append(_parse_l1_top3(out, candidates=cands))

        # 7) For each entity, retrieve L2 topK for each of its top3 L1, then union pairs
        expanded_vecs = np.repeat(ent_vecs, repeats=3, axis=0)
        expanded_l1 = [l1 for top3 in l1_top3_list for l1 in top3]  # length = 3*N
        expanded_l2_cands = retriever.topk_l2(expanded_vecs, expanded_l1, k=l2_topk_each)  # 3N lists

        pair_cands_per_ent: List[List[str]] = []
        for i in range(len(need)):
            pairs: List[str] = []
            seen_pair = set()
            for j in range(3):
                l1 = l1_top3_list[i][j]
                l2s = expanded_l2_cands[i * 3 + j]
                for l2 in l2s:
                    key = f"{l1}/{l2}"
                    if key not in seen_pair:
                        pairs.append(key)
                        seen_pair.add(key)
            if not pairs:
                pairs = ["OTHER/Other"]
            pair_cands_per_ent.append(pairs)

        # 8) LLM final choose (L1,L2) from union pairs (WITH context)
        pair_prompts = []
        for e, pairs in zip(need, pair_cands_per_ent):
            ctx = entity_ctx.get(e, "")
            pairs_trim = pairs[:max_pair_candidates]
            pair_prompts.append(
f"""Choose the best (L1,L2) type pair for the entity.

Rules:
- Output MUST be valid JSON only: {{"L1":"...","L2":"..."}}
- You MUST choose a pair from candidates.
- The output (L1,L2) must correspond to one of the candidate pairs.

Entity: "{e}"
Context:
{ctx}

Candidates (L1/L2 pairs): {json.dumps(pairs_trim, ensure_ascii=False)}

Output JSON:
"""
            )
        pair_outs = clients.chat_batch(pair_prompts, batch_size=llm_batch_size)

        # 9) fill entity2type (validate chosen pair is in candidates, else fallback)
        for e, out, pairs in zip(need, pair_outs, pair_cands_per_ent):
            l1, l2 = _parse_pair_choice(out, fallback_pair=pairs[0])
            if f"{l1}/{l2}" not in set(pairs):
                l1, l2 = pairs[0].split("/", 1)
            entity2type[e] = (l1, l2)
            # Update cache for newly typed entities
            if cache is not None:
                cache.set(e, (l1, l2))

    # 10) backfill typed triples (preserve order)
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


def process_single_record_typed(
    record: dict,
    ctxs_all: List[dict],
    retriever: "FaissTypeRetriever",
    clients: "VLLMClients",
    cache: Optional["EntityTypeCache"],
    l1_topk: int,
    l2_topk: int,
) -> Optional[dict]:
    """Process a single record and return the output dict."""
    question = record.get("question", "")
    answer = record.get("answer", [])
    triples_list = record.get("triples", [])
    ctxs = record.get("ctxs", [])

    if not triples_list:
        return {
            "question": question,
            "answer": answer,
            "triples": [],
            "entity2type": {}
        }

    # Extract raw triples and build contexts
    raw_triples = []
    contexts = []
    doc_ids = []
    for t in triples_list:
        doc_id = t.get("doc_id", 0)
        raw_triple = t.get("raw_triple", [])
        if len(raw_triple) == 3:
            raw_triples.append(raw_triple)
            doc_ids.append(doc_id)
            ctx_idx = doc_id - 1
            if 0 <= ctx_idx < len(ctxs):
                ctx = ctxs[ctx_idx]
                contexts.append({
                    "doc_title": ctx.get("title", ""),
                    "evidence_sentence": ctx.get("doc", "") or ctx.get("text", "")
                })
            else:
                contexts.append({"doc_title": "", "evidence_sentence": ""})

    if not raw_triples:
        return {
            "question": question,
            "answer": answer,
            "triples": [],
            "entity2type": {}
        }

    try:
        result = type_triples_batch(
            raw_triples,
            retriever,
            clients,
            contexts=contexts,
            cache=cache,
            l1_topk=l1_topk,
            l2_topk_each=l2_topk,
        )
        typed_triples = result["typed_triples"]
        type_only_triples = result["type_only_triples"]
        entity2type = result["entity2type"]
    except Exception as e:
        print(f"[FAILED] question='{question}': {e}")
        typed_triples = [[s, p, o] for s, p, o in raw_triples]
        type_only_triples = [["OTHER/Other", p, "OTHER/Other"] for s, p, o in raw_triples]
        entity2type = {}

    output_triples = []
    for i, (doc_id, raw_t, typed_t, type_only_t) in enumerate(zip(
        doc_ids, raw_triples, typed_triples, type_only_triples
    )):
        output_triples.append({
            "doc_id": doc_id,
            "raw_triple": raw_t,
            "typed_triple": typed_t,
            "type_only_triple": type_only_t
        })

    return {
        "question": question,
        "answer": answer,
        "triples": output_triples,
        "entity2type": entity2type
    }


# ============================================================
# 7) Main Entry Point (Parallel)
# ============================================================
def main():
    import argparse
    from pathlib import Path
    from tqdm import tqdm
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import re as regex
    import os
    import threading
    try:
        from config import EMBED_API_BASE, CHAT_API_BASE, EMBED_MODEL, CHAT_MODEL, API_KEY
        from utils import VLLMClients, FaissTypeRetriever
    except ImportError:
        from absQA.config import EMBED_API_BASE, CHAT_API_BASE, EMBED_MODEL, CHAT_MODEL, API_KEY
        from absQA.utils import VLLMClients, FaissTypeRetriever

    # Strips Qwen3 <think>...</think> tags from VLLMClients.chat_batch output.
    
    def strip_thinking_tags(text: str) -> str:
        """Remove Qwen3's <think>...</think> tags from output."""
        text = regex.sub(r'<think>.*?</think>', '', text, flags=regex.DOTALL)
        return text.strip()

    # Monkey-patch to strip thinking tags
    # Check if we need to patch (avoid double patching if run multiple times in some envs)
    if not hasattr(VLLMClients.chat_batch, "_is_patched"):
        original_chat_batch = VLLMClients.chat_batch
        def patched_chat_batch(self, prompts, batch_size=64):
            outs = original_chat_batch(self, prompts, batch_size)
            return [strip_thinking_tags(o) for o in outs]
        patched_chat_batch._is_patched = True
        VLLMClients.chat_batch = patched_chat_batch

    parser = argparse.ArgumentParser(description="Convert raw triples to typed triples using FAISS + LLM.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to Step 1 output JSONL.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to write typed triples JSONL.")
    parser.add_argument("--faiss_dir", type=str, default="type_faiss", help="Directory containing FAISS indexes.")
    parser.add_argument("--embed_url", type=str, default=EMBED_API_BASE, help="Embedding API base URL.")
    parser.add_argument("--chat_url", type=str, default=CHAT_API_BASE, help="Chat API base URL.")
    parser.add_argument("--embed_model", type=str, default=EMBED_MODEL, help="Embedding model name.")
    parser.add_argument("--chat_model", type=str, default=CHAT_MODEL, help="Chat model name.")
    parser.add_argument("--l1_topk", type=int, default=12, help="Top-K candidates for L1 retrieval.")
    parser.add_argument("--l2_topk", type=int, default=20, help="Top-K candidates for L2 retrieval per L1.")
    parser.add_argument("--sample", type=int, default=None, help="Process only first N records (default: all).")
    parser.add_argument("--start", type=int, default=0, help="Start processing from this index (0-indexed, default: 0).")
    parser.add_argument("--cache_file", type=str, default=None, help="Path to JSON cache file for entity types (optional).")
    parser.add_argument("--workers", type=int, default=16, help="Number of parallel workers for processing records.")
    args = parser.parse_args()

    # Initialize retriever and clients
    retriever = FaissTypeRetriever(args.faiss_dir)
    clients = VLLMClients(
        embed_base_url=args.embed_url,
        chat_base_url=args.chat_url,
        api_key=API_KEY,
        embed_model=args.embed_model,
        chat_model=args.chat_model,
    )

    # Initialize cache (optional)
    cache = EntityTypeCache(args.cache_file) if args.cache_file else None
    if cache:
        print(f"[Cache] Loaded {len(cache)} cached entity types from {args.cache_file}")

    # Ensure output directory exists
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)

    # Load all data
    with open(args.input_file, "r", encoding="utf-8") as infile:
        all_data = [json.loads(line) for line in infile]

    # Apply start and sample
    all_data = all_data[args.start:]
    if args.sample is not None:
        all_data = all_data[:args.sample]

    # Load existing results for resume capability
    processed_questions = set()
    existing_results = []
    if os.path.exists(args.output_file):
        with open(args.output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line.strip())
                    q = rec.get("question", "")
                    if q:
                        processed_questions.add(q)
                        existing_results.append(rec)
                except:
                    pass
        print(f"[Resume] Found {len(processed_questions)} already processed records in output file.")
    
    # Filter out already processed records
    remaining_data = [(idx, rec) for idx, rec in enumerate(all_data) 
                      if rec.get("question", "") not in processed_questions]
    
    print(f"Loaded {len(all_data)} records, {len(remaining_data)} remaining. Processing with {args.workers} workers...")
    
    if not remaining_data:
        print("All records already processed. Nothing to do.")
        return

    # Open output file in append mode for incremental saving
    outfile = open(args.output_file, "a", encoding="utf-8")
    save_lock = threading.Lock()
    saved_count = [0]  # Use list for mutable in closure
    
    def save_result(result):
        """Thread-safe incremental save"""
        with save_lock:
            outfile.write(json.dumps(result, ensure_ascii=False) + "\n")
            outfile.flush()
            saved_count[0] += 1
            # Save cache every 50 records
            if saved_count[0] % 50 == 0 and cache:
                cache.save()

    # Process in parallel
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_single_record_typed,
                record, [], retriever, clients, cache, args.l1_topk, args.l2_topk
            ): orig_idx
            for orig_idx, record in remaining_data
        }

        with tqdm(total=len(remaining_data), desc="Processing records", unit="record") as pbar:
            for future in as_completed(futures):
                orig_idx = futures[future]
                try:
                    result = future.result()
                    if result is not None:
                        save_result(result)
                except Exception as exc:
                    print(f"[ERROR] idx={orig_idx}: {exc}")
                pbar.update(1)

    outfile.close()

    # Save cache at the end
    if cache:
        cache.save()
        print(f"[Cache] Saved {len(cache)} entity types to {args.cache_file}")

    print(f"Done. Output: {args.output_file} (total: {len(existing_results) + saved_count[0]} records)")


if __name__ == "__main__":
    main()
