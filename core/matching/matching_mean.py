#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
from openai import OpenAI

# Imported from config
import sys
from pathlib import Path

# Add parent directory to sys.path to allow imports from absQA
current_dir = Path(__file__).resolve().parent
if str(current_dir.parent) not in sys.path:
    sys.path.append(str(current_dir.parent))

from config import EMBED_API_BASE, EMBED_MODEL, API_KEY

# -----------------------------
# Data structures
# -----------------------------
Triple = Tuple[str, str, str]  # (s, p, o)

@dataclass(frozen=True)
class QueryTriple:
    raw: Triple              # used for embeddings
    typed: Triple            # use typed s/o, keep p as raw (or typed too if you want)

@dataclass(frozen=True)
class DocTriple:
    doc_id: str
    raw: Triple              # used for embeddings
    typed: Triple            # typed s/o, keep p as raw (or typed too)


# -----------------------------
# Embedding client + cache
# -----------------------------
class Embedder:
    def __init__(self, base_url: str = EMBED_API_BASE, api_key: str = API_KEY, model: str = EMBED_MODEL, batch_size: int = 256):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.batch_size = batch_size
        self.cache: Dict[str, np.ndarray] = {}

    def embed_texts(self, texts: List[str]) -> None:
        """Populate cache for any texts not embedded yet (normalized vectors)."""
        missing = [t for t in texts if t not in self.cache]
        if not missing:
            return

        for i in range(0, len(missing), self.batch_size):
            batch = missing[i:i + self.batch_size]
            resp = self.client.embeddings.create(model=self.model, input=batch)
            resp.data.sort(key=lambda x: x.index)
            vecs = np.asarray([d.embedding for d in resp.data], dtype=np.float32)
            # faiss.normalize_L2(vecs)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / (norms + 1e-10)
            for t, v in zip(batch, vecs):
                self.cache[t] = v

    def get_vec(self, text: str) -> np.ndarray:
        v = self.cache.get(text)
        if v is None:
            self.embed_texts([text])
            v = self.cache[text]
        return v


# -----------------------------
# Type parsing + typed similarity
# -----------------------------
def parse_type(t: str) -> Tuple[str, str]:
    """
    Parse "L1/L2". Robust fallback.
    """
    t = (t or "").strip()
    if "/" not in t:
        # allow "PERSON" only
        return (t.upper() if t else "OTHER", "Other")
    l1, l2 = t.split("/", 1)
    l1 = (l1 or "").strip()
    l2 = (l2 or "").strip()
    return (l1, l2)

def sim_type_entity(
    q_type: str,
    d_type: str,
    *,
    w_l1: float = 0.4,
    w_l2: float = 0.6,
) -> float:
    """
    L1 score + L2 score weighted.
    """
    ql1, ql2 = parse_type(q_type)
    dl1, dl2 = parse_type(d_type)

    s_l1 = 1.0 if ql1 == dl1 else 0.0
    s_l2 = 1.0 if ql2 == dl2 else 0.0
    return w_l1 * s_l1 + w_l2 * s_l2

def typed_score_triple(
    q_typed: Triple,
    d_typed: Triple,
    *,
    w_s: float = 0.5,
    w_o: float = 0.5,
    w_l1: float = 0.4,
    w_l2: float = 0.6,
) -> float:
    """
    Typed score uses only subject/object types. Predicate NOT used (per your choice).
    """
    qs, _, qo = q_typed
    ds, _, do = d_typed

    s_sim = sim_type_entity(qs, ds, w_l1=w_l1, w_l2=w_l2)
    o_sim = sim_type_entity(qo, do, w_l1=w_l1, w_l2=w_l2)
    return w_s * s_sim + w_o * o_sim


# -----------------------------
# Embedding score (S/P/O separately)
# -----------------------------
def emb_key(role: str, text: str) -> str:
    return f"{role}: {text}"

def emb_score_triple(
    q_raw: Triple,
    d_raw: Triple,
    embedder: Embedder,
    *,
    w_s: float = 0.3,
    w_p: float = 0.4,
    w_o: float = 0.3,
) -> float:
    qs, qp, qo = q_raw
    ds, dp, do = d_raw

    v_qs = embedder.get_vec(emb_key("S", qs))
    v_qp = embedder.get_vec(emb_key("P", qp))
    v_qo = embedder.get_vec(emb_key("O", qo))

    v_ds = embedder.get_vec(emb_key("S", ds))
    v_dp = embedder.get_vec(emb_key("P", dp))
    v_do = embedder.get_vec(emb_key("O", do))

    # cosine via dot because normalized
    s = float(np.dot(v_qs, v_ds))
    p = float(np.dot(v_qp, v_dp))
    o = float(np.dot(v_qo, v_do))
    return w_s * s + w_p * p + w_o * o


# -----------------------------
# Main ranking logic
# -----------------------------
@dataclass
class MatchResult:
    q_idx: int
    d_idx: int
    doc_id: str
    score: float
    typed: float
    emb: float

def rank_docs_by_triple_matching(
    query_triples: List[QueryTriple],
    doc_triples: List[DocTriple],
    embedder: Embedder,
    *,
    alpha_type: float = 0.5,        # final = alpha*typed + (1-alpha)*emb
    # typed weights
    type_w_s: float = 0.5,
    type_w_o: float = 0.5,
    type_w_l1: float = 0.4,
    type_w_l2: float = 0.6,
    # embedding weights
    emb_w_s: float = 0.3,
    emb_w_p: float = 0.4,
    emb_w_o: float = 0.3,
    # triple threshold for sum aggregation
    threshold: float = 0.5,         # only sum triples with score > threshold
) -> Tuple[List[Tuple[str, float]], List[str], List[MatchResult], Dict[str, float]]:
    """
    Returns:
      ranked_docs: [(doc_id, doc_score), ...] sorted desc
      kept_doc_ids: [doc_id ...] after threshold
      global_matches: all (q,d) matches sorted desc by score
      doc_scores: dict doc_id -> score
    """
    Q = len(query_triples)
    M = len(doc_triples)
    if Q == 0 or M == 0:
        return [], [], [], {}

    # 0) Pre-embed all needed texts (big speedup)
    all_texts: List[str] = []
    for qt in query_triples:
        s, p, o = qt.raw
        all_texts.extend([emb_key("S", s), emb_key("P", p), emb_key("O", o)])
    for dt in doc_triples:
        s, p, o = dt.raw
        all_texts.extend([emb_key("S", s), emb_key("P", p), emb_key("O", o)])
    # dedup
    all_texts = list(dict.fromkeys(all_texts))
    embedder.embed_texts(all_texts)

    # 1) Compute all matches Q x M
    global_matches: List[MatchResult] = []
    for i, qt in enumerate(query_triples):
        for j, dt in enumerate(doc_triples):
            t = typed_score_triple(
                qt.typed, dt.typed,
                w_s=type_w_s, w_o=type_w_o,
                w_l1=type_w_l1, w_l2=type_w_l2
            )
            e = emb_score_triple(
                qt.raw, dt.raw, embedder,
                w_s=emb_w_s, w_p=emb_w_p, w_o=emb_w_o
            )
            s = alpha_type * t + (1.0 - alpha_type) * e
            global_matches.append(MatchResult(
                q_idx=i, d_idx=j, doc_id=dt.doc_id,
                score=float(s), typed=float(t), emb=float(e)
            ))

    # 2) Sort global matches (triple-level ranking)
    global_matches.sort(key=lambda x: x.score, reverse=True)

    # 3) Doc aggregation from per-query best matches inside doc
    # Build doc_id -> list of indices of doc_triples
    doc2didxs: Dict[str, List[int]] = {}
    for j, dt in enumerate(doc_triples):
        doc2didxs.setdefault(dt.doc_id, []).append(j)

    doc_scores: Dict[str, float] = {}
    doc_score_details: Dict[str, Dict] = {}  # doc_id -> {subquery_0: score, subquery_1: score, ..., final_score: score}
    
    # Build a matrix-like dict keyed by (q_idx, d_idx) -> MatchResult (full)
    match_map: Dict[Tuple[int, int], MatchResult] = {(m.q_idx, m.d_idx): m for m in global_matches}

    # Average-over-Threshold Aggregation
    # Collect all triple scores > threshold for each document, then compute mean
    for doc_id, didxs in doc2didxs.items():
        all_passing_scores = []  # All triple scores above threshold for this doc
        subquery_scores = []  # For detailed breakdown
        
        for qi in range(Q):
            qi_passing_scores = []
            for dj in didxs:
                m = match_map[(qi, dj)]
                if m.score > threshold:
                    qi_passing_scores.append(m.score)
                    all_passing_scores.append(m.score)
            # Subquery score: average of passing triples for this subquery (or 0 if none)
            sq_score = sum(qi_passing_scores) / len(qi_passing_scores) if qi_passing_scores else 0.0
            subquery_scores.append(sq_score)
        
        # Aggregate: average of all passing triple scores (or 0 if none)
        final_score = sum(all_passing_scores) / len(all_passing_scores) if all_passing_scores else 0.0
        doc_scores[doc_id] = final_score
        
        # Store detailed breakdown
        detail = {"final_score": final_score, "num_passing_triples": len(all_passing_scores)}
        for i, sq_score in enumerate(subquery_scores):
            detail[f"subquery_{i}"] = sq_score
        doc_score_details[doc_id] = detail

    ranked_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    kept_doc_ids = [doc_id for doc_id, _ in ranked_docs]  # Keep all, threshold already applied per-triple


    # Return doc_score_details as the 5th element
    return ranked_docs, kept_doc_ids, global_matches, doc_scores, doc_score_details


# -----------------------------
# CLI Entry Point
# -----------------------------
if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description="Match query triples with document triples and rank documents.")
    parser.add_argument("--query_file", type=str, required=True, help="Input JSONL from query_typed.py.")
    parser.add_argument("--doc_file", type=str, required=True, help="Input JSONL from typed_triple.py.")
    parser.add_argument("--data_file", type=str, required=True, help="Original data file with ctxs (for retrieval scores).")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSONL with ranked_doc_ids.")
    parser.add_argument("--embed_url", type=str, default=EMBED_API_BASE, help="Embedding server URL.")
    parser.add_argument("--sample", type=int, default=None, help="Process only first N records.")
    parser.add_argument("--alpha_type", type=float, default=0.5, help="Weight for typed score (vs embedding).")
    parser.add_argument("--threshold", type=float, default=0.3, help="Triple score threshold for sum aggregation.")
    parser.add_argument("--with_retrieval", action="store_true", help="If set, multiply final score by retrieval score.")
    args = parser.parse_args()

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)

    embedder = Embedder(
        base_url=args.embed_url,
        api_key=API_KEY,
        model=EMBED_MODEL,
        batch_size=256,
    )

    # Load all query records
    query_records = []
    with open(args.query_file, "r", encoding="utf-8") as f:
        for line in f:
            query_records.append(json.loads(line))

    # Load all doc records and build index by query_id
    # Note: typed_triple_result.jsonl has query_id as top-level "id", 
    # and doc_id inside each triple
    doc_records_by_query = {}
    with open(args.doc_file, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            query_id = rec["id"]
            doc_records_by_query[query_id] = rec

    # Load original data file to get retrieval scores from ctxs
    data_records_by_query = {}
    with open(args.data_file, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            query_id = rec["id"]
            data_records_by_query[query_id] = rec

    with open(args.output_file, "w", encoding="utf-8") as outfile, \
         tqdm(desc="Matching", unit="query") as pbar:

        for idx, q_rec in enumerate(query_records):
            if args.sample is not None and idx >= args.sample:
                break

            q_id = q_rec["id"]
            question = q_rec.get("question", "")
            answers = q_rec.get("answer", [])

            # Build QueryTriple list from query record
            query_triples: List[QueryTriple] = []
            for t in q_rec.get("triples", []):
                raw = tuple(t.get("raw_triple", ["", "", ""]))
                typed = tuple(t.get("type_only_triple", t.get("raw_triple", ["", "", ""])))
                query_triples.append(QueryTriple(raw=raw, typed=typed))

            if not query_triples:
                # No triples, output empty result
                output_record = {
                    "id": q_id,
                    "question": question,
                    "answer": answers,
                    "ranked_doc_ids": [],
                }
                outfile.write(json.dumps(output_record, ensure_ascii=False) + "\n")
                outfile.flush()
                pbar.update(1)
                continue

            # Find corresponding doc record by query id
            d_rec = doc_records_by_query.get(q_id)
            if not d_rec:
                output_record = {
                    "id": q_id,
                    "question": question,
                    "answer": answers,
                    "ranked_doc_ids": [],
                }
                outfile.write(json.dumps(output_record, ensure_ascii=False) + "\n")
                outfile.flush()
                pbar.update(1)
                continue

            # Build DocTriple list from the matched doc record
            # doc_id is the inner field (1, 2, 3, ...) not the query id
            doc_triples: List[DocTriple] = []
            for t in d_rec.get("triples", []):
                inner_doc_id = str(t.get("doc_id", 0))  # Convert to string for consistency
                raw = tuple(t.get("raw_triple", ["", "", ""]))
                typed = tuple(t.get("type_only_triple", t.get("raw_triple", ["", "", ""])))
                doc_triples.append(DocTriple(doc_id=inner_doc_id, raw=raw, typed=typed))

            if not doc_triples:
                output_record = {
                    "id": q_id,
                    "question": question,
                    "answer": answers,
                    "ranked_doc_ids": [],
                }
                outfile.write(json.dumps(output_record, ensure_ascii=False) + "\n")
                outfile.flush()
                pbar.update(1)
                continue

            # Get all unique doc_ids in original order (for fallback)
            all_doc_ids = []
            seen_doc_ids = set()
            for t in d_rec.get("triples", []):
                did = t.get("doc_id", 0)
                if did not in seen_doc_ids:
                    all_doc_ids.append(did)
                    seen_doc_ids.add(did)

            # Run matching
            ranked_docs, kept_doc_ids, _, _, doc_score_details = rank_docs_by_triple_matching(
                query_triples=query_triples,
                doc_triples=doc_triples,
                embedder=embedder,
                alpha_type=args.alpha_type,
                threshold=args.threshold,
            )

            # Get retrieval scores from original data file's ctxs
            data_rec = data_records_by_query.get(q_id, {})
            ctxs = data_rec.get("ctxs", [])
            retrieval_scores = {}
            for i, ctx in enumerate(ctxs, start=1):
                retrieval_scores[i] = ctx.get("score", 0.0)

            # If with_retrieval, multiply by retrieval score and re-rank
            if args.with_retrieval:
                final_ranked_docs = []
                for doc_id_str, mean_score in ranked_docs:
                    doc_id = int(doc_id_str)
                    ret_score = retrieval_scores.get(doc_id, 0.0)
                    combined_score = mean_score * ret_score
                    final_ranked_docs.append((doc_id, combined_score, mean_score, ret_score))
                    if doc_id_str in doc_score_details:
                        doc_score_details[doc_id_str]["mean_score"] = mean_score
                        doc_score_details[doc_id_str]["final_score"] = combined_score
                final_ranked_docs.sort(key=lambda x: x[1], reverse=True)
                ranked_doc_ids = [doc_id for doc_id, _, _, _ in final_ranked_docs] if final_ranked_docs else all_doc_ids
            else:
                if ranked_docs:
                    ranked_doc_ids = [int(doc_id) for doc_id, _ in ranked_docs]
                else:
                    ranked_doc_ids = all_doc_ids
            
            # Append missing doc_ids (1-10) in original order
            ranked_set = set(ranked_doc_ids)
            for doc_id in range(1, 11):
                if doc_id not in ranked_set:
                    ranked_doc_ids.append(doc_id)

            # Build score_detail list (retrieval_scores already built above)
            score_detail = []
            for rank, doc_id in enumerate(ranked_doc_ids, start=1):
                doc_id_str = str(doc_id)
                detail = doc_score_details.get(doc_id_str, {"final_score": 0.0})
                score_detail.append({
                    "doc_id": doc_id,
                    "final_rank": rank,
                    "retrieval_score": retrieval_scores.get(doc_id, 0.0),
                    **detail
                })

            output_record = {
                "id": q_id,
                "question": question,
                "answer": answers,
                "ranked_doc_ids": ranked_doc_ids,
                "score_detail": score_detail,
            }
            outfile.write(json.dumps(output_record, ensure_ascii=False) + "\n")
            outfile.flush()
            pbar.update(1)

    print(f"Done. Output: {args.output_file}")

