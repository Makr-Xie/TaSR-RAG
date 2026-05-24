#!/usr/bin/env python3
import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np

try:
    import faiss
except ImportError as e:
    raise ImportError("faiss not found. Install with: pip install faiss-cpu") from e

try:
    from openai import OpenAI
except ImportError as e:
    raise ImportError("openai not found. Install with: pip install openai") from e


TAXONOMY: Dict[str, List[str]] = {
  "PERSON": [
    "Scientist", "Engineer", "Academic", "Politician", "Businessperson",
    "Athlete", "Actor", "Musician", "Writer", "Journalist", "Inventor", "MilitaryPerson"
  ],
  "ORGANIZATION": [
    "Company", "University", "ResearchInstitute", "GovernmentAgency", "Nonprofit",
    "InternationalOrganization", "MilitaryUnit", "SportsTeam", "PoliticalParty",
    "MediaOutlet", "Hospital", "School"
  ],
  "LOCATION": [
    "Country", "StateOrProvince", "City", "Region", "Continent",
    "River", "Lake", "Mountain", "Island", "SeaOrOcean", "Desert", "Park"
  ],
  "FACILITY": [
    "Building", "Bridge", "Airport", "Station", "Port", "Museum",
    "Stadium", "Campus", "Laboratory", "PowerPlant"
  ],
  "EVENT": [
    "War", "Election", "Tournament", "Conference", "Festival",
    "Disaster", "Protest", "LaunchEvent", "MergerEvent", "Trial"
  ],
  "WORK": [
    "Book", "Film", "TVSeries", "Song", "Album",
    "VideoGame", "SoftwareProject", "ResearchPaper", "LawOrPolicy", "Dataset"
  ],
  "PRODUCT": [
    "CloudService", "Database", "ProgrammingLanguage", "HardwareDevice", "VehicleModel",
    "Drug", "Chemical", "ConsumerProduct", "ModelOrAlgorithm"
  ],
  "BIOENTITY": [
    "Animal", "Plant", "Bacteria", "Virus", "Disease", "ProteinOrGene"
  ],
  "TIME": [
    "Year", "Date", "TimePeriod"
  ],
  "QUANTITY": [
    "Count", "Money", "Percentage", "Measurement"
  ],
  "CONCEPT": [
    "Technology", "Method", "Theory", "FieldOfStudy", "RoleOrTitle"
  ],
  "OTHER": [
    "Other"
  ]
}


def l2_prompt_text(l1: str, l2: str) -> str:
    return f"L2 subtype under {l1}: {l2}"


def l1_prompt_text(l1: str) -> str:
    return f"L1 type: {l1}"


def embed_texts(client: OpenAI, model: str, texts: List[str], batch_size: int) -> np.ndarray:
    vecs: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        resp.data.sort(key=lambda x: x.index)
        vecs.extend([d.embedding for d in resp.data])
    arr = np.asarray(vecs, dtype=np.float32)
    faiss.normalize_L2(arr)
    return arr


def build_ip_index(vectors: np.ndarray) -> "faiss.Index":
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_url", type=str, default="http://localhost:1225/v1")
    ap.add_argument("--api_key", type=str, default="EMPTY")
    ap.add_argument("--embed_model", type=str, default="qwen3-8b-embedding")
    ap.add_argument("--out_dir", type=str, default="type_faiss")
    ap.add_argument("--batch_size", type=int, default=128)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    # ===== L1 index =====
    l1_labels = list(TAXONOMY.keys())
    l1_texts = [l1_prompt_text(x) for x in l1_labels]
    l1_vecs = embed_texts(client, args.embed_model, l1_texts, args.batch_size)
    l1_index = build_ip_index(l1_vecs)

    faiss.write_index(l1_index, os.path.join(args.out_dir, "l1.index"))
    with open(os.path.join(args.out_dir, "l1_meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            [{"label": lab, "level": "L1"} for lab in l1_labels],
            f, ensure_ascii=False, indent=2
        )

    # ===== per-L1 L2 indexes =====
    for l1 in l1_labels:
        l2_labels = TAXONOMY[l1]
        l2_texts = [l2_prompt_text(l1, l2) for l2 in l2_labels]
        l2_vecs = embed_texts(client, args.embed_model, l2_texts, args.batch_size)
        l2_index = build_ip_index(l2_vecs)

        faiss.write_index(l2_index, os.path.join(args.out_dir, f"l2_{l1}.index"))
        with open(os.path.join(args.out_dir, f"l2_{l1}_meta.json"), "w", encoding="utf-8") as f:
            json.dump(
                [{"label": lab, "level": "L2", "parent": l1} for lab in l2_labels],
                f, ensure_ascii=False, indent=2
            )

    with open(os.path.join(args.out_dir, "taxonomy.json"), "w", encoding="utf-8") as f:
        json.dump(TAXONOMY, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved FAISS indexes to: {args.out_dir}")


if __name__ == "__main__":
    main()
