import time
import asyncio
import httpx
from typing import List, Tuple, Dict, Optional, Union
import numpy as np
import faiss
from openai import OpenAI
import re
import os
import json

# -------------------------------------------------------------------------
# LLM Client Class (Async/Sync Hybrid)
# -------------------------------------------------------------------------
class VLLMClients:
    def __init__(
        self,
        embed_base_url: str,
        chat_base_url: str,
        api_key: str,
        embed_model: str,
        chat_model: str,
    ):
        self.embed_client = OpenAI(base_url=embed_base_url, api_key=api_key)
        self.chat_client_sync = OpenAI(base_url=chat_base_url, api_key=api_key) # Keep sync client if needed
        self.chat_base_url = chat_base_url # Needed for httpx
        self.embed_model = embed_model
        self.chat_model = chat_model

    def embed_batch(self, texts: List[str], batch_size: int = 256, max_workers: int = 64, max_retries: int = 5) -> np.ndarray:
        """
        Parallel embedding using ThreadPoolExecutor with retry logic.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def call_single_batch_with_retry(batch_idx: int, batch: List[str]) -> Tuple[int, List[List[float]]]:
            for attempt in range(max_retries):
                try:
                    resp = self.embed_client.embeddings.create(model=self.embed_model, input=batch)
                    resp.data.sort(key=lambda x: x.index)
                    return batch_idx, [d.embedding for d in resp.data]
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4, 8, 16 seconds
                        time.sleep(wait_time)
                    else:
                        raise e

        # Split into batches
        batches = [(i, texts[i:i + batch_size]) for i in range(0, len(texts), batch_size)]
        
        if len(batches) == 1:
            # Single batch, no need for threading overhead
            _, vecs = call_single_batch_with_retry(0, batches[0][1])
            arr = np.asarray(vecs, dtype=np.float32)
            faiss.normalize_L2(arr)
            return arr

        # Process batches in parallel
        results: Dict[int, List[List[float]]] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(call_single_batch_with_retry, idx, batch): idx for idx, batch in batches}
            for future in as_completed(futures):
                batch_idx, vecs = future.result()
                results[batch_idx] = vecs

        # Reassemble in order
        all_vecs: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            all_vecs.extend(results[i])

        arr = np.asarray(all_vecs, dtype=np.float32)
        faiss.normalize_L2(arr)  # cosine -> inner product
        return arr

    def chat_batch(self, prompts: List[str], batch_size: int = 256, max_concurrent: int = 128) -> List[str]:
        """
        True parallel chat completion using asyncio + httpx.
        max_concurrent: maximum number of concurrent requests
        """
        
        async def call_single(client: httpx.AsyncClient, idx: int, prompt: str, semaphore: asyncio.Semaphore, max_retries: int = 10) -> Tuple[int, str]:
            async with semaphore:
                payload = {
                    "model": self.chat_model,
                    "messages": [
                        {"role": "system", "content": "You must follow the user instructions exactly."},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.7,
                    "max_tokens": 768,
                }
                url = f"{self.chat_base_url}chat/completions" # Use base_url from init
                if not url.endswith("/"):
                     if not self.chat_base_url.endswith("/"):
                         url = f"{self.chat_base_url}/chat/completions"
                
                # Careful with double slash, httpx handles it usually but be safe
                # If chat_base_url is "http://localhost:1225/v1", then we want "http://localhost:1225/v1/chat/completions"
                
                for attempt in range(max_retries):
                    try:
                        resp = await client.post(url, json=payload, timeout=120.0)
                        resp.raise_for_status()
                        data = resp.json()
                        content = data["choices"][0]["message"]["content"]
                        return idx, content.strip()
                    except Exception as e:
                        if attempt < max_retries - 1:
                            wait_time = 2 ** attempt  # Exponential backoff
                            await asyncio.sleep(wait_time)
                        else:
                            raise e

        async def run_all():
            semaphore = asyncio.Semaphore(max_concurrent)
            async with httpx.AsyncClient() as client:
                tasks = [call_single(client, i, p, semaphore) for i, p in enumerate(prompts)]
                results = await asyncio.gather(*tasks, return_exceptions=True)
            return results

        # Run the async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            raw_results = loop.run_until_complete(run_all())
        finally:
            loop.close()

        # Process results
        output: Dict[int, str] = {}
        for r in raw_results:
            if isinstance(r, Exception):
                raise r
            idx, content = r
            output[idx] = content

        return [output[i] for i in range(len(prompts))]


# -------------------------------------------------------------------------
# Simple Sync Client (for non-batched use cases like decomposition)
# -------------------------------------------------------------------------
def simple_call_vllm(
    prompt: str, 
    client: OpenAI, 
    model: str, 
    temperature: float = 0.0, 
    max_tokens: int = 512,
    enable_thinking: bool = False
) -> str:
    """Wrapper for a simple sync call to vLLM via OpenAI client."""
    extra_body = {}
    if not enable_thinking:
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body=extra_body
    )
    return resp.choices[0].message.content.strip()


# ============================================================
# FAISS type retriever (L1 + per-L1 L2)
# ============================================================
class FaissTypeRetriever:
    def __init__(self, faiss_dir: str):
        self.faiss_dir = faiss_dir

        self.l1_index = faiss.read_index(os.path.join(faiss_dir, "l1.index"))
        with open(os.path.join(faiss_dir, "l1_meta.json"), "r", encoding="utf-8") as f:
            self.l1_meta = json.load(f)  # list of {"label":...}

        self.l2_index: Dict[str, faiss.Index] = {}
        self.l2_meta: Dict[str, List[Dict[str, Any]]] = {}

        for fn in os.listdir(faiss_dir):
            if fn.startswith("l2_") and fn.endswith(".index"):
                l1 = fn[len("l2_"):-len(".index")]
                self.l2_index[l1] = faiss.read_index(os.path.join(faiss_dir, fn))
                with open(os.path.join(faiss_dir, f"l2_{l1}_meta.json"), "r", encoding="utf-8") as f:
                    self.l2_meta[l1] = json.load(f)

    def topk_l1(self, vecs: np.ndarray, k: int) -> List[List[str]]:
        _, I = self.l1_index.search(vecs, k)
        return [[self.l1_meta[int(j)]["label"] for j in row] for row in I]

    def topk_l2(self, vecs: np.ndarray, l1_list: List[str], k: int) -> List[List[str]]:
        out: List[List[str]] = [None] * len(l1_list)
        groups: Dict[str, List[int]] = {}
        for i, l1 in enumerate(l1_list):
            groups.setdefault(l1, []).append(i)

        for l1, idxs in groups.items():
            index = self.l2_index.get(l1)
            meta = self.l2_meta.get(l1)
            if index is None or meta is None:
                for gi in idxs:
                    out[gi] = ["Other"]
                continue

            sub_vecs = vecs[idxs]
            _, I = index.search(sub_vecs, k)
            for local_row, global_i in enumerate(idxs):
                out[global_i] = [meta[int(j)]["label"] for j in I[local_row]]

        return out

