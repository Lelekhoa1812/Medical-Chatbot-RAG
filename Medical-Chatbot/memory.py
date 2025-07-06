# memory.py
import re, time, hashlib, asyncio
from collections import defaultdict, deque
from typing import List, Dict
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from google import genai  # must be configured in app.py and imported globally
import logging

_LLM_SMALL = "gemini-2.5-flash-lite-preview-06-17"
# Load embedding model
EMBED = SentenceTransformer("/app/model_cache", device="cpu").half()
logger = logging.getLogger("rag-agent")
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(name)s — %(levelname)s — %(message)s", force=True) # Change INFO to DEBUG for full-ctx JSON loader

class MemoryManager:
    def __init__(self, max_users=1000, history_per_user=10, max_chunks=30):
        self.text_cache   = defaultdict(lambda: deque(maxlen=history_per_user))
        self.chunk_index  = defaultdict(self._new_index)     # user_id -> faiss index
        self.chunk_meta   = defaultdict(list)                #   ''   -> list[{text,tag}]
        self.user_queue   = deque(maxlen=max_users)          # LRU of users
        self.max_chunks   = max_chunks                       # hard cap per user
        self.chunk_cache  = {}                               # hash(query+resp) -> [chunks]

    # ---------- Public API ----------
    def add_exchange(self, user_id: str, query: str, response: str, lang: str = "EN"):
        self._touch_user(user_id)
        self.text_cache[user_id].append((query.strip(), response.strip()))
        # Avoid re-chunking identical response
        cache_key = hashlib.md5((query + response).encode()).hexdigest()
        if cache_key in self.chunk_cache:
            chunks = self.chunk_cache[cache_key]
        else:
            chunks = self.chunk_response(response, lang)
            self.chunk_cache[cache_key] = chunks
        # Store chunks → faiss
        for chunk in chunks:
            vec = self._embed(chunk["text"])
            self.chunk_index[user_id].add(np.array([vec]))
            self.chunk_meta[user_id].append(chunk)
        # Trim to max_chunks to keep latency O(1)
        if len(self.chunk_meta[user_id]) > self.max_chunks:
            self._rebuild_index(user_id, keep_last=self.max_chunks)

    def get_relevant_chunks(self, user_id: str, query: str, top_k: int = 3, min_sim: float = 0.30) -> List[str]:
        """Return texts of chunks whose cosine similarity ≥ min_sim."""
        if self.chunk_index[user_id].ntotal == 0:
            return []
        # Encode chunk
        qvec   = self._embed(query)
        sims, idxs = self.chunk_index[user_id].search(np.array([qvec]), k=top_k)
        results = []
        for sim, idx in zip(sims[0], idxs[0]):
            if idx < len(self.chunk_meta[user_id]) and sim >= min_sim:
                results.append(self.chunk_meta[user_id][idx]["text"])
        return results

    def get_context(self, user_id: str, num_turns: int = 3) -> str:
        history = list(self.text_cache.get(user_id, []))[-num_turns:]
        return "\n".join(f"User: {q}\nBot: {r}" for q, r in history)

    def reset(self, user_id: str):
        self._drop_user(user_id)

    # ---------- Internal helpers ----------
    def _touch_user(self, user_id: str):
        if user_id not in self.text_cache and len(self.user_queue) >= self.user_queue.maxlen:
            self._drop_user(self.user_queue.popleft())
        if user_id in self.user_queue:
            self.user_queue.remove(user_id)
        self.user_queue.append(user_id)

    def _drop_user(self, user_id: str):
        self.text_cache.pop(user_id, None)
        self.chunk_index.pop(user_id, None)
        self.chunk_meta.pop(user_id, None)
        if user_id in self.user_queue:
            self.user_queue.remove(user_id)

    def _rebuild_index(self, user_id: str, keep_last: int):
        """Trim chunk list + rebuild FAISS index for user."""
        self.chunk_meta[user_id] = self.chunk_meta[user_id][-keep_last:]
        index = self._new_index()
        for chunk in self.chunk_meta[user_id]:
            vec = self._embed(chunk["text"])
            index.add(np.array([vec]))
        self.chunk_index[user_id] = index

    @staticmethod
    def _new_index():
        # Use cosine similarity (vectors must be L2-normalised)
        return faiss.IndexFlatIP(384)

    @staticmethod
    def _embed(text: str):
        vec = EMBED.encode(text, convert_to_numpy=True)
        # L2 normalise for cosine on IndexFlatIP
        return vec / (np.linalg.norm(vec) + 1e-9)

    def chunk_response(self, response: str, lang: str) -> List[Dict]:
        """
        Calls Gemini to:
          - Translate (if needed)
          - Chunk by context/topic
          - Summarise
        Returns: [{"tag": ..., "text": ...}, ...]
        """
        # Gemini instruction
        instructions = []
        if lang.upper() != "EN":
            instructions.append("- Translate the response to English.")
        instructions.append("- Break the translated (or original) text into semantically distinct parts, grouped by medical topic or symptom.")
        instructions.append("- For each part, generate a clear, concise summary. The summary may vary in length depending on the complexity of the topic — do not omit key clinical instructions.")
        instructions.append("- Separate each part using three dashes `---` on a new line.")
        # Gemini prompt
        prompt = f"""
        You are a medical assistant helping organize and condense a clinical response.
        Below is the user-provided medical response written in `{lang}`:
        ------------------------
        {response}
        ------------------------
        Please perform the following tasks:
        {chr(10).join(instructions)}

        Output only the structured summaries, separated by dashes.
        """
        try:
            client = genai.Client()
            result = client.models.generate_content(
                model=_LLM_SMALL,
                contents=prompt,
                generation_config={"temperature": 0.4}
            )
            output = result.text.strip()
            logger.info(f"📦 Gemini summarized chunk output: {output}")
            print(f"📦 Gemini summarized chunk output: {output}")
            return [
                {"tag": self._quick_extract_topic(chunk), "text": chunk.strip()}
                for chunk in output.split('---') if chunk.strip()
            ]
        except Exception as e:
            logger.warning(f"❌ Gemini chunking failed: {e}")
            return [{"tag": "general", "text": response.strip()}]
        
    @staticmethod
    def _quick_extract_topic(chunk: str) -> str:
        """Heuristically extract the topic from a chunk (title line or first 3 words)."""
        lines = chunk.strip().splitlines()
        for line in lines:
            if len(line.split()) <= 8 and line.strip().endswith(":"):
                return line.strip().rstrip(":")
        return " ".join(chunk.split()[:3]).rstrip(":.,")

