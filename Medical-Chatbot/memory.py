# memory.py
import re
import numpy as np
import faiss
from collections import defaultdict, deque
from typing import List
from sentence_transformers import SentenceTransformer
from google import genai  # must be configured in app.py and imported globally

_LLM = "gemini-2.5-flash-lite-preview-06-17" # Small model for NLP simple tasks
# Load embedding model
embedding_model = SentenceTransformer("/app/model_cache", device="cpu").half()

class MemoryManager:
    def __init__(self, max_users=1000, history_per_user=10):
        self.text_cache = defaultdict(lambda: deque(maxlen=history_per_user))
        self.chunk_index = defaultdict(lambda: faiss.IndexFlatL2(384))
        self.chunk_texts = defaultdict(list)
        self.user_queue = deque(maxlen=max_users)

    def add_exchange(self, user_id: str, query: str, response: str, lang: str = "EN"):
        if user_id not in self.text_cache:
            if len(self.user_queue) >= self.user_queue.maxlen:
                oldest = self.user_queue.popleft()
                self._drop_user(oldest)
            self.user_queue.append(user_id)

        self.text_cache[user_id].append((query.strip(), response.strip()))
        # Use Gemini to summarize and chunk smartly
        chunks = self.chunk_response(response, lang)
        # Encode chunk
        for chunk in chunks:
            vec = embedding_model.encode(chunk, convert_to_numpy=True)
            self.chunk_index[user_id].add(np.array([vec]))
            self.chunk_texts[user_id].append(chunk)

    def get_relevant_chunks(self, user_id: str, query: str, top_k: int = 2):
        if user_id not in self.chunk_index or self.chunk_index[user_id].ntotal == 0:
            return []
        # Encode user query
        vec = embedding_model.encode(query, convert_to_numpy=True)
        D, I = self.chunk_index[user_id].search(np.array([vec]), k=top_k)
        return [self.chunk_texts[user_id][i] for i in I[0] if i < len(self.chunk_texts[user_id])]

    def get_context(self, user_id: str, num_turns: int = 3):
        history = list(self.text_cache.get(user_id, []))[-num_turns:]
        return "\n".join(f"User: {q}\nBot: {r}" for q, r in history)

    def reset(self, user_id: str):
        self._drop_user(user_id)
        if user_id in self.user_queue:
            self.user_queue.remove(user_id)

    def _drop_user(self, user_id):
        self.text_cache.pop(user_id, None)
        self.chunk_index.pop(user_id, None)
        self.chunk_texts.pop(user_id, None)

    def chunk_response(self, response: str, lang: str) -> List[str]:
        """
        Use Gemini to translate (if needed), summarize, and chunk the response.
        Assumes Gemini API is configured via google.genai globally in app.py.
        """
        # Full instruction
        instructions = []
        # Only add translation if necessary
        if lang.upper() != "EN":
            instructions.append("- Translate the response to English.")
        instructions.append("- Break the translated (or original) text into semantically distinct parts, grouped by medical topic or symptom.")
        instructions.append("- For each part, generate a clear, concise summary. The summary may vary in length depending on the complexity of the topic — do not omit key clinical instructions.")
        instructions.append("- Separate each part using three dashes `---` on a new line.")
        # Grouped sub-instructions
        joined_instructions = "\n".join(instructions)
        # Prompting
        prompt = f"""
        You are a medical assistant helping organize and condense a clinical response.
        Below is the user-provided medical response written in `{lang}`:
        ------------------------
        {response}
        ------------------------
        Please perform the following tasks:
        {joined_instructions}
        Output only the structured summaries, separated by dashes.
        """
        try:
            client = genai.Client()
            result = client.models.generate_content(
                model=_LLM,
                contents=prompt,
                generation_config={"temperature": 0.4}
            )
            output = result.text.strip()
            return [chunk.strip() for chunk in output.split('---') if chunk.strip()]
        except Exception as e:
            print(f"❌ Gemini chunking failed: {e}")
            return [response.strip()]
