# memory.py
import re, time, hashlib, asyncio, os
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

api_key = os.getenv("FlashAPI")
client = genai.Client(api_key=api_key)

class MemoryManager:
    def __init__(self, max_users=1000, history_per_user=20, max_chunks=60):
        # STM: recent conversation summaries (topic + summary), up to 5 entries
        self.stm_summaries = defaultdict(lambda: deque(maxlen=history_per_user))  # deque of {topic,text,vec,timestamp,used}
        # Legacy raw cache (kept for compatibility if needed)
        self.text_cache   = defaultdict(lambda: deque(maxlen=history_per_user))
        # LTM: semantic chunk store (approx 3 chunks x 20 rounds)
        self.chunk_index  = defaultdict(self._new_index)     # user_id -> faiss index
        self.chunk_meta   = defaultdict(list)                #  ''  -> list[{text,tag,vec,timestamp,used}]
        self.user_queue   = deque(maxlen=max_users)          # LRU of users
        self.max_chunks   = max_chunks                       # hard cap per user
        self.chunk_cache  = {}                               # hash(query+resp) -> [chunks]

    # ---------- Public API ----------
    def add_exchange(self, user_id: str, query: str, response: str, lang: str = "EN"):
        self._touch_user(user_id)
        # Keep raw record (optional)
        self.text_cache[user_id].append(((query or "").strip(), (response or "").strip()))
        if not response: return []
        # Avoid re-chunking identical response
        cache_key = hashlib.md5((query + response).encode()).hexdigest()
        if cache_key in self.chunk_cache:
            chunks = self.chunk_cache[cache_key]
        else:
            chunks = self.chunk_response(response, lang, question=query)
            self.chunk_cache[cache_key] = chunks
        # Update STM with merging/deduplication
        for chunk in chunks:
            self._upsert_stm(user_id, chunk, lang)
        # Update LTM with merging/deduplication
        self._upsert_ltm(user_id, chunks, lang)
        return chunks

    def get_relevant_chunks(self, user_id: str, query: str, top_k: int = 3, min_sim: float = 0.30) -> List[str]:
        """Return texts of chunks whose cosine similarity ≥ min_sim."""
        if self.chunk_index[user_id].ntotal == 0:
            return []
        # Encode chunk
        qvec   = self._embed(query)
        sims, idxs = self.chunk_index[user_id].search(np.array([qvec]), k=top_k)
        results = []
        # Append related result with smart-decay to optimize storage and prioritize most-recent chat
        for sim, idx in zip(sims[0], idxs[0]):
            if idx < len(self.chunk_meta[user_id]) and sim >= min_sim:
                chunk = self.chunk_meta[user_id][idx]
                chunk["used"] += 1  # increment usage
                # Decay function
                age_sec = time.time() - chunk["timestamp"]
                decay = 1.0 / (1.0 + age_sec / 300)  # 5-min half-life
                score = sim * decay * (1 + 0.1 * chunk["used"])
                # Append chunk with score
                results.append((score, chunk))
        # Sort result on best scored
        results.sort(key=lambda x: x[0], reverse=True)
        # logger.info(f"[Memory] RAG Retrieved Topic: {results}") # Inspect vector data
        return [f"### Topic: {c['tag']}\n{c['text']}" for _, c in results]

    def get_recent_chat_history(self, user_id: str, num_turns: int = 5) -> List[Dict]:
        """
        Get the most recent short-term memory summaries.
        Returns: a list of entries containing only the summarized bot context.
        """
        if user_id not in self.stm_summaries:
            return []
        recent = list(self.stm_summaries[user_id])[-num_turns:]
        formatted = []
        for entry in recent:
            formatted.append({
                "user": "",
                "bot": f"Topic: {entry['topic']}\n{entry['text']}",
                "timestamp": entry.get("timestamp", time.time())
            })
        return formatted

    def get_context(self, user_id: str, num_turns: int = 5) -> str:
        # Prefer STM summaries
        history = self.get_recent_chat_history(user_id, num_turns=num_turns)
        return "\n".join(h["bot"] for h in history)

    def get_contextual_chunks(self, user_id: str, current_query: str, lang: str = "EN") -> str:
        """
        Use Gemini Flash Lite to create a summarization of relevant context from both recent history and RAG chunks.
        This ensures conversational continuity while providing a concise summary for the main LLM.
        """
        # Get both types of context
        recent_history = self.get_recent_chat_history(user_id, num_turns=5)
        rag_chunks = self.get_relevant_chunks(user_id, current_query, top_k=3)
        
        logger.info(f"[Contextual] Retrieved {len(recent_history)} recent history items")
        logger.info(f"[Contextual] Retrieved {len(rag_chunks)} RAG chunks")
        
        # Return empty string if no context is found
        if not recent_history and not rag_chunks:
            logger.info(f"[Contextual] No context found, returning empty string")
            return ""
        # Prepare context for Gemini to summarize
        context_parts = []
        # Add recent chat history
        if recent_history:
            history_text = "\n".join([
                f"User: {item['user']}\nBot: {item['bot']}"
                for item in recent_history
            ])
            context_parts.append(f"Recent conversation history:\n{history_text}")
        # Add RAG chunks
        if rag_chunks:
            rag_text = "\n".join(rag_chunks)
            context_parts.append(f"Semantically relevant historical medical information:\n{rag_text}")
        
        # Build summarization prompt
        summarization_prompt = f"""
        You are a medical assistant creating a concise summary of conversation context for continuity.
        
        Current user query: "{current_query}"
        
        Available context information:
        {chr(10).join(context_parts)}
        
        Task: Create a brief, coherent summary that captures the key points from the conversation history and relevant medical information that are important for understanding the current query.
        
        Guidelines:
        1. Focus on medical symptoms, diagnoses, treatments, or recommendations mentioned
        2. Include any patient concerns or questions that are still relevant
        3. Highlight any follow-up needs or pending clarifications
        4. Keep the summary concise but comprehensive enough for context
        5. Maintain conversational flow and continuity
        
        Output: Provide a single, well-structured summary paragraph that can be used as context for the main LLM to provide a coherent response.
        If no relevant context exists, return "No relevant context found."
        
        Language context: {lang}
        """
        
        logger.debug(f"[Contextual] Full prompt: {summarization_prompt}")
        # Loop through the prompt and log the length of each part
        try:
            # Use Gemini Flash Lite for summarization
            client = genai.Client(api_key=os.getenv("FlashAPI"))
            result = client.models.generate_content(
                model=_LLM_SMALL,
                contents=summarization_prompt
            )
            summary = result.text.strip()
            if "No relevant context found" in summary:
                logger.info(f"[Contextual] Gemini indicated no relevant context found")
                return ""
            
            logger.info(f"[Contextual] Gemini created summary: {summary[:100]}...")
            return summary
            
        except Exception as e:
            logger.warning(f"[Contextual] Gemini summarization failed: {e}")
            logger.info(f"[Contextual] Using fallback summarization method")
            # Fallback: create a simple summary
            fallback_summary = []
            # Fallback: add recent history
            if recent_history:
                recent_summary = f"Recent conversation: User asked about {recent_history[-1]['user'][:50]}... and received a response about {recent_history[-1]['bot'][:50]}..."
                fallback_summary.append(recent_summary)
                logger.info(f"[Contextual] Fallback: Added recent history summary")
            # Fallback: add RAG chunks
            if rag_chunks:
                rag_summary = f"Relevant medical information: {len(rag_chunks)} chunks found covering various medical topics."
                fallback_summary.append(rag_summary)
                logger.info(f"[Contextual] Fallback: Added RAG chunks summary")
            final_fallback = " ".join(fallback_summary) if fallback_summary else ""
            return final_fallback

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
        # Store each chunk's vector once and reuse it.
        for chunk in self.chunk_meta[user_id]:
            index.add(np.array([chunk["vec"]]))
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

    def chunk_response(self, response: str, lang: str, question: str = "") -> List[Dict]:
        """
        Calls Gemini to:
          - Translate (if needed)
          - Chunk by context/topic (exclude disclaimer section)
          - Summarise
        Returns: [{"tag": ..., "text": ...}, ...]
        """
        if not response: return []
        # Gemini instruction
        instructions = []
        # if lang.upper() != "EN":
        #     instructions.append("- Translate the response to English.")
        instructions.append("- Break the translated (or original) text into semantically distinct parts, grouped by medical topic, symptom, assessment, plan, or instruction (exclude disclaimer section).")
        instructions.append("- For each part, generate a clear, concise summary. The summary may vary in length depending on the complexity of the topic — do not omit key clinical instructions and exact medication names/doses if present.")
        instructions.append("- At the start of each part, write `Topic: <concise but specific sentence (10-20 words) capturing patient context, condition, and action>`.")
        instructions.append("- Separate each part using three dashes `---` on a new line.")
        # if lang.upper() != "EN":
        #     instructions.append(f"Below is the user-provided medical response written in `{lang}`")
        # Gemini prompt
        prompt = f"""
        You are a medical assistant helping organize and condense a clinical response.
        If helpful, use the user's latest question for context to craft specific topics.
        User's latest question (context): {question}
        ------------------------
        {response}
        ------------------------
        Please perform the following tasks:
        {chr(10).join(instructions)}

        Output only the structured summaries, separated by dashes.
        """
        retries = 0
        while retries < 5:
            try:
                client = genai.Client(api_key=os.getenv("FlashAPI"))
                result = client.models.generate_content(
                    model=_LLM_SMALL,
                    contents=prompt
                    # ,generation_config={"temperature": 0.4} # Skip temp configs for gem-flash
                )
                output = result.text.strip()
                logger.info(f"[Memory] 📦 Gemini summarized chunk output: {output}")
                return [
                    {"tag": self._quick_extract_topic(chunk), "text": chunk.strip()}
                    for chunk in output.split('---') if chunk.strip()
                ]
            except Exception as e:
                logger.warning(f"[Memory] ❌ Gemini chunking failed: {e}")
                retries += 1
                time.sleep(0.5)
        return [{"tag": "general", "text": response.strip()}]  # fallback
        
    @staticmethod
    def _quick_extract_topic(chunk: str) -> str:
        """Heuristically extract the topic from a chunk (title line or first 3 words)."""
        # Expecting 'Topic: <something>'
        match = re.search(r'^Topic:\s*(.+)', chunk, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).strip()
        lines = chunk.strip().splitlines()
        for line in lines:
            if len(line.split()) <= 8 and line.strip().endswith(":"):
                return line.strip().rstrip(":")
        return " ".join(chunk.split()[:3]).rstrip(":.,")

    # ---------- New merging/dedup logic ----------
    def _upsert_stm(self, user_id: str, chunk: Dict, lang: str):
        """Insert or merge a summarized chunk into STM with semantic dedup/merge.
        Identical: replace the older with new. Partially similar: merge extra details from older into newer.
        """
        topic = self._enrich_topic(chunk.get("tag", ""), chunk.get("text", ""))
        text  = chunk.get("text", "").strip()
        vec   = self._embed(text)
        now   = time.time()
        entry = {"topic": topic, "text": text, "vec": vec, "timestamp": now, "used": 0}
        stm = self.stm_summaries[user_id]
        if not stm:
            stm.append(entry)
            return
        # find best match
        best_idx = -1
        best_sim = -1.0
        for i, e in enumerate(stm):
            sim = float(np.dot(vec, e["vec"]))
            if sim > best_sim:
                best_sim = sim
                best_idx = i
        if best_sim >= 0.92:  # nearly identical
            # replace older with current
            stm.rotate(-best_idx)
            stm.popleft()
            stm.rotate(best_idx)
            stm.append(entry)
        elif best_sim >= 0.75:  # partially similar → merge
            base = stm[best_idx]
            merged_text = self._merge_texts(new_text=text, old_text=base["text"])  # add bits from old not in new
            merged_topic = base["topic"] if len(base["topic"]) > len(topic) else topic
            merged_vec = self._embed(merged_text)
            merged_entry = {"topic": merged_topic, "text": merged_text, "vec": merged_vec, "timestamp": now, "used": base.get("used", 0)}
            stm.rotate(-best_idx)
            stm.popleft()
            stm.rotate(best_idx)
            stm.append(merged_entry)
        else:
            stm.append(entry)

    def _upsert_ltm(self, user_id: str, chunks: List[Dict], lang: str):
        """Insert or merge chunks into LTM with semantic dedup/merge, then rebuild index.
        Keeps only the most recent self.max_chunks entries.
        """
        current_list = self.chunk_meta[user_id]
        for chunk in chunks:
            text = chunk.get("text", "").strip()
            if not text:
                continue
            vec = self._embed(text)
            topic = self._enrich_topic(chunk.get("tag", ""), text)
            now = time.time()
            new_entry = {"tag": topic, "text": text, "vec": vec, "timestamp": now, "used": 0}
            if not current_list:
                current_list.append(new_entry)
                continue
            # find best similar entry
            best_idx = -1
            best_sim = -1.0
            for i, e in enumerate(current_list):
                sim = float(np.dot(vec, e["vec"]))
                if sim > best_sim:
                    best_sim = sim
                    best_idx = i
            if best_sim >= 0.92:
                # replace older with new
                current_list[best_idx] = new_entry
            elif best_sim >= 0.75:
                # merge details
                base = current_list[best_idx]
                merged_text = self._merge_texts(new_text=text, old_text=base["text"])  # add unique sentences from old
                merged_topic = base["tag"] if len(base["tag"]) > len(topic) else topic
                merged_vec = self._embed(merged_text)
                current_list[best_idx] = {"tag": merged_topic, "text": merged_text, "vec": merged_vec, "timestamp": now, "used": base.get("used", 0)}
            else:
                current_list.append(new_entry)
        # Trim and rebuild index
        if len(current_list) > self.max_chunks:
            current_list[:] = current_list[-self.max_chunks:]
        self._rebuild_index(user_id, keep_last=self.max_chunks)

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        # naive sentence splitter by ., !, ?
        parts = re.split(r"(?<=[\.!?])\s+", text.strip())
        return [p.strip() for p in parts if p.strip()]

    def _merge_texts(self, new_text: str, old_text: str) -> str:
        """Append sentences from old_text that are not already contained in new_text (by fuzzy match)."""
        new_sents = self._split_sentences(new_text)
        old_sents = self._split_sentences(old_text)
        new_set = set(s.lower() for s in new_sents)
        merged = list(new_sents)
        for s in old_sents:
            s_norm = s.lower()
            # consider present if significant overlap with any existing sentence
            if s_norm in new_set:
                continue
            # simple containment check
            if any(self._overlap_ratio(s_norm, t.lower()) > 0.8 for t in merged):
                continue
            merged.append(s)
        return " ".join(merged)

    @staticmethod
    def _overlap_ratio(a: str, b: str) -> float:
        """Compute token overlap ratio between two sentences."""
        ta = set(re.findall(r"\w+", a))
        tb = set(re.findall(r"\w+", b))
        if not ta or not tb:
            return 0.0
        inter = len(ta & tb)
        union = len(ta | tb)
        return inter / union

    @staticmethod
    def _enrich_topic(topic: str, text: str) -> str:
        """Make topic more descriptive if it's too short by using the first sentence of the text.
        Does not call LLM to keep latency low.
        """
        topic = (topic or "").strip()
        if len(topic.split()) < 5 or len(topic) < 20:
            sents = re.split(r"(?<=[\.!?])\s+", text.strip())
            if sents:
                first = sents[0]
                # cap to ~16 words
                words = first.split()
                if len(words) > 16:
                    first = " ".join(words[:16])
                # ensure capitalized
                return first.strip().rstrip(':')
        return topic


