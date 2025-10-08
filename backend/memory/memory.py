# memory_updated.py
import re, time, hashlib, asyncio, os
from collections import defaultdict, deque
from typing import List, Dict
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from google import genai  # must be configured in app.py and imported globally
import logging
from models.summarizer import get_summarizer

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
        Use NVIDIA Llama to create a summarization of relevant context from both recent history and RAG chunks.
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
        
        # Prepare context for summarization
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
        
        # Combine all context
        full_context = "\n\n".join(context_parts)
        
        # Use summarizer to create concise summary
        try:
            summary = summarizer.summarize_text(full_context, max_length=300)
            logger.info(f"[Contextual] Generated summary using NVIDIA Llama: {len(summary)} characters")
            return summary
        except Exception as e:
            logger.error(f"[Contextual] Summarization failed: {e}")
            return full_context[:500] + "..." if len(full_context) > 500 else full_context

    def chunk_response(self, response: str, lang: str, question: str = "") -> List[Dict]:
        """
        Use NVIDIA Llama to chunk and summarize response by medical topics.
        Returns: [{"tag": ..., "text": ...}, ...]
        """
        if not response: 
            return []
        
        try:
            # Use summarizer to chunk and summarize
            chunks = summarizer.chunk_response(response, max_chunk_size=500)
            
            # Convert to the expected format
            result_chunks = []
            for i, chunk in enumerate(chunks):
                # Extract topic from chunk (first sentence or key medical terms)
                topic = self._extract_topic_from_chunk(chunk)
                
                result_chunks.append({
                    "tag": topic,
                    "text": chunk
                })
            
            logger.info(f"[Memory] 📦 NVIDIA Llama summarized {len(result_chunks)} chunks")
            return result_chunks
            
        except Exception as e:
            logger.error(f"[Memory] NVIDIA Llama chunking failed: {e}")
            # Fallback to simple chunking
            return self._fallback_chunking(response)

    def _extract_topic_from_chunk(self, chunk: str) -> str:
        """Extract a concise topic from a chunk"""
        # Look for medical terms or first sentence
        sentences = chunk.split('.')
        if sentences:
            first_sentence = sentences[0].strip()
            if len(first_sentence) > 50:
                first_sentence = first_sentence[:50] + "..."
            return first_sentence
        return "Medical Information"

    def _fallback_chunking(self, response: str) -> List[Dict]:
        """Fallback chunking when NVIDIA Llama fails"""
        # Simple sentence-based chunking
        sentences = re.split(r'[.!?]+', response)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            if len(current_chunk) + len(sentence) > 300:
                if current_chunk:
                    chunks.append({
                        "tag": "Medical Information",
                        "text": current_chunk.strip()
                    })
                current_chunk = sentence
            else:
                current_chunk += sentence + ". "
        
        if current_chunk:
            chunks.append({
                "tag": "Medical Information", 
                "text": current_chunk.strip()
            })
        
        return chunks

    # ---------- Private Methods ----------
    def _touch_user(self, user_id: str):
        """Update LRU queue"""
        if user_id in self.user_queue:
            self.user_queue.remove(user_id)
        self.user_queue.append(user_id)

    def _new_index(self):
        """Create new FAISS index"""
        return faiss.IndexFlatIP(384)  # 384-dim embeddings

    def _upsert_stm(self, user_id: str, chunk: Dict, lang: str):
        """Update short-term memory with merging/deduplication"""
        topic = chunk["tag"]
        text = chunk["text"]
        
        # Check for similar topics in STM
        for entry in self.stm_summaries[user_id]:
            if self._topics_similar(topic, entry["topic"]):
                # Merge with existing entry
                entry["text"] = summarizer.summarize_text(
                    f"{entry['text']}\n{text}", 
                    max_length=200
                )
                entry["timestamp"] = time.time()
                return
        
        # Add new entry
        self.stm_summaries[user_id].append({
            "topic": topic,
            "text": text,
            "vec": self._embed(f"{topic} {text}"),
            "timestamp": time.time(),
            "used": 0
        })

    def _upsert_ltm(self, user_id: str, chunks: List[Dict], lang: str):
        """Update long-term memory with merging/deduplication"""
        for chunk in chunks:
            # Check for similar chunks in LTM
            similar_idx = self._find_similar_chunk(user_id, chunk["text"])
            
            if similar_idx is not None:
                # Merge with existing chunk
                existing = self.chunk_meta[user_id][similar_idx]
                merged_text = summarizer.summarize_text(
                    f"{existing['text']}\n{chunk['text']}", 
                    max_length=300
                )
                existing["text"] = merged_text
                existing["timestamp"] = time.time()
            else:
                # Add new chunk
                if len(self.chunk_meta[user_id]) >= self.max_chunks:
                    # Remove oldest chunk
                    self._remove_oldest_chunk(user_id)
                
                vec = self._embed(chunk["text"])
                self.chunk_index[user_id].add(np.array([vec]))
                self.chunk_meta[user_id].append({
                    "text": chunk["text"],
                    "tag": chunk["tag"],
                    "vec": vec,
                    "timestamp": time.time(),
                    "used": 0
                })

    def _topics_similar(self, topic1: str, topic2: str) -> bool:
        """Check if two topics are similar"""
        # Simple similarity check based on common words
        words1 = set(topic1.lower().split())
        words2 = set(topic2.lower().split())
        intersection = words1.intersection(words2)
        return len(intersection) >= 2

    def _find_similar_chunk(self, user_id: str, text: str) -> int:
        """Find similar chunk in LTM"""
        if not self.chunk_meta[user_id]:
            return None
        
        text_vec = self._embed(text)
        sims, idxs = self.chunk_index[user_id].search(np.array([text_vec]), k=3)
        
        for sim, idx in zip(sims[0], idxs[0]):
            if sim > 0.8:  # High similarity threshold
                return int(idx)
        return None

    def _remove_oldest_chunk(self, user_id: str):
        """Remove the oldest chunk from LTM"""
        if not self.chunk_meta[user_id]:
            return
        
        # Find oldest chunk
        oldest_idx = min(range(len(self.chunk_meta[user_id])), 
                        key=lambda i: self.chunk_meta[user_id][i]["timestamp"])
        
        # Remove from index and metadata
        self.chunk_meta[user_id].pop(oldest_idx)
        # Note: FAISS doesn't support direct removal, so we rebuild the index
        self._rebuild_index(user_id)

    def _rebuild_index(self, user_id: str):
        """Rebuild FAISS index after removal"""
        if not self.chunk_meta[user_id]:
            self.chunk_index[user_id] = self._new_index()
            return
        
        vectors = [chunk["vec"] for chunk in self.chunk_meta[user_id]]
        self.chunk_index[user_id] = self._new_index()
        self.chunk_index[user_id].add(np.array(vectors))

    @staticmethod
    def _embed(text: str):
        vec = EMBED.encode(text, convert_to_numpy=True)
        # L2 normalise for cosine on IndexFlatIP
        return vec / (np.linalg.norm(vec) + 1e-9)
