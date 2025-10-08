# api/retrieval.py
import os
import re
import time
import requests
import numpy as np
import logging
from typing import List, Dict
from .database import db_manager
from models import summarizer

logger = logging.getLogger("retrieval-bot")

class RetrievalEngine:
    def __init__(self):
        self.db_manager = db_manager
        # Lazy-init reranker to avoid NameError during module import ordering
        self._reranker = None

    def _get_reranker(self):
        """Initialize the NVIDIA reranker on first use."""
        if self._reranker is None:
            self._reranker = _NvidiaReranker()
        return self._reranker
    
    @staticmethod
    def _is_cpg_text(text: str) -> bool:
        """Heuristic to detect Clinical Practice Guideline (CPG) content."""
        if not text:
            return False
        keywords = [
            # common CPG indicators
            r"\bguideline(s)?\b", r"\bclinical practice\b", r"\brecommend(ation|ed|s)?\b",
            r"\bshould\b", r"\bmust\b", r"\bstrongly (recommend|suggest)\b",
            r"\bNICE\b", r"\bAHA\b", r"\bACC\b", r"\bWHO\b", r"\bUSPSTF\b", r"\bIDSA\b",
            r"\bclass (I|IIa|IIb|III)\b", r"\blevel (A|B|C)\b"
        ]
        text_lc = text.lower()
        return any(re.search(p, text_lc, flags=re.IGNORECASE) for p in keywords)
    
    @staticmethod
    def _extract_guideline_sentences(text: str) -> str:
        """Extract likely guideline sentences to reduce conversational/noisy content before summarization."""
        if not text:
            return ""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        keep_patterns = [
            r"\b(recommend|should|must|indicated|contraindicated|preferred|first-line|consider)\b",
            r"\b(class\s*(I|IIa|IIb|III)|level\s*(A|B|C))\b",
            r"\b(dose|mg|route|frequency)\b",
            r"\b(screen|treat|manage|evaluate|monitor)\b"
        ]
        kept = []
        for s in sentences:
            s_norm = s.strip()
            if not s_norm:
                continue
            if any(re.search(p, s_norm, flags=re.IGNORECASE) for p in keep_patterns):
                kept.append(s_norm)
        # Fallback: if filtering too aggressive, keep truncated original
        if not kept:
            return text[:1200]
        return " ".join(kept)[:2000]
    
    def retrieve_medical_info(self, query: str, k: int = 5, min_sim: float = 0.8) -> list:
        """
        Retrieve medical information from FAISS index
        Min similarity between query and kb is to be 80%
        """
        index = self.db_manager.load_faiss_index()
        if index is None:
            return [""]
        
        embedding_model = self.db_manager.get_embedding_model()
        qa_collection = self.db_manager.get_qa_collection()
        
        # Embed query
        query_vec = embedding_model.encode([query], convert_to_numpy=True)
        D, I = index.search(query_vec, k=k)
        
        # Filter by cosine threshold
        results = []
        kept = []
        kept_vecs = []
        
        # Smart dedup on cosine threshold between similar candidates
        for score, idx in zip(D[0], I[0]):
            if score < min_sim:
                continue
            
            # List sim docs
            doc = qa_collection.find_one({"i": int(idx)})
            if not doc:
                continue
            
            # Only compare answers
            answer = doc.get("Doctor", "").strip()
            if not answer:
                continue
            
            # Check semantic redundancy among previously kept results
            new_vec = embedding_model.encode([answer], convert_to_numpy=True)[0]
            is_similar = False
            
            for i, vec in enumerate(kept_vecs):
                sim = np.dot(vec, new_vec) / (np.linalg.norm(vec) * np.linalg.norm(new_vec) + 1e-9)
                if sim >= 0.9:  # High semantic similarity
                    is_similar = True
                    # Keep only better match to original query
                    cur_sim_to_query = np.dot(vec, query_vec[0]) / (np.linalg.norm(vec) * np.linalg.norm(query_vec[0]) + 1e-9)
                    new_sim_to_query = np.dot(new_vec, query_vec[0]) / (np.linalg.norm(new_vec) * np.linalg.norm(query_vec[0]) + 1e-9)
                    if new_sim_to_query > cur_sim_to_query:
                        kept[i] = answer
                        kept_vecs[i] = new_vec
                    break
            
            # Non-similar candidates
            if not is_similar:
                kept.append(answer)
                kept_vecs.append(new_vec)
        
        # If any CPG-like content is present, rerank with NVIDIA NIM reranker and summarize to key guidelines
        try:
            cpg_candidates = [t for t in kept if self._is_cpg_text(t)]
            if cpg_candidates:
                logger.info("[Retrieval] CPG content detected; invoking NVIDIA reranker")
                reranked = self._get_reranker().rerank(query, cpg_candidates)
                # Keep only valid high-scoring items
                filtered: List[Dict] = [r for r in reranked if r.get("score", 0) >= 0.3 and r.get("text")]
                # Limit to top 3 for prompt efficiency
                top_items = filtered[:3]
                if top_items:
                    summarized: List[str] = []
                    for item in top_items:
                        guideline_text = self._extract_guideline_sentences(item["text"])
                        # Summarize to key clinical guidelines only (no conversational content)
                        concise = summarizer.summarize_text(guideline_text, max_length=300)
                        if concise:
                            summarized.append(concise)
                    # If summarization produced results, replace kept with these
                    if summarized:
                        kept = summarized
        except Exception as e:
            logger.warning(f"[Retrieval] CPG rerank/summarize step skipped due to error: {e}")

        return kept if kept else [""]
    
    def retrieve_diagnosis_from_symptoms(self, symptom_text: str, top_k: int = 5, min_sim: float = 0.5) -> list:
        """
        Retrieve diagnosis information from symptom vectors
        """
        self.db_manager.load_symptom_vectors()
        embedding_model = self.db_manager.get_embedding_model()
        
        # Embed input
        qvec = embedding_model.encode(symptom_text, convert_to_numpy=True)
        qvec = qvec / (np.linalg.norm(qvec) + 1e-9)
        
        # Similarity compute
        sims = self.db_manager.symptom_vectors @ qvec  # cosine
        sorted_idx = np.argsort(sims)[-top_k:][::-1]
        seen_diag = set()
        final = []  # Dedup
        
        for i in sorted_idx:
            sim = sims[i]
            if sim < min_sim:
                continue
            label = self.db_manager.symptom_docs[i]["prognosis"]
            if label not in seen_diag:
                final.append(self.db_manager.symptom_docs[i]["answer"])
                seen_diag.add(label)
        
        return final

# Global retrieval engine instance
retrieval_engine = RetrievalEngine()


class _NvidiaReranker:
    """Simple client for NVIDIA NIM reranking: nvidia/rerank-qa-mistral-4b"""
    def __init__(self):
        self.api_key = os.getenv("NVIDIA_URI")
        self.model = "nvidia/rerank-qa-mistral-4b"
        # NIM rerank endpoint (subject to environment); keep configurable
        self.base_url = os.getenv("NVIDIA_RERANK_ENDPOINT", "https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking")
        self.timeout_s = 30

    def rerank(self, query: str, documents: List[str]) -> List[Dict]:
        if not self.api_key:
            raise ValueError("NVIDIA_URI not set for reranker")
        if not documents:
            return []
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "query": query,
            "documents": [{"text": d} for d in documents],
        }
        try:
            resp = requests.post(self.base_url, headers=headers, json=payload, timeout=self.timeout_s)
            resp.raise_for_status()
            data = resp.json()
            # Expecting a list with scores and indices or texts
            results = []
            entries = data.get("results") or data.get("data") or []
            if isinstance(entries, list) and entries:
                for entry in entries:
                    # Common patterns: {index, score} or {text, score}
                    idx = entry.get("index")
                    text = entry.get("text") if entry.get("text") else (documents[idx] if idx is not None and idx < len(documents) else None)
                    score = entry.get("score", 0)
                    if text:
                        results.append({"text": text, "score": float(score)})
            else:
                # Fallback: if API returns scores aligned to input order
                scores = data.get("scores")
                if isinstance(scores, list) and len(scores) == len(documents):
                    for t, s in zip(documents, scores):
                        results.append({"text": t, "score": float(s)})
            # Sort by score desc
            results.sort(key=lambda x: x.get("score", 0), reverse=True)
            return results
        except Exception as e:
            logger.warning(f"[Reranker] Failed calling NVIDIA reranker: {e}")
            # On failure, return original order with neutral scores
            return [{"text": d, "score": 0.0} for d in documents]
