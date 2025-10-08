# api/retrieval.py
import numpy as np
import logging
from api.database import db_manager

logger = logging.getLogger("medical-chatbot")

class RetrievalEngine:
    def __init__(self):
        self.db_manager = db_manager
    
    def retrieve_medical_info(self, query: str, k: int = 5, min_sim: float = 0.9) -> list:
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
