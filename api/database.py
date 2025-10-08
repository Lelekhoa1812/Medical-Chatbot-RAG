# api/database.py
import faiss
import numpy as np
import gridfs
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from api.config import mongo_uri, index_uri, MODEL_CACHE_DIR, EMBEDDING_MODEL_DEVICE
import logging

logger = logging.getLogger("medical-chatbot")

class DatabaseManager:
    def __init__(self):
        self.embedding_model = None
        self.index = None
        self.symptom_vectors = None
        self.symptom_docs = None
        
        # MongoDB connections
        self.client = None
        self.iclient = None
        self.symptom_client = None
        
        # Collections
        self.qa_collection = None
        self.index_collection = None
        self.symptom_col = None
        self.fs = None
        
    def initialize_embedding_model(self):
        """Initialize the SentenceTransformer model"""
        logger.info("[Embedder] 📥 Loading SentenceTransformer Model...")
        try:
            self.embedding_model = SentenceTransformer(MODEL_CACHE_DIR, device=EMBEDDING_MODEL_DEVICE)
            self.embedding_model = self.embedding_model.half()  # Reduce memory
            logger.info("✅ Model Loaded Successfully.")
        except Exception as e:
            logger.error(f"❌ Model Loading Failed: {e}")
            raise
    
    def initialize_mongodb(self):
        """Initialize MongoDB connections and collections"""
        # QA data
        self.client = MongoClient(mongo_uri)
        db = self.client["MedicalChatbotDB"]
        self.qa_collection = db["qa_data"]
        
        # FAISS Index data
        self.iclient = MongoClient(index_uri)
        idb = self.iclient["MedicalChatbotDB"]
        self.index_collection = idb["faiss_index_files"]
        
        # Symptom Diagnosis data
        self.symptom_client = MongoClient(mongo_uri) 
        self.symptom_col = self.symptom_client["MedicalChatbotDB"]["symptom_diagnosis"]
        
        # GridFS for FAISS index
        self.fs = gridfs.GridFS(idb, collection="faiss_index_files")
    
    def load_faiss_index(self):
        """Lazy load FAISS index from GridFS"""
        if self.index is None:
            logger.info("[KB] ⏳ Loading FAISS index from GridFS...")
            existing_file = self.fs.find_one({"filename": "faiss_index.bin"})
            if existing_file:
                stored_index_bytes = existing_file.read()
                index_bytes_np = np.frombuffer(stored_index_bytes, dtype='uint8')
                self.index = faiss.deserialize_index(index_bytes_np)
                logger.info("[KB] ✅ FAISS Index Loaded")
            else:
                logger.error("[KB] ❌ FAISS index not found in GridFS.")
        return self.index
    
    def load_symptom_vectors(self):
        """Lazy load symptom vectors for diagnosis"""
        if self.symptom_vectors is None:
            all_docs = list(self.symptom_col.find({}, {"embedding": 1, "answer": 1, "question": 1, "prognosis": 1}))
            self.symptom_docs = all_docs
            self.symptom_vectors = np.array([doc["embedding"] for doc in all_docs], dtype=np.float32)
    
    def get_embedding_model(self):
        """Get the embedding model"""
        if self.embedding_model is None:
            self.initialize_embedding_model()
        return self.embedding_model
    
    def get_qa_collection(self):
        """Get QA collection"""
        if self.qa_collection is None:
            self.initialize_mongodb()
        return self.qa_collection
    
    def get_symptom_collection(self):
        """Get symptom collection"""
        if self.symptom_col is None:
            self.initialize_mongodb()
        return self.symptom_col

# Global database manager instance
db_manager = DatabaseManager()
