# app.py
import os
import faiss
import numpy as np
import time
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pymongo import MongoClient
from google import genai
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from memory import MemoryManager
from translation import translate_query
from vlm import process_medical_image

# ✅ Enable Logging for Debugging
import logging
# —————— Silence Noisy Loggers ——————
for name in [
    "uvicorn.error", "uvicorn.access",
    "fastapi", "starlette",
    "pymongo", "gridfs",
    "sentence_transformers", "faiss",
    "google", "google.auth",
]:
    logging.getLogger(name).setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(name)s — %(levelname)s — %(message)s", force=True) # Change INFO to DEBUG for full-ctx JSON loader
logger = logging.getLogger("medical-chatbot")
logger.setLevel(logging.DEBUG)

# Debug Start
logger.info("🚀 Starting Medical Chatbot API...")

# ✅ Environment Variables
mongo_uri = os.getenv("MONGO_URI")
index_uri = os.getenv("INDEX_URI")
gemini_flash_api_key = os.getenv("FlashAPI")
# Validate environment endpoint
if not all([gemini_flash_api_key, mongo_uri, index_uri]):
    raise ValueError("❌ Missing API keys! Set them in Hugging Face Secrets.")
# logger.info(f"🔎 MongoDB URI: {mongo_uri}")
# logger.info(f"🔎 FAISS Index URI: {index_uri}")

# ✅ Monitor Resources Before Startup
import psutil
def check_system_resources():
    memory = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=1)
    disk = psutil.disk_usage("/")
    # Defines log info messages
    logger.info(f"[System] 🔍 System Resources - RAM: {memory.percent}%, CPU: {cpu}%, Disk: {disk.percent}%")
    if memory.percent > 85:
        logger.warning("⚠️ High RAM usage detected!")
    if cpu > 90:
        logger.warning("⚠️ High CPU usage detected!")
    if disk.percent > 90:
        logger.warning("⚠️ High Disk usage detected!")
check_system_resources()

# ✅ Reduce Memory usage with optimizers
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ✅ Initialize FastAPI app
app = FastAPI(title="Medical Chatbot API")
memory = MemoryManager()

from fastapi.middleware.cors import CORSMiddleware # Bypassing CORS origin
# Define the origins
origins = [
    "http://localhost:5173",                    # Vite dev server
    "http://localhost:3000",                    # Another vercel local dev
    "https://medical-chatbot-henna.vercel.app", # ✅ Vercel frontend production URL
    
]
# Add the CORS middleware:
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,   # or ["*"] to allow all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Use Lazy Loading for FAISS Index
index = None  # Delay FAISS Index loading until first query

# ✅ Load SentenceTransformer Model (Quantized/Halved)
logger.info("[Embedder] 📥 Loading SentenceTransformer Model...")
MODEL_CACHE_DIR = "/app/model_cache"
try:
    embedding_model = SentenceTransformer(MODEL_CACHE_DIR, device="cpu")
    embedding_model = embedding_model.half()  # Reduce memory
    logger.info("✅ Model Loaded Successfully.")
except Exception as e:
    logger.error(f"❌ Model Loading Failed: {e}")
    exit(1)

# Cache in-memory vectors (optional — useful for <10k rows)
SYMPTOM_VECTORS = None
SYMPTOM_DOCS = None

# ✅ Setup MongoDB Connection
# QA data
client = MongoClient(mongo_uri)
db = client["MedicalChatbotDB"]
qa_collection = db["qa_data"]
# FAISS Index data
iclient = MongoClient(index_uri)
idb = iclient["MedicalChatbotDB"]
index_collection = idb["faiss_index_files"]
# Symptom Diagnosis data
symptom_client = MongoClient(mongo_uri) 
symptom_col = symptom_client["MedicalChatbotDB"]["symptom_diagnosis"]

# ✅ Load FAISS Index (Lazy Load)
import gridfs
fs = gridfs.GridFS(idb, collection="faiss_index_files")

def load_faiss_index():
    global index
    if index is None:
        logger.info("[KB] ⏳ Loading FAISS index from GridFS...")
        existing_file = fs.find_one({"filename": "faiss_index.bin"})
        if existing_file:
            stored_index_bytes = existing_file.read()
            index_bytes_np = np.frombuffer(stored_index_bytes, dtype='uint8')
            index = faiss.deserialize_index(index_bytes_np)
            logger.info("[KB] ✅ FAISS Index Loaded")
        else:
            logger.error("[KB] ❌ FAISS index not found in GridFS.")
    return index

# ✅ Retrieve Medical Info (256,916 scenario)
def retrieve_medical_info(query, k=5, min_sim=0.8): # Min similarity between query and kb is to be 80%
    global index
    index = load_faiss_index()
    if index is None:
        return [""]
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
    # Final
    return kept if kept else [""]


# ✅ Retrieve Sym-Dia Info (4,962 scenario)
def retrieve_diagnosis_from_symptoms(symptom_text, top_k=5, min_sim=0.4):
    global SYMPTOM_VECTORS, SYMPTOM_DOCS
    # Lazy load
    if SYMPTOM_VECTORS is None:
        all_docs = list(symptom_col.find({}, {"embedding": 1, "answer": 1, "question": 1, "prognosis": 1}))
        SYMPTOM_DOCS = all_docs
        SYMPTOM_VECTORS = np.array([doc["embedding"] for doc in all_docs], dtype=np.float32)
    # Embed input
    qvec = embedding_model.encode(symptom_text, convert_to_numpy=True)
    qvec = qvec / (np.linalg.norm(qvec) + 1e-9)
    # Similarity compute
    sims = SYMPTOM_VECTORS @ qvec  # cosine
    sorted_idx = np.argsort(sims)[-top_k:][::-1]
    seen_diag = set()
    final = [] # Dedup
    for i in sorted_idx:
        sim = sims[i]
        if sim < min_sim:
            continue
        label = SYMPTOM_DOCS[i]["prognosis"]
        if label not in seen_diag:
            final.append(SYMPTOM_DOCS[i]["answer"])
            seen_diag.add(label)
    return final

# ✅ Gemini Flash API Call
def gemini_flash_completion(prompt, model, temperature=0.7):
    client_genai = genai.Client(api_key=gemini_flash_api_key)
    try:
        response = client_genai.models.generate_content(model=model, contents=prompt)
        return response.text
    except Exception as e:
        logger.error(f"[LLM] ❌ Error calling Gemini API: {e}")
        return "Error generating response from Gemini."

# ✅ Chatbot Class
class RAGMedicalChatbot:
    def __init__(self, model_name, retrieve_function):
        self.model_name = model_name
        self.retrieve = retrieve_function

    def chat(self, user_id: str, user_query: str, lang: str = "EN", image_diagnosis: str = "") -> str:
        # 0. Translate query if not EN, this help our RAG system
        if lang.upper() in {"VI", "ZH"}:
            user_query = translate_query(user_query, lang.lower())

        # 1. Fetch knowledge
        ## a. KB for generic QA retrieval
        retrieved_info = self.retrieve(user_query)
        knowledge_base = "\n".join(retrieved_info)
        ## b. Diagnosis RAG from symptom query
        diagnosis_guides = retrieve_diagnosis_from_symptoms(user_query)  # smart matcher

        # 2. Use relevant chunks from short-term memory FAISS index (nearest 3 chunks)
        context = memory.get_relevant_chunks(user_id, user_query, top_k=3)

        # 3. Build prompt parts
        parts = ["You are a medical chatbot, designed to answer medical questions."]
        parts.append("Please format your answer using MarkDown.")
        parts.append("**Bold for titles**, *italic for emphasis*, and clear headings.")
        # Append image diagnosis from VLM
        if image_diagnosis:
            parts.append(
                "User medical image is diagnosed by VLM agent:\n"
                f"{image_diagnosis}\n\n"
                "➡️ Please incorporate the above findings in your response if medically relevant.\n\n"
            )
        # Historical chat retrieval case
        if context:
            parts.append("Relevant context from prior conversation:\n" + "\n".join(context))
        # Load up guideline
        if knowledge_base:
            parts.append(f"Medical scenario knowledge: {knowledge_base}")
        # Symptom-Diagnosis prediction RAG
        if diagnosis_guides:
            parts.append("Symptom-based diagnosis guidance:\n" + "\n".join(diagnosis_guides))
        parts.append(f"Question: {user_query}")
        parts.append(f"Language: {lang}")
        prompt = "\n\n".join(parts)
        logger.info(f"[LLM] Question query in `prompt`: {prompt}") # Debug out checking RAG on kb and history
        response = gemini_flash_completion(prompt, model=self.model_name, temperature=0.7)
         # Store exchange + chunking
        if user_id:
            memory.add_exchange(user_id, user_query, response, lang=lang)
        logger.info(f"[LLM] Response on `prompt`: {response.strip()}") # Debug out base response
        return response.strip()

# ✅ Initialize Chatbot
chatbot = RAGMedicalChatbot(model_name="gemini-2.5-flash-preview-04-17", retrieve_function=retrieve_medical_info)

# ✅ Chat Endpoint
@app.post("/chat")
async def chat_endpoint(req: Request):
    body = await req.json()
    user_id = body.get("user_id", "anonymous")
    query_raw = body.get("query")
    query = query_raw.strip() if isinstance(query_raw, str) else ""
    lang    = body.get("lang", "EN")
    image_base64 = body.get("image_base64", None)
    img_desc = body.get("img_desc", "Describe and investigate any clinical findings from this medical image.")
    start = time.time()
    image_diagnosis = ""
    # LLM Only
    if not image_base64:
        logger.info("[BOT] LLM scenario.")
    # LLM+VLM
    else:
        # If image is present → diagnose first
        safe_load = len(image_base64.encode("utf-8"))
        if safe_load > 5_000_000: # Img size safe processor
            return JSONResponse({"response": "⚠️ Image too large. Please upload smaller images (<5MB)."})
        logger.info("[BOT] VLM+LLM scenario.")
        logger.info(f"[VLM] Process medical image size: {safe_load}, desc: {img_desc}, {lang}.")
        image_diagnosis = process_medical_image(image_base64, img_desc, lang)
    answer = chatbot.chat(user_id, query, lang, image_diagnosis)
    elapsed = time.time() - start
    # Final
    return JSONResponse({"response": f"{answer}\n\n(Response time: {elapsed:.2f}s)"})


# ✅ Run Uvicorn
if __name__ == "__main__":
    logger.info("[System] ✅ Starting FastAPI Server...")
    try:
        uvicorn.run(app, host="0.0.0.0", port=7860, log_level="debug")
    except Exception as e:
        logger.error(f"❌ Server Startup Failed: {e}")
        exit(1)
