import os
import faiss
import numpy as np
import time
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pymongo import MongoClient
from google import genai
from sentence_transformers import SentenceTransformer

# ✅ Enable Logging for Debugging
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("medical-chatbot")
# Debug Start
logger.info("🚀 Starting Medical Chatbot API...")
print("🚀 Starting Medical Chatbot API...")

# ✅ Environment Variables
mongo_uri = os.getenv("MONGO_URI")
index_uri = os.getenv("INDEX_URI")
gemini_flash_api_key = os.getenv("FlashAPI")
# Validate environment endpoint
if not all([gemini_flash_api_key, mongo_uri, index_uri]):
    raise ValueError("❌ Missing API keys! Set them in Hugging Face Secrets.")
logger.info(f"🔎 MongoDB URI: {mongo_uri}")
logger.info(f"🔎 FAISS Index URI: {index_uri}")

# ✅ Monitor Resources Before Startup
import psutil
def check_system_resources():
    memory = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=1)
    disk = psutil.disk_usage("/")
    # Defines log info messages
    logger.info(f"🔍 System Resources - RAM: {memory.percent}%, CPU: {cpu}%, Disk: {disk.percent}%")
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
logger.info("📥 Loading SentenceTransformer Model...")
print("📥 Loading SentenceTransformer Model...")
MODEL_CACHE_DIR = "/app/model_cache"
try:
    embedding_model = SentenceTransformer(MODEL_CACHE_DIR, device="cpu")
    embedding_model = embedding_model.half()  # Reduce memory
    logger.info("✅ Model Loaded Successfully.")
    print("✅ Model Loaded Successfully.")
except Exception as e:
    logger.error(f"❌ Model Loading Failed: {e}")
    exit(1)


# ✅ Setup MongoDB Connection
# QA data
client = MongoClient(mongo_uri)
db = client["MedicalChatbotDB"]
qa_collection = db["qa_data"]
# FAISS Index data
iclient = MongoClient(index_uri)
idb = iclient["MedicalChatbotDB"]
index_collection = idb["faiss_index_files"]

# ✅ Load FAISS Index (Lazy Load)
import gridfs
fs = gridfs.GridFS(idb, collection="faiss_index_files")

def load_faiss_index():
    global index
    if index is None:
        print("⏳ Loading FAISS index from GridFS...")
        existing_file = fs.find_one({"filename": "faiss_index.bin"})
        if existing_file:
            stored_index_bytes = existing_file.read()
            index_bytes_np = np.frombuffer(stored_index_bytes, dtype='uint8')
            index = faiss.deserialize_index(index_bytes_np)
            print("✅ FAISS Index Loaded")
            logger.info("✅ FAISS Index Loaded")
        else:
            print("❌ FAISS index not found in GridFS.")
            logger.error("❌ FAISS index not found in GridFS.")
    return index

# ✅ Retrieve Medical Info
def retrieve_medical_info(query):
    global index
    index = load_faiss_index()  # Load FAISS on demand
    # N/A question
    if index is None:
        return ["No medical information available."]
    # Embed the query and send to QA db to lookup
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    _, idxs = index.search(query_embedding, k=3)
    results = [qa_collection.find_one({"i": int(i)}).get("Doctor", "No answer available.") for i in idxs[0]]
    return results

# ✅ Gemini Flash API Call
def gemini_flash_completion(prompt, model, temperature=0.7):
    client_genai = genai.Client(api_key=gemini_flash_api_key)
    try:
        response = client_genai.models.generate_content(model=model, contents=prompt)
        return response.text
    except Exception as e:
        logger.error(f"❌ Error calling Gemini API: {e}")
        print(f"❌ Error calling Gemini API: {e}")
        return "Error generating response from Gemini."

# ✅ Chatbot Class
class RAGMedicalChatbot:
    def __init__(self, model_name, retrieve_function):
        self.model_name = model_name
        self.retrieve = retrieve_function

    def chat(self, user_query, lang="EN"):
        retrieved_info = self.retrieve(user_query)
        knowledge_base = "\n".join(retrieved_info)

        # ✅ Construct Prompt
        prompt = f"""
        You are a medical chatbot, designed to answer medical questions.
        
        Please format your answer using markdown. 
        **Bold for titles**, *italic for emphasis*, and clear headings.

        **Medical knowledge (trained with 256,916 data entries):**
        {knowledge_base}

        **Question:** {user_query}

        **Language Required:** {lang}
        """
        completion = gemini_flash_completion(prompt, model=self.model_name, temperature=0.7)
        return completion.strip()

# ✅ Initialize Chatbot
chatbot = RAGMedicalChatbot(model_name="gemini-2.5-flash-preview-04-17", retrieve_function=retrieve_medical_info)

# ✅ Chat Endpoint
@app.post("/chat")
async def chat_endpoint(data: dict):
    user_query = data.get("query", "")
    lang = data.get("lang", "EN")
    if not user_query:
        return JSONResponse(content={"response": "No query provided."})
    # Output parameter
    start_time = time.time()
    response_text = chatbot.chat(user_query, lang)
    end_time = time.time()
    response_text += f"\n\n(Response time: {end_time - start_time:.2f} seconds)"
    # Send JSON response
    return JSONResponse(content={"response": response_text})

# ✅ Run Uvicorn
if __name__ == "__main__":
    logger.info("✅ Starting FastAPI Server...")
    print("✅ Starting FastAPI Server...")
    try:
        uvicorn.run(app, host="0.0.0.0", port=7860, log_level="debug")
    except Exception as e:
        logger.error(f"❌ Server Startup Failed: {e}")
        exit(1)
