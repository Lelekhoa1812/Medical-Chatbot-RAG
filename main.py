# ==========================
# Medical Chatbot Backend (Gemini Flash API + RAG) - Local Prebuilt Model with FAISS Index & Data Stored in MongoDB
# ==========================
"""
This script loads:
  1) A FAISS index stored in MongoDB (in the "faiss_index" collection)
  2) A local SentenceTransformer model (downloaded via snapshot_download)
  3) QA data (the full dataset of 256916 QA entries) stored in MongoDB (in the "qa_data" collection)

If the QA data or FAISS index are not found in MongoDB, the script loads the full dataset from Hugging Face,
computes embeddings for all QA pairs (concatenating the "Patient" and "Doctor" fields), and stores both the raw QA data
and the FAISS index in MongoDB.

The chatbot instructs Gemini Flash to format its answer using markdown.
"""

import os
import faiss
import numpy as np
import gc
import time
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pathlib import Path
# from dotenv import load_dotenv

# 🔹 Load environment variables from .env
# load_dotenv()
# gemini_flash_api_key = os.getenv("FlashAPI")
# mongo_uri = os.getenv("MONGO_URI")
# index_uri = os.getenv("INDEX_URI")
# 🔹 Load Streamlit secrets from .toml
gemini_flash_api_key = st.secrets["general"]["FlashAPI"]
mongo_uri = st.secrets["general"]["MONGO_URI"]
index_uri = st.secrets["general"]["INDEX_URI"]
if not gemini_flash_api_key:
    # raise ValueError("❌ Gemini Flash API key (FlashAPI) is missing!")
    st.error("❌ Gemini Flash API key (FlashAPI) is missing!")
    st.stop()  # Prevent the app from running without necessary API keys
if not mongo_uri:
    # raise ValueError("❌ MongoDB URI (MongoURI) is missing!")
    st.error("❌ MongoDB URI (MongoURI) is missing!")
    st.stop()  # Prevent the app from running without necessary API keys
if not index_uri:
    # raise ValueError("❌ INDEX_URI for FAISS index cluster is missing!")
    st.error("❌ INDEX_URI for FAISS index cluster is missing!")
    st.stop()  # Prevent the app from running without necessary API keys

# 1. Environment variables to mitigate segmentation faults 
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# 2. Setup local project directory (for model cache) 
project_dir = "./AutoGenRAGMedicalChatbot"
os.makedirs(project_dir, exist_ok=True)
huggingface_cache_dir = os.path.join(project_dir, "huggingface_models")
os.environ["HF_HOME"] = huggingface_cache_dir  # Use this folder for HF cache
# 3. Download (or load from cache) the SentenceTransformer model 
from huggingface_hub import snapshot_download
print("⏳ Checking or downloading the all-MiniLM-L6-v2 model from huggingface_hub...")
st.write("⏳ Checking or downloading the all-MiniLM-L6-v2 model from huggingface_hub...")
model_loc = snapshot_download(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    cache_dir=os.environ["HF_HOME"],
    local_files_only=False
)
print(f"✅ Model directory: {model_loc}")
st.write(f"✅ Model directory: {model_loc}")

from sentence_transformers import SentenceTransformer
print("📥 **Loading Embedding Model...**")
st.write("📥 **Loading Embedding Model...**")
embedding_model = SentenceTransformer(model_loc, device="cpu")

# 🔹 MongoDB Setup
from pymongo import MongoClient
# 1. QA client
client = MongoClient(mongo_uri)
db = client["MedicalChatbotDB"]  # Use your chosen database name
qa_collection = db["qa_data"]
# 2. FAISS index client
iclient = MongoClient(index_uri)
idb = iclient["MedicalChatbotDB"]  # Use your chosen database name
index_collection = idb["faiss_index_files"]

##---------------------------##
## EMBEDDING AND DATA RETRIEVAL
##---------------------------##

# 🔹 Load or Build QA Data in MongoDB
print("⏳ Checking MongoDB for existing QA data...")
st.write("⏳ Checking MongoDB for existing QA data...")
if qa_collection.count_documents({}) == 0:
    print("⚠️ QA data not found in MongoDB. Loading dataset from Hugging Face...")
    st.write("⚠️ QA data not found in MongoDB. Loading dataset from Hugging Face...")
    from datasets import load_dataset
    dataset = load_dataset("ruslanmv/ai-medical-chatbot", cache_dir=huggingface_cache_dir)
    df = dataset["train"].to_pandas()[["Patient", "Doctor"]]
    # Add an index column "i" to preserve order.
    df["i"] = range(len(df))
    qa_data = df.to_dict("records")
    # Insert in batches (e.g., batches of 1000) to avoid document size limits.
    batch_size = 1000
    for i in range(0, len(qa_data), batch_size):
        qa_collection.insert_many(qa_data[i:i+batch_size])
    print(f"📦 QA data stored in MongoDB. Total entries: {len(qa_data)}")
    st.success(f"📦 QA data stored in MongoDB. Total entries: {len(qa_data)}")
else:
    print("✅ Loaded existing QA data from MongoDB.")
    st.write("✅ Loaded existing QA data from MongoDB.")
    # Use an aggregation pipeline with allowDiskUse to sort by "i" without creating an index.
    qa_docs = list(qa_collection.aggregate([
        {"$sort": {"i": 1}},
        {"$project": {"_id": 0}}
    ], allowDiskUse=True))
    qa_data = qa_docs
    print("📦 Total QA entries loaded:", len(qa_data))
    st.success("📦 Total QA entries loaded:", len(qa_data))

# 🔹 Build or Load the FAISS Index from MongoDB using GridFS (on the separate cluster) 
print("⏳ Checking GridFS for existing FAISS index...")
st.write("⏳ Checking GridFS for existing FAISS index...")
import gridfs
fs = gridfs.GridFS(idb, collection="faiss_index_files")  # 'idb' is connected using INDEX_URI
# 1. Find the FAISS index file by filename.
existing_file = fs.find_one({"filename": "faiss_index.bin"})
if existing_file is None:
    print("⚠️ FAISS index not found in GridFS. Building FAISS index from QA data...")
    st.write("⚠️ FAISS index not found in GridFS. Building FAISS index from QA data...")
    # Compute embeddings for each QA pair by concatenating "Patient" and "Doctor" fields.
    texts = [item.get("Patient", "") + " " + item.get("Doctor", "") for item in qa_data]
    batch_size = 512  # Adjust as needed
    embeddings_list = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_embeddings = embedding_model.encode(batch, convert_to_numpy=True).astype(np.float32)
        embeddings_list.append(batch_embeddings)
        print(f"Encoded batch {i} to {i + len(batch)}")
        st.write(f"Encoded batch {i} to {i + len(batch)}")
    embeddings = np.vstack(embeddings_list)
    dim = embeddings.shape[1]
    # Create a FAISS index (using IndexHNSWFlat; or use IVFPQ for compression)
    index = faiss.IndexHNSWFlat(dim, 32)
    index.add(embeddings)
    print("FAISS index built. Total vectors:", index.ntotal)
    # Serialize the index
    index_bytes = faiss.serialize_index(index)
    index_data = np.frombuffer(index_bytes, dtype='uint8').tobytes()
    # Store in GridFS (this bypasses the 16 MB limit)
    file_id = fs.put(index_data, filename="faiss_index.bin")
    print("📦 FAISS index built and stored in GridFS with file_id:", file_id)
    st.success("📦 FAISS index built and stored in GridFS with file_id:", file_id)
    del embeddings
    gc.collect()
else:
    print("✅ Found FAISS index in GridFS. Loading...")
    st.write("✅ Found FAISS index in GridFS. Loading...")
    stored_index_bytes = existing_file.read()
    index_bytes_np = np.frombuffer(stored_index_bytes, dtype='uint8')
    index = faiss.deserialize_index(index_bytes_np)
print("📦 FAISS index loaded from GridFS successfully!")
st.success("📦 FAISS index loaded from GridFS successfully!")


##---------------------------##
## INFERENCE BACK+FRONT END
##---------------------------##

# 🔹 Prepare Retrieval and Chat Logic
def retrieve_medical_info(query):
    """Retrieve relevant medical knowledge using the FAISS index."""
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    _, idxs = index.search(query_embedding, k=3)
    results = []
    for i in idxs[0]:
        if i < len(qa_data):
            results.append(qa_data[i].get("Doctor", "No answer available."))
        else:
            results.append("No answer available.")
    return results

# 🔹 Gemini Flash API Call
from google import genai
def gemini_flash_completion(prompt, model, temperature=0.7):
    client_genai = genai.Client(api_key=gemini_flash_api_key)
    try:
        response = client_genai.models.generate_content(model=model, contents=prompt)
        return response.text
    except Exception as e:
        print(f"⚠️ Error calling Gemini API: {e}")
        st.error(f"⚠️ Error calling Gemini API: {e}")
        return "Error generating response from Gemini."

# Define a simple language mapping (modify or add more as needed)
language_map = {
    "EN": "English",
    "VI": "Vietnamese",
    "ZH": "Chinese"
}

# 🔹 Chatbot Class 
class RAGMedicalChatbot:
    def __init__(self, model_name, retrieve_function):
        self.model_name = model_name
        self.retrieve = retrieve_function

    def chat(self, user_query, lang="EN"):
        retrieved_info = self.retrieve(user_query)
        knowledge_base = "\n".join(retrieved_info)
        # Construct prompt for Gemini Flash
        prompt = (
            "Please format your answer using markdown. Use **bold** for titles, *italic* for emphasis, "
            "and ensure that headings and paragraphs are clearly separated.\n\n"
            f"Using the following medical knowledge:\n{knowledge_base} \n(trained with 256,916 data entries).\n\n"
            f"Answer the following question in a professional and medically accurate manner:\n{user_query}.\n\n"
            f"Your response answer must be in {lang} language."
        )
        completion = gemini_flash_completion(prompt, model=self.model_name, temperature=0.7)
        return completion.strip()

# 🔹 Model Class (change to others if needed)
chatbot = RAGMedicalChatbot(
    model_name="gemini-2.0-flash",
    retrieve_function=retrieve_medical_info
)
print("✅ Medical chatbot is ready! 🤖")
st.success("✅ Medical chatbot is ready! 🤖")

# 🔹 FastAPI Server
# from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware # Bypassing CORS origin
app = FastAPI(title="Medical Chatbot")
# 1. Define the origins
origins = [
    "http://localhost:5173",            # Vite dev server
    "http://localhost:3000",            # Another vercel dev server
    "medical-chatbot-henna.vercel.app", # ✅ Vercel frontend production URL
]
# 2. Then add the CORS middleware:
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,   # or ["*"] to allow all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# (02/03/2025) Move static files UI to Vercel
# 3. Mount static files (make sure the "static" folder exists and contains your images)
# app.mount("/static", StaticFiles(directory="static"), name="static")
# 4. Get statics template route
# @app.get("/", response_class=HTMLResponse)
# async def get_home():
#     return HTML_CONTENT

# 🔹 Chat route
@app.post("/chat")
async def chat_endpoint(data: dict):
    user_query = data.get("query", "")
    lang = data.get("lang", "EN")  # Expect a language code from the request
    if not user_query:
        return JSONResponse(content={"response": "No query provided."})
    start_time = time.time()
    response_text = chatbot.chat(user_query, lang)  # Pass language selection
    end_time = time.time()
    response_text += f"\n\n(Response time: {end_time - start_time:.2f} seconds)"
    return JSONResponse(content={"response": response_text})

# 🔹 Main Execution
# 1. On Streamlit (free-tier allowance 1GB)
import streamlit as st
import threading
import requests
# 🌐 Start FastAPI server in a separate thread
def run_fastapi():
    uvicorn.run(app, host="0.0.0.0", port=8000)
threading.Thread(target=run_fastapi, daemon=True).start()

# 🔍 Streamlit UI for Testing
st.title("🩺 Medical Chatbot API")
st.info("This is a **FastAPI Backend running on Streamlit Cloud**")
user_query = st.text_input("Enter your medical question:")
selected_lang = st.selectbox("Select Language:", ["English (EN)", "Vietnamese (VI)", "Chinese (ZH)"])
if st.button("Ask Doctor Bot"):
    lang_code = selected_lang.split("(")[-1].strip(")")
    st.markdown("🤖 **DocBot is thinking...**")
    # a) API request to FastAPI
    response = requests.post("http://localhost:8000/chat", json={"query": user_query, "lang": lang_code})
    response_json = response.json()
    # b) Display response
    st.markdown(response_json["response"])

# 2. On Render (free-tier allowance 52MB)
# if __name__ == "__main__":
#     import uvicorn
#     print("\n🩺 Starting Medical Chatbot FastAPI server...\n")
#     # 🌐 Start app
#     uvicorn.run(app, host="0.0.0.0", port=8000)