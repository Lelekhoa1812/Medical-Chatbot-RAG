# ==========================
# Medical Chatbot Backend (Gemini Flash API + RAG) - Local Prebuilt Model with FAISS Index Stored in MongoDB
# ==========================
"""
This script loads:
  1) A FAISS index stored in MongoDB (under the "faiss_index" collection)
  2) A local Hugging Face model cache (huggingface_models)
from the AutoGenRAGMedicalChatbot folder.
If the FAISS index is not found in MongoDB, the script will build a placeholder index,
serialize it, and store it in MongoDB.
The chatbot prompt instructs Gemini Flash to format its answer using markdown.
"""

import os
import faiss
import numpy as np
import gc
import time
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()
gemini_flash_api_key = os.getenv("FlashAPI")
mongo_uri = os.getenv("MongoURI")
if not gemini_flash_api_key:
    raise ValueError("❌ Gemini Flash API key (FlashAPI) is missing!")
if not mongo_uri:
    raise ValueError("❌ MongoDB URI (MongoURI) is missing!")

# --- Environment variables to mitigate segmentation faults ---
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Setup local project directory (for model cache) ---
project_dir = "./AutoGenRAGMedicalChatbot"
os.makedirs(project_dir, exist_ok=True)
huggingface_cache_dir = os.path.join(project_dir, "huggingface_models")
os.environ["HF_HOME"] = huggingface_cache_dir  # Use this folder for HF cache

# --- Download (or load from cache) the SentenceTransformer model ---
from huggingface_hub import snapshot_download
print("Checking or downloading the all-MiniLM-L6-v2 model from huggingface_hub...")
model_loc = snapshot_download(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    cache_dir=os.environ["HF_HOME"],
    local_files_only=False
)
print(f"Model directory: {model_loc}")

from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer(model_loc, device="cpu")

# --- MongoDB Setup for FAISS Index ---
from pymongo import MongoClient
client = MongoClient(mongo_uri)
db = client["MedicalChatbotDB"]  # Use your desired database name
index_collection = db["faiss_index"]

print("✅ Checking MongoDB for existing FAISS index...")
doc = index_collection.find_one({"_id": "faiss_index"})
if doc is None:
    print("⚠️ FAISS index not found in MongoDB. Building a placeholder index...")
    # For demonstration, we'll build an index from placeholder embeddings.
    # In a real scenario, you would load your dataset and compute embeddings.
    dim = embedding_model.get_sentence_embedding_dimension()
    # Create a small random array (e.g., 10 vectors) as a placeholder.
    placeholder_embeddings = np.random.rand(10, dim).astype(np.float32)
    index = faiss.IndexHNSWFlat(dim, 32)
    index.add(placeholder_embeddings)
    # Serialize the index to bytes and convert to a proper bytes object
    index_bytes = faiss.serialize_index(index)
    index_bytes = np.frombuffer(index_bytes, dtype='uint8')
    index_collection.insert_one({"_id": "faiss_index", "index": index_bytes.tobytes()})
    del placeholder_embeddings
    gc.collect()
    print("✅ FAISS index built and stored in MongoDB.")
else:
    print("✅ Loading existing FAISS index from MongoDB...")
    index_bytes = doc["index"]
    # Convert stored bytes back into a NumPy array of type uint8
    index_bytes_np = np.frombuffer(index_bytes, dtype='uint8')
    index = faiss.deserialize_index(index_bytes_np)
print("✅ FAISS index loaded successfully!")

# --- Prepare Retrieval and Chat Logic ---
# In production, you would load your actual QA pairs; here we use a placeholder.
# (Ensure your real system is trained with 50 data entries.)
# For this example, we simply set a placeholder list.
medical_qa = ["Dummy placeholder answer"]

def retrieve_medical_info(query):
    """Retrieve relevant medical knowledge using the FAISS index."""
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    _, idxs = index.search(query_embedding, k=3)
    # If you had a list of QA pairs, you might return:
    # return [medical_qa[i]["answer"] for i in idxs[0]]
    return [f"Prebuilt answer {i}" for i in idxs[0]]

# --- Gemini Flash API Call ---
from google import genai
def gemini_flash_completion(prompt, model, temperature=0.7):
    client_genai = genai.Client(api_key=gemini_flash_api_key)
    try:
        response = client_genai.models.generate_content(model=model, contents=prompt)
        return response.text
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return "Error generating response from Gemini."

# --- Chatbot Class ---
class RAGMedicalChatbot:
    def __init__(self, model_name, retrieve_function):
        self.model_name = model_name
        self.retrieve = retrieve_function

    def chat(self, user_query):
        retrieved_info = self.retrieve(user_query)
        knowledge_base = "\n".join(retrieved_info)
        # Add formatting instructions to the prompt:
        prompt = (
            "Please format your answer using markdown, where ** indicates bold text, "
            "and ensure that headings, rows, and sentences are clearly separated.\n\n"
            f"Using the following medical knowledge:\n{knowledge_base}\n\n"
            f"Answer the following question in a professional and medically accurate manner (trained with 50 data entries): {user_query}"
        )
        completion = gemini_flash_completion(prompt, model=self.model_name, temperature=0.7)
        return completion.strip()

chatbot = RAGMedicalChatbot(
    model_name="gemini-2.0-flash",
    retrieve_function=retrieve_medical_info
)
print("✅ Medical chatbot is ready!")

# --- FastAPI Server ---
app = FastAPI(title="Medical Chatbot")
HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <title>Medical Chatbot</title>
    <style>
        body { font-family: Arial, sans-serif; background-color: #f2f9fc; margin: 0; padding: 0; }
        .chat-container { width: 60%; margin: 40px auto; background: #fff; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        .chat-header { background-color: #0077b6; color: #fff; padding: 20px; text-align: center; border-top-left-radius: 8px; border-top-right-radius: 8px; }
        .chat-messages { padding: 20px; height: 400px; overflow-y: scroll; }
        .chat-input { display: flex; border-top: 1px solid #ddd; }
        .chat-input input { flex: 1; padding: 15px; border: none; outline: none; }
        .chat-input button { padding: 15px; background-color: #0077b6; color: #fff; border: none; cursor: pointer; }
        .message { margin-bottom: 15px; }
        .user { color: #0077b6; }
        .bot { color: #023e8a; }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            <h2>Medical Chatbot Doctor</h2>
        </div>
        <div class="chat-messages" id="chat-messages"></div>
        <div class="chat-input">
            <input type="text" id="user-input" placeholder="Type your question here..." />
            <button onclick="sendMessage()">Send</button>
        </div>
    </div>
    <script>
        async function sendMessage() {
            const input = document.getElementById('user-input');
            const message = input.value;
            if (!message) return;
            appendMessage('user', message);
            input.value = '';
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: message })
            });
            const data = await response.json();
            appendMessage('bot', data.response);
        }
        function appendMessage(role, text) {
            const messagesDiv = document.getElementById('chat-messages');
            const messageDiv = document.createElement('div');
            messageDiv.classList.add('message');
            messageDiv.innerHTML = `<strong class="${role}">${role === 'user' ? 'You' : 'Doctor'}:</strong> ${text}`;
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def get_home():
    return HTML_CONTENT

@app.post("/chat")
async def chat_endpoint(data: dict):
    user_query = data.get("query", "")
    if not user_query:
        return JSONResponse(content={"response": "No query provided."})
    start_time = time.time()
    response_text = chatbot.chat(user_query)
    end_time = time.time()
    response_text += f"\n\n(Response time: {end_time - start_time:.2f} seconds)"
    return JSONResponse(content={"response": response_text})

if __name__ == "__main__":
    import uvicorn
    print("\n🩺 Starting Medical Chatbot FastAPI server...\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
