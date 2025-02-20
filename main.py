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
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()
gemini_flash_api_key = os.getenv("FlashAPI")
mongo_uri = os.getenv("MONGO_URI")
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

# --- MongoDB Setup ---
from pymongo import MongoClient
client = MongoClient(mongo_uri)
db = client["MedicalChatbotDB"]  # Use your chosen database name
index_collection = db["faiss_index"]
qa_collection = db["qa_data"]

# --- Load or Build QA Data in MongoDB ---
print("✅ Checking MongoDB for existing QA data...")
if qa_collection.count_documents({}) == 0:
    print("⚠️ QA data not found in MongoDB. Loading dataset from Hugging Face...")
    from datasets import load_dataset
    dataset = load_dataset("ruslanmv/ai-medical-chatbot", cache_dir=huggingface_cache_dir)
    df = dataset["train"].to_pandas()[["Patient", "Doctor"]]
    # Add an index column to preserve order.
    df["i"] = range(len(df))
    qa_data = df.to_dict("records")
    # Insert in batches (e.g., batches of 1000) to avoid document size limits.
    batch_size = 1000
    for i in range(0, len(qa_data), batch_size):
        qa_collection.insert_many(qa_data[i:i+batch_size])
    print(f"✅ QA data stored in MongoDB. Total entries: {len(qa_data)}")
else:
    print("✅ Loaded existing QA data from MongoDB.")
    # Create an index on "i" (if not exists) to help with sorting.
    qa_collection.create_index("i")
    # Load all QA documents sorted by "i" using a cursor with batch size.
    qa_data = list(qa_collection.find({}, {"_id": 0}).sort("i", 1).batch_size(1000))
    print("Total QA entries loaded:", len(qa_data))

# --- Build or Load the FAISS Index from MongoDB ---
print("✅ Checking MongoDB for existing FAISS index...")
doc = index_collection.find_one({"_id": "faiss_index"})
if doc is None:
    print("⚠️ FAISS index not found in MongoDB. Building FAISS index from QA data...")
    # Compute embeddings for each QA pair by concatenating "Patient" and "Doctor" fields.
    texts = [item.get("Patient", "") + " " + item.get("Doctor", "") for item in qa_data]
    batch_size = 512  # Adjust based on available memory
    embeddings_list = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_embeddings = embedding_model.encode(batch, convert_to_numpy=True).astype(np.float32)
        embeddings_list.append(batch_embeddings)
        print(f"Encoded batch {i} to {i + len(batch)}")
    embeddings = np.vstack(embeddings_list)
    dim = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(dim, 32)
    index.add(embeddings)
    # Serialize the index to bytes and store it in MongoDB.
    index_bytes = faiss.serialize_index(index)
    index_bytes = np.frombuffer(index_bytes, dtype='uint8')
    index_collection.insert_one({"_id": "faiss_index", "index": index_bytes.tobytes()})
    del embeddings
    gc.collect()
    print("✅ FAISS index built and stored in MongoDB.")
else:
    print("✅ Loading existing FAISS index from MongoDB...")
    index_bytes = doc["index"]
    index_bytes_np = np.frombuffer(index_bytes, dtype='uint8')
    index = faiss.deserialize_index(index_bytes_np)
print("✅ FAISS index loaded successfully!")

# --- Prepare Retrieval and Chat Logic ---
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
        prompt = (
            "Please format your answer using markdown. Use **bold** for titles, *italic* for emphasis, "
            "and ensure that headings and paragraphs are clearly separated.\n\n"
            f"Using the following medical knowledge:\n{knowledge_base}\n\n"
            f"Answer the following question in a professional and medically accurate manner (trained with 256,916 data entries): {user_query}"
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

# HTML Template
HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Medical Chatbot</title>
  <link rel="website icon" type="png" href="img/logo.png">
  <!-- Google Font -->
  <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap" rel="stylesheet">
  <style>
    body {
      font-family: 'Roboto', sans-serif;
      background: url('img/background.gif') repeat-x center top;
      background-size: contain;
      margin: 0;
      padding: 0;
    }
    /* Nav bar */
    .navbar {
      display: flex;
      padding: 22px 0;
      align-items: center;
      max-width: 1200px;
      margin: 0 auto;
      justify-content: space-between;
    }
    .navbar .logo {
      gap: 10px;
      display: flex;
      align-items: center;
      text-decoration: none;
      position: relative;
    }
    .navbar .logo img {
      width: 70px;
      border-radius: 10%;
      transition: transform 0.2s ease;
    }
    .navbar .logo img:hover {
      transform: scale(1.1);
    }
    /* Tooltip (Thinking Cloud) Styles */
    .logo-tooltip {
      display: none;
      position: absolute;
      bottom: calc(100% - 12px);
      left: 50%;
      transform: translateX(-90%);
      background-color: #fff;
      color: rgb(18, 129, 144);
      padding: 8px 12px;
      border: 1px solid #ddd;
      border-radius: 10px;
      font-size: 0.9rem;
      white-space: nowrap;
      box-shadow: 0 2px 6px rgba(0,0,0,0.1);
      z-index: 10;
    }
    .logo-tooltip::after {
      content: "";
      position: absolute;
      top: 100%;
      left: 40%;
      transform: translateX(-50%);
      border-width: 6px;
      border-style: solid;
      border-color: #fff transparent transparent transparent;
    }
    .navbar .logo:hover .logo-tooltip {
      display: block;
    }
    .navbar .links {
      display: flex;
      gap: 35px;
      list-style: none;
      align-items: center;
      margin: 0;
      padding: 0;
    }
    .navbar .links a {
      color: rgb(18, 129, 144);
      font-size: 1.1rem;
      font-weight: 500;
      text-decoration: none;
      transition: 0.1s ease;
    }
    .navbar .links a:hover {
      color: rgb(144, 100, 18);
    }
    /* Language Dropdown */
    .dropdown {
      position: relative;
      display: inline-block;
    }
    .dropdown-btn {
      background: none;
      border: none;
      color: rgb(18, 129, 144);
      font-size: 1.1rem;
      font-weight: 500;
      cursor: pointer;
      transition: 0.1s ease;
    }
    .dropdown-btn:hover {
      color: rgb(144, 100, 18);
    }
    .dropdown-menu {
      display: none;
      position: absolute;
      top: 110%;
      left: 0;
      background-color: #fff;
      min-width: 140px;
      box-shadow: 0 8px 16px rgba(0,0,0,0.2);
      z-index: 1;
      list-style: none;
      padding: 0;
      margin: 0;
      border-radius: 4px;
      overflow: hidden;
    }
    .dropdown-menu li {
      padding: 10px;
      cursor: pointer;
      color: rgb(18, 129, 144);
      transition: background-color 0.2s ease;
    }
    .dropdown-menu li:hover {
      background-color: #f1f1f1;
      color: rgb(144, 100, 18);
    }
    /* Text content */
    h1 {
      color: rgb(18, 129, 144);
    }
    h1:hover {
      color: rgb(144, 100, 18);
    }
    /* Chat container */
    .chat-container {
      width: 90%;
      max-width: 800px;
      margin: 15px auto;
      background: #fff;
      border-radius: 10px;
      box-shadow: 0 8px 16px rgba(0,0,0,0.15);
      overflow: hidden;
    }
    .chat-header {
      background-color: rgb(18, 129, 144);
      color: #fff;
      padding: 20px;
      text-align: center;
      font-size: 1.5em;
    }
    .chat-messages {
      padding: 20px;
      height: 450px;
      overflow-y: auto;
      background-color: #f9f9f9;
      position: relative;
    }
    /* Welcome screen (before chat stage) */
    #welcome-container {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: top;
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      text-align: center;
      pointer-events: none;
    }
    #welcome-container img {
      width: 80px;
      margin: 30px 10px;
      filter: grayscale(90%);
    }
    #welcome-container p {
      font-size: 0.8rem;
      color: rgb(117, 117, 117);
      margin: 0 150px;
    }
    #welcome-container a {
      font-size: 0.8rem;
      color: rgb(117, 117, 117);
      margin: 0 150px;
    }
    #welcome-container h1 {
      font-size: 1rem;
      color: rgb(117, 117, 117);
      margin: 5px 5px;
    }
    .chat-input {
      display: flex;
      border-top: 1px solid #ddd;
    }
    .chat-input input {
      flex: 1;
      padding: 15px;
      border: none;
      font-size: 1em;
      outline: none;
    }
    .chat-input button {
      padding: 15px;
      background-color: rgb(18, 129, 144);
      color: #fff;
      border: none;
      font-size: 1em;
      cursor: pointer;
      transition: background-color 0.3s ease, transform 0.3s ease;
    }
    .chat-input button:hover {
      background-color: #4e1402;
      transform: scale(1.1);
    }
    .message {
      margin-bottom: 15px;
      padding: 10px;
      border-radius: 5px;
      animation: fadeIn 0.5s ease-in-out;
    }
    .user {
      background-color: #fafafa;
      color: #b68f00;
    }
    .bot {
      background-color: #fafafa;
      color: #00b68f;
    }
    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(10px); }
      to { opacity: 1; transform: translateY(0); }
    }
  </style>
  <!-- Include Marked.js for markdown rendering -->
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
</head>
<body>
  <header>
    <nav class="navbar">
      <a href="#" class="logo">
        <img src="img/logo.png" alt="logo">
        <!-- Tooltip container -->
        <div class="logo-tooltip">Hello, how can I help you today?</div>
        <h1>Medical Chatbot Doctor</h1>
      </a>
      <ul class="links">
        <li><a href="account.html">Account</a></li>
        <li><a href="subscription.html">Subscription</a></li>
        <li><a href="about.html">About</a></li>
        <!-- Dropdown Language Selector -->
        <li class="dropdown">
          <button class="dropdown-btn">VI &#x25BC;</button>
          <ul class="dropdown-menu">
            <li data-lang="VI">Vietnamese</li>
            <li data-lang="EN">English</li>
            <li data-lang="ZH">Mandarin</li>
          </ul>
        </li>
      </ul>
    </nav>
  </header>
  <div class="chat-container">
    <div class="chat-header">
      Medical Chatbot Doctor
    </div>
    <div class="chat-messages" id="chat-messages">
      <!-- Welcome Screen (visible initially) -->
      <div id="welcome-container">
        <img src="img/logo.png" alt="Welcome Logo">
        <p>Hi! I’m your dedicated health assistant, here to support you with all your wellness questions. Feel free to ask me any question about your health and well-being.</p>
        <h1>Acknowledgement</h1>
        <p>Author: Dang Khoa Le</p>
        <a id="license" href="https://github.com/Lelekhoa1812/AutoGen-RAG-Medical-Chatbot/blob/main/LICENSE">License: Apache 2.0 License</a>
      </div>
    </div>
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
      // Remove the welcome screen if it exists
      const welcomeContainer = document.getElementById('welcome-container');
      if (welcomeContainer) {
        welcomeContainer.remove();
      }
      appendMessage('user', message, false);
      input.value = '';
      const response = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: message })
      });
      const data = await response.json();
      const htmlResponse = marked.parse(data.response);
      appendMessage('bot', htmlResponse, true);
    }
    function appendMessage(role, text, isHTML) {
      const messagesDiv = document.getElementById('chat-messages');
      const messageDiv = document.createElement('div');
      messageDiv.classList.add('message');
      if (isHTML) {
        messageDiv.innerHTML = `<strong class="${role}">${role === 'user' ? 'You' : 'DocBot'}:</strong><br/>${text}`;
      } else {
        messageDiv.innerHTML = `<strong class="${role}">${role === 'user' ? 'You' : 'DocBot'}:</strong> ${text}`;
      }
      messagesDiv.appendChild(messageDiv);
      messagesDiv.scrollTop = messagesDiv.scrollHeight;
    }

    // Dropdown language selector functionality
    document.addEventListener('DOMContentLoaded', function() {
      const dropdownBtn = document.querySelector('.dropdown-btn');
      const dropdownMenu = document.querySelector('.dropdown-menu');
      
      dropdownBtn.addEventListener('click', function(event) {
        event.stopPropagation();
        dropdownMenu.style.display = dropdownMenu.style.display === 'block' ? 'none' : 'block';
      });

      // Update dropdown button text when a language option is selected
      document.querySelectorAll('.dropdown-menu li').forEach(item => {
        item.addEventListener('click', function(event) {
          event.stopPropagation();
          const selectedLang = this.getAttribute('data-lang');
          dropdownBtn.innerHTML = selectedLang + " &#x25BC;";
          dropdownMenu.style.display = 'none';
          // Optional: Add any language change functionality here.
        });
      });

      // Close the dropdown if clicking outside
      document.addEventListener('click', function() {
        dropdownMenu.style.display = 'none';
      });
    });
  </script>
</body>
</html>
"""

# Get statics template route
@app.get("/", response_class=HTMLResponse)
async def get_home():
    return HTML_CONTENT

# Chat route
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
