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
index_uri = os.getenv("INDEX_URI")
if not gemini_flash_api_key:
    raise ValueError("❌ Gemini Flash API key (FlashAPI) is missing!")
if not mongo_uri:
    raise ValueError("❌ MongoDB URI (MongoURI) is missing!")
if not index_uri:
    raise ValueError("❌ INDEX_URI for FAISS index cluster is missing!")

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
# QA client
client = MongoClient(mongo_uri)
db = client["MedicalChatbotDB"]  # Use your chosen database name
qa_collection = db["qa_data"]

# FAISS index client
iclient = MongoClient(index_uri)
idb = iclient["MedicalChatbotDB"]  # Use your chosen database name
index_collection = idb["faiss_index_files"]

##---------------------------##
## EMBEDDING AND DATA RETRIEVAL
##---------------------------##

# --- Load or Build QA Data in MongoDB ---
print("✅ Checking MongoDB for existing QA data...")
if qa_collection.count_documents({}) == 0:
    print("⚠️ QA data not found in MongoDB. Loading dataset from Hugging Face...")
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
    print(f"✅ QA data stored in MongoDB. Total entries: {len(qa_data)}")
else:
    print("✅ Loaded existing QA data from MongoDB.")
    # Use an aggregation pipeline with allowDiskUse to sort by "i" without creating an index.
    qa_docs = list(qa_collection.aggregate([
        {"$sort": {"i": 1}},
        {"$project": {"_id": 0}}
    ], allowDiskUse=True))
    qa_data = qa_docs
    print("Total QA entries loaded:", len(qa_data))

# --- Build or Load the FAISS Index from MongoDB using GridFS (on the separate cluster) ---
print("✅ Checking GridFS for existing FAISS index...")
import gridfs
fs = gridfs.GridFS(idb, collection="faiss_index_files")  # 'idb' is connected using INDEX_URI

# Find the FAISS index file by filename.
existing_file = fs.find_one({"filename": "faiss_index.bin"})
if existing_file is None:
    print("⚠️ FAISS index not found in GridFS. Building FAISS index from QA data...")
    # Compute embeddings for each QA pair by concatenating "Patient" and "Doctor" fields.
    texts = [item.get("Patient", "") + " " + item.get("Doctor", "") for item in qa_data]
    batch_size = 512  # Adjust as needed
    embeddings_list = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_embeddings = embedding_model.encode(batch, convert_to_numpy=True).astype(np.float32)
        embeddings_list.append(batch_embeddings)
        print(f"Encoded batch {i} to {i + len(batch)}")
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
    print("✅ FAISS index built and stored in GridFS with file_id:", file_id)
    del embeddings
    gc.collect()
else:
    print("✅ Found FAISS index in GridFS. Loading...")
    stored_index_bytes = existing_file.read()
    index_bytes_np = np.frombuffer(stored_index_bytes, dtype='uint8')
    index = faiss.deserialize_index(index_bytes_np)
print("✅ FAISS index loaded from GridFS successfully!")


##---------------------------##
## INFERENCE BACK+FRONT END
##---------------------------##

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

# --- Define a simple language mapping (modify or add more as needed)
language_map = {
    "EN": "English",
    "VI": "Vietnamese",
    "ZH": "Chinese"
}

# --- Chatbot Class ---
class RAGMedicalChatbot:
    def __init__(self, model_name, retrieve_function):
        self.model_name = model_name
        self.retrieve = retrieve_function
        self.translator = Translator()  # Initialize Google Translator

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

# --- Model Class (change to others if needed) ---
chatbot = RAGMedicalChatbot(
    model_name="gemini-2.0-flash",
    retrieve_function=retrieve_medical_info
)
print("✅ Medical chatbot is ready!")

# --- FastAPI Server ---
from fastapi.staticfiles import StaticFiles
app = FastAPI(title="Medical Chatbot")

# HTML Template
HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Medical Chatbot</title>
  <link rel="website icon" type="png" href="/static/img/logo.png">
  <!-- Google Font -->
  <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap" rel="stylesheet">
  <style>
    /* General Styling */
    body {
      font-family: 'Roboto', sans-serif;
      background: linear-gradient(270deg, rgb(44, 13, 58), rgb(13, 58, 56));
      background-size: cover;
      margin: 0;
      padding: 0;
    }
    /* Navbar & Logo */
    .navbar {
      display: flex;
      padding: 22px 0;
      align-items: center;
      max-width: 1200px;
      margin: 0 auto;
      justify-content: space-between;
    }
    .navbar .logo {
      display: flex;
      align-items: center;
      gap: 10px;
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
    #nav-header {
      color: rgb(18, 129, 144);
    }
    #nav-header:hover {
      color: rgb(144, 100, 18);
      transform: translateX(5px) translateY(-1px) scale(1.1);
      transition: transform 0.2s ease;
    }
    /* Tooltip (Thinking Cloud) */
    .logo-tooltip {
      display: none;
      position: absolute;
      bottom: calc(100% - 12px);
      left: 0; /* Align left edge with logo image */
      background: url('/static/img/cloud.gif') repeat;
      background-size: 100% 100%;
      color: rgb(18, 129, 144);
      padding: 10px 60px;
      font-size: 0.9rem;
      white-space: nowrap;
      z-index: 10;
    }
    .navbar .logo:hover .logo-tooltip {
      display: block;
    }
    /* Navbar Links & Language Dropdown */
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
      left: -90px;
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
    /* Chat Container */
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
      background: linear-gradient(270deg, rgb(13, 58, 56), rgb(44, 13, 58));
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
      min-height: 60vh; /* Ensure the container covers the full viewport height */
    }
    /* Tablet Devices */
    @media (max-width: 1100px) {
      .chat-messages {
        min-height: 70vh; /* Ensure the container covers the full viewport height */
      }
    }
    /* Mobile Devices */
    @media (max-width: 768px) {
      .chat-messages {
        min-height: 80vh; /* Ensure the container covers the full viewport height */
      }
    }
    /* Welcome Screen */
    #welcome-container {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: flex-start;
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
    #welcome-container p, 
    #welcome-container h1, 
    #welcome-container a {
      margin: 5px 150px;
      color: rgb(117, 117, 117);
      font-size: 0.8rem;
    }
    #welcome-container h1 {
      font-size: 1rem;
    }
    /* Chat Input */
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
      background: linear-gradient(270deg, rgb(13, 58, 56), rgb(44, 13, 58));
      color: #fff;
      border: none;
      font-size: 1em;
      cursor: pointer;
      transition: background-color 0.3s ease, transform 0.3s ease;
    }
    .chat-input button:hover {
      background: linear-gradient(270deg, rgb(144, 100, 18), rgb(52, 18, 8)); 
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
      color: #942402;
    }
    .bot {
      background-color: #fafafa;
      color: #00b68f;
    }
    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(10px); }
      to { opacity: 1; transform: translateY(0); }
    }
    /* Loader Styles */
    .loader-container {
      display: flex;
      flex-direction: column;
      align-items: center;
      margin: 10px 0;
    }
    .loader {
      border: 8px solid #f3f3f3;
      border-top: 8px solid rgb(18, 129, 144);
      border-radius: 50%;
      width: 60px;
      height: 60px;
      animation: spin 1s linear infinite;
    }
    .loader-text {
      font-size: 1rem;
      color: rgb(18, 129, 144);
      margin-top: 8px;
    }
    @keyframes spin {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
    }
    /* Modal Styles */
    #language-modal {
      display: flex;
      position: fixed;
      z-index: 1000;
      left: 0;
      top: 0;
      width: 100%;
      height: 100%;
      overflow: auto;
      background-color: rgba(0,0,0,0.5);
      align-items: center;
      justify-content: center;
    }
    #language-modal .modal-content {
      background-color: #fff;
      padding: 30px;
      border-radius: 10px;
      text-align: center;
      max-width: 500px;
      width: 90%;
      box-shadow: 0 8px 16px rgba(0,0,0,0.3);
    }
    #language-modal h2 {
      color: rgb(18, 129, 144);
      margin-bottom: 20px;
    }
    #language-modal button {
      background: linear-gradient(270deg, rgb(44, 13, 58), rgb(13, 58, 56));
      color: #fff;
      border: none;
      padding: 10px 20px;
      margin: 10px;
      border-radius: 5px;
      cursor: pointer;
      font-size: 1rem;
      transition: background-color 0.2s ease;
    }
    #language-modal button:hover {
      background: linear-gradient(270deg, rgb(144, 100, 18), rgb(52, 18, 8)); 
    }
  </style>
</head>
<body>
  <!-- Language Selection Modal -->
  <div id="language-modal">
    <div class="modal-content">
      <h2>Please select your preferred language</h2>
      <button data-lang="EN">English</button>
      <button data-lang="VI">Tiếng Việt</button>
      <button data-lang="ZH">中文</button>
    </div>
  </div>

  <header>
    <nav class="navbar">
      <a href="#" class="logo">
        <img src="/static/img/logo.png" alt="logo">
        <!-- Tooltip container using cloud.gif -->
        <div id="tooltip" class="logo-tooltip">Hello, how can I support you today?</div>
        <h1 id="nav-header">Medical Chatbot Doctor</h1>
      </a>
      <ul class="links">
        <li><a id="nav-account" href="account.html">Account</a></li>
        <li><a id="nav-subscription" href="subscription.html">Subscription</a></li>
        <li><a id="nav-about" href="https://github.com/Lelekhoa1812/AutoGen-RAG-Medical-Chatbot">About</a></li>
        <!-- Dropdown Language Selector -->
        <li class="dropdown">
          <button class="dropdown-btn">EN &#x25BC;</button>
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
    <div id="chat-header" class="chat-header">Medical Chatbot Doctor</div>
    <div class="chat-messages" id="chat-messages">
      <!-- Welcome Screen (visible initially) -->
      <div id="welcome-container">
        <img src="/static/img/logo.png" alt="Welcome Logo">
        <p id="welcome-text">Hi! I’m your dedicated health assistant, here to support you with all your wellness questions. Feel free to ask me any question about your health and well-being.</p>
        <h1 id="acknowledgement">Acknowledgement</h1>
        <p id="author">Author: (Liam) Dang Khoa Le</p>
        <a id="license" href="https://github.com/Lelekhoa1812/AutoGen-RAG-Medical-Chatbot/blob/main/LICENSE">License: Apache 2.0 License</a>
      </div>
    </div>
    <div class="chat-input">
      <input type="text" id="user-input" placeholder="Type your question here..." />
      <button onclick="sendMessage()">Send</button>
    </div>
  </div>

  <!-- Include Marked.js for markdown rendering -->
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <script>
    // Global variable for current language (default English)
    let currentLang = "EN";

    // Translation strings
    const translations = {
      "EN": {
        header: "Medical Chatbot Doctor",
        tooltip: "Hello, how can I help you today?",
        welcomeText: "Hi! I’m your dedicated health assistant, here to support you with all your wellness questions. Feel free to ask me any question about your health and well-being.",
        acknowledgement: "Acknowledgement",
        author: "Author: (Liam) Dang Khoa Le",
        license: "License: Apache 2.0 License",
        chatInputPlaceholder: "Type your question here...",
        you: "You",
        bot: "DocBot",
        account: "Account",
        subscription: "Subscription",
        about: "About",
        loaderMessage: "Doctor Chatbot is finding the best solution for you, hang tight..."
      },
      "VI": {
        header: "Bác Sĩ Chatbot",
        tooltip: "Xin chào, tôi có thể giúp gì cho bạn?",
        welcomeText: "Chào bạn! Tôi là trợ lý sức khỏe tận tâm của bạn, sẵn sàng hỗ trợ mọi thắc mắc về sức khỏe và phúc lợi của bạn. Hãy thoải mái đặt câu hỏi nhé!",
        acknowledgement: "Thông tin",
        author: "Tác giả: Lê Đăng Khoa",
        license: "Giấy phép: Apache 2.0",
        chatInputPlaceholder: "Nhập câu hỏi của bạn...",
        you: "Bạn",
        bot: "Bác Sĩ Chatbot",
        account: "Tài Khoản",
        subscription: "Đăng Ký",
        about: "Thông Tin",
        loaderMessage: "Bác sĩ Chatbot đang tìm giải pháp tốt nhất cho bạn, vui lòng chờ trong giây lát..."
      },
      "ZH": {
        header: "医疗聊天机器人医生",
        tooltip: "您好，我今天能为您提供什么帮助？",
        welcomeText: "您好！我是您专属的健康助手，随时为您解答关于健康与福祉的问题。请随时向我提问。",
        acknowledgement: "鸣谢",
        author: "作者：黎登科",
        license: "许可证：Apache 2.0 许可证",
        chatInputPlaceholder: "请输入您的问题...",
        you: "您",
        bot: "医生机器人",
        account: "账户",
        subscription: "订阅",
        about: "关于",
        loaderMessage: "医生聊天机器人正在为您寻找最佳解决方案，请稍候…"
      }
    };

    // Function to update all UI strings based on selected language
    function updateLanguage(lang) {
      currentLang = lang;
      // Update nav header and tooltip
      document.getElementById('nav-header').innerText = translations[lang].header;
      document.getElementById('tooltip').innerText = translations[lang].tooltip;
      // Update chat header
      document.getElementById('chat-header').innerText = translations[lang].header;
      // Update welcome screen texts
      document.getElementById('welcome-text').innerText = translations[lang].welcomeText;
      document.getElementById('acknowledgement').innerText = translations[lang].acknowledgement;
      document.getElementById('author').innerText = translations[lang].author;
      document.getElementById('license').innerText = translations[lang].license;
      // Update chat input placeholder
      document.getElementById('user-input').placeholder = translations[lang].chatInputPlaceholder;
      // Update nav links
      document.getElementById('nav-account').innerText = translations[lang].account;
      document.getElementById('nav-subscription').innerText = translations[lang].subscription;
      document.getElementById('nav-about').innerText = translations[lang].about;
    }

    // Remove last message (for loader)
    function removeLastMessage() {
      const messagesDiv = document.getElementById('chat-messages');
      if (messagesDiv.lastChild) {
        messagesDiv.removeChild(messagesDiv.lastChild);
      }
    }

    async function sendMessage() {
      const input = document.getElementById('user-input');
      const message = input.value;
      if (!message) return;
      // Remove welcome screen if exists
      const welcomeContainer = document.getElementById('welcome-container');
      if (welcomeContainer) {
        welcomeContainer.remove();
      }
      appendMessage('user', message, false);
      input.value = '';
      // Insert loader message as bot message
      const loaderHTML = `<div class="loader-container"><div class="loader"></div><div class="loader-text">${translations[currentLang].loaderMessage}</div></div>`;
      appendMessage('bot', loaderHTML, true);

      // Post the query (and language) to the backend
      const response = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: message, lang: currentLang })
      });
      const data = await response.json();
      const htmlResponse = marked.parse(data.response);
      removeLastMessage();
      appendMessage('bot', htmlResponse, true);
    }

    function appendMessage(role, text, isHTML) {
      const messagesDiv = document.getElementById('chat-messages');
      const messageDiv = document.createElement('div');
      messageDiv.classList.add('message');
      const prefix = role === 'user' ? translations[currentLang].you : translations[currentLang].bot;
      if (isHTML) {
        messageDiv.innerHTML = `<strong class="${role}">${prefix}:</strong><br/>${text}`;
      } else {
        messageDiv.innerHTML = `<strong class="${role}">${prefix}:</strong> ${text}`;
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

      // When a language option is selected from the dropdown
      document.querySelectorAll('.dropdown-menu li').forEach(item => {
        item.addEventListener('click', function(event) {
          event.stopPropagation();
          const selectedLang = this.getAttribute('data-lang');
          dropdownBtn.innerHTML = selectedLang + " &#x25BC;";
          dropdownMenu.style.display = 'none';
          updateLanguage(selectedLang);
        });
      });

      // Close the dropdown if clicking outside
      document.addEventListener('click', function() {
        dropdownMenu.style.display = 'none';
      });
    });

    // Modal Language Selection Functionality
    document.addEventListener('DOMContentLoaded', function() {
      const modal = document.getElementById('language-modal');
      const modalButtons = modal.querySelectorAll('button');
      // When any modal button is clicked:
      modalButtons.forEach(btn => {
        btn.addEventListener('click', function() {
          const lang = this.getAttribute('data-lang');
          updateLanguage(lang);
          // Also update the dropdown button text
          document.querySelector('.dropdown-btn').innerHTML = lang + " &#x25BC;";
          // Hide the modal
          modal.style.display = 'none';
        });
      });
    });

    // Replay thinking GIF without hard flicker
    const GIF_DURATION = 800;
    setInterval(() => {
      const tooltip = document.getElementById('tooltip');
      const newImg = new Image();
      const newSrc = `/static/img/cloud.gif?t=${Date.now()}`;
      newImg.onload = () => {
        tooltip.style.background = `url('${newSrc}') repeat`;
        tooltip.style.backgroundSize = '100% 100%';
      };
      newImg.src = newSrc;
    }, GIF_DURATION);
  </script>
</body>
</html>
"""

# Mount static files (make sure the "static" folder exists and contains your images)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Get statics template route
@app.get("/", response_class=HTMLResponse)
async def get_home():
    return HTML_CONTENT

# Chat route
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

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000)) # Default to 8000 if PORT isn't set
    print(f"\n🩺 Starting Medical Chatbot FastAPI server on port {port}...\n")
    uvicorn.run(app, host="0.0.0.0", port=port)
