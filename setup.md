## **AutoGen-Based Medical Chatbot with RAG**

This guide will help you deploy a **Medical Chatbot** using **AutoGen**, **RAG (Retrieval-Augmented Generation)**, and **OpenAI API** with a **custom Hugging Face medical dataset**.

---

## **Installation**
### **Step 1: Set Up Conda Environment**
Create a new Conda environment with Python 3.11:   
```bash
conda create -n chatbot_env python=3.11 -y
conda activate chatbot_env
```
**Notice:** Ensure your machine has required Anaconda or Miniconda version installed.  

---

### **Step 2: Install Dependencies**
Ensure your environment has the necessary libraries:   
```bash
pip install -U "autogen-agentchat" "autogenstudio" \
                "faiss-cpu" "chromadb" "datasets" "sentence-transformers" "python-dotenv" "google-genai" "huggingface_hub" "pymongo" \
                "uvicorn" "fastapi"
```
Or install all:  
```bash
pip install -U requirements.txt
```

- `faiss-cpu`: Efficient vector search for RAG.
- `chromadb`: Local vector database for retrieval (Not using now).
- `datasets`: Loading and processing custom data.
- `sentence-transformers`: Converting text into embeddings.
- `python-dotenv`: Managing API keys securely from `.env`.
- `pymongo`: Utilization of MongoDB and gridfs (for compressed FAISS data saving).

Installation of server and router on FastAPI for a client-server interface (optional):  
```bash
pip install fastapi uvicorn requests python-multipart
```

---

## **Configuration**
### **Step 3: Set Up a genAI API Key**
To integrate RAG and answer intelligently, you'll need an **OpenAI API key**:  
```bash
export OPENAI_API_KEY="your_openai_api_key"
```
Alternatively using Flash GeminiAPI for lower/free-tier cost:
```bash
export Flash_API="your_gemini_api_key"
```

Alternatively, store it in a `.env` file:
```bash
echo 'OPENAI_API_KEY="your_openai_api_key"' > .env
```
Or:
```bash
echo 'Flash_API="your_gemini_api_key"' > .env
```

### **Optional: Saving embedded vector database to MongoDB**
Store MongoDB cluster API key into `.env` file. You need 2 MongoDB clusters (you can store in 1 individually, however, notice that it could consume high storage availability and wouldn't be ideal for free tier).

1. First cluster (db) storing your QA entries data:
```bash
echo 'MONGO_URI=your_mongodb1_api_key' > .env
```

2. Second cluster (db) storing your embedded FAISS vector index:
```bash
echo 'INDEX_URI=your_mongodb2_api_key' > .env
```

---

## **Project Structure**
/Medical-Chatbot         # Backend
  ├── app.py             # main server
  ├── download_model.py  # model loader
  ├── clear_mongo.py     # debugs
  ├── connect_mongo.py
  ├── migrate.py 
  ├── requirements.txt   # module installation
  ├── Dockerfile         # configs and build
  ├── .env
  ├── .huggingface.yml
/static                  # Frontend
  ├── dist               # built-in package by vite
  ├── node_modules
  ├── index.html         # interface level
  ├── script.js
  ├── styles.css
  ├── img
  ├── vercel.json        # build configs
  ├── vite.config.js
  ├── package.json
  ├── package-lock.json
├── setup.md
├── README.md
├── Embedding.ipynb