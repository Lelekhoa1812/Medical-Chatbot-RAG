# Medical Chatbot Platform

Welcome to the RAG-based Medical Chatbot project! This project leverages cutting‑edge technologies such as Retrieval-Augmented Generation (RAG), Gemini Flash 2.5 (Backbone reasoning LLM), MedGemma (VLM), Llama, Mistral, and GPT varieties to deliver an intelligent medical chatbot . It uses a custom medical dataset (over 256,916 QA curation), diagnosis retrieval agent (over 4,962 symptom scenarios), and employs FAISS for efficient similarity search. The server runs on FastAPI and dynamically renders HTML using MarkdownJS.

## Access Now
- [Web App](https://medical-chatbot-henna.vercel.app/)
- [HF Space](https://huggingface.co/spaces/BinKhoaLe1812/Medical-Chatbot)

2. **Frontend**:  
   “Frontend (UI), built with Node.js and incorporating Vite, Axios, is deployed on Vercel.”

### Frontend Capabilities
- Multilingual UX (EN/VI/ZH), light/dark themes, responsive design
- Two modes: Chat and Search (web‑augmented answers)
- Inline markdown, tables, lists; image upload and preview for medical imaging
- Source transparency: clickable magnifier icons linking to cited sources
- Polished UX with notifications, accessibility, and mobile-first layout

### Backend Capabilities
- RAG over curated medical KB (FAISS) with deduplication and decay scoring
- Query‑Focused Web Search Agent:
  - Aggregates up to 10 results via DuckDuckGo and expands select same‑domain links
  - Crawls large pages, chunks with overlap, and uses Llama to extract only query‑relevant facts per chunk
  - Merges distilled snippets per source and builds URL mappings for citations
- Summarization Everywhere (Llama):
  - Conversation/context compression, response chunking, and document synthesis
  - Text cleaning and key‑phrase priming to minimize token footprint
- Vision (VLM) Pipeline: medical image description and analysis
- Safety Guardrails: Llama Guard validates user input and model output
- Citations: `<#ID>` tagging in model output → resolved to `<URL>` server‑side → rendered as source icons in UI

## 🧠 Technical Highlights
- FastAPI modular services (api, models, memory, search, utils)
- FAISS vector retrieval; MongoDB for KB and index artifacts
- NVIDIA Llama for concise summarization and query relevance extraction
- Llama Guard for policy compliance on inputs and outputs
- Robust logging, retries, and graceful degradation in network‑bound paths

## 📸 Screenshots

### Chatbot Console Example
<img src="imgsrc/chatbot_console1.png" alt="Chatbot Medical Answer Example" style="width: 80%; max-width: 1000px;">

### Chatbot with Answer
<img src="imgsrc/chat-en1.png" alt="Chatbot UI 1" style="width: 80%; max-width: 1000px;">
<img src="imgsrc/chat-en2.png" alt="Chatbot UI 2" style="width: 80%; max-width: 1000px;">
<img src="imgsrc/chat-en3.png" alt="Chatbot UI 3" style="width: 80%; max-width: 1000px;">
<img src="imgsrc/chat-en4.png" alt="Chatbot UI 4" style="width: 80%; max-width: 1000px;">

## 📊 Memory & Retrieval
- Per‑user LRU short‑term memory with FAISS; up to 30 recent chunks
- Topic‑level chunking and summarization (Llama) for compact context
- Time‑decay and usage signals to prioritize what matters now

### 🧠 Knowledge Embedding & Diagnosis Retrieval
- SentenceTransformer embeddings for queries and KB entries (`all‑MiniLM‑L6‑v2`)
- Symptom diagnosis agent leveraging structured prognosis data
- MongoDB vector cache and FAISS for fast similarity search

## 🛡️ Trust & Safety
- Llama Guard validates user requests and assistant responses
- Unsafe requests are blocked; unsafe responses are redacted with safe fallback
- Web‑derived claims carry explicit citations for auditability

## 🔗 Deployment

|  **Component** | **Hosting**           | **URL**                                           |
|----------------|-----------------------|---------------------------------------------------|
| Backend        | Hugging Face Spaces   | `https://binkhoale1812-medical-chatbot.hf.space/` |
| Frontend       | Vercel                | `https://medical-chatbot.vercel.app`              |
| Database       | MongoDB Atlas         | 2 DBs                                             |


## 🧩 Flowchart
<img src="med-flow/flow.png" alt="Flowchart" style="width: 80%; max-width: 1000px;">

## 📝 License
This project is licensed under the [Apache 2.0 License](LICENSE).

---
Author: (Liam) Dang Khoa Le
Latest Update: 08/10/2025
