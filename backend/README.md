---
title: Medical Chatbot
emoji: 🤖🩺
colorFrom: blue
colorTo: purple
sdk: docker
sdk_version: latest
pinned: false
license: apache-2.0
short_description: MedicalChatbot, FAISS, Gemini, MongoDB vDB, LRU
---

# Medical Chatbot Backend

## At-a-glance
Production-grade medical RAG with safe-guarding, query-focused web search, memory compression, and precise source citations.

## Key Features

### 🔍 Web Search Agent (Query-Focused)
- Aggregates up to 10+ sources (DuckDuckGo) and expands a few same‑domain links per result
- Crawls large pages, chunks with overlap, and uses Llama to extract only query‑relevant facts per chunk
- Merges per‑source distilled snippets; maps citations to URLs for the UI

### 🧠 Memory + Retrieval
- Conversation memory compressed via Llama (concise, no fluff)
- FAISS retrieval with dedup; optional reranker on guideline‑like texts
- Context builder blends recent memory + RAG into a tight summary for the main LLM

### 🛡️ Safety Guard
- Llama Guard (meta/llama-guard-4-12b) validates both user input and model output
- Unsafe requests are blocked; unsafe answers are replaced with a safe fallback

### 📝 Summarization
- Text cleaning + key‑phrase priming
- Query‑focused summarizer returns ONLY relevant medical facts
- Used across memory, search chunking, and document synthesis

## Usage

### Running the Application
```bash
# Using main entry point
python main.py

# Or directly
python api/app.py
```

### Environment Variables
- `NVIDIA_URI` - API key for Llama model
- `FlashAPI` - Gemini API key
- `MONGO_URI` - MongoDB connection string
- `INDEX_URI` - FAISS index database URI
- `NVIDIA_RERANK_ENDPOINT` (optional) - reranker endpoint

## API Endpoints

### POST `/chat`
Main chat endpoint with search mode support.

**Request Body:**
```json
{
  "query": "User's medical question",
  "lang": "EN",
  "search": true,
  "user_id": "unique_user_id",
  "image_base64": "optional_base64_image",
  "img_desc": "image_description"
}
```

**Response:**
```json
{
  "response": "Medical response with citations <URL>",
  "response_time": "2.34s"
}
```

## Search Mode Features

When `search: true`:
1. Fetch up to 10 results and expand a few intrasite links per result
2. Chunk each page; Llama extracts only query‑relevant facts per chunk
3. Combine per‑doc summaries; instruct main LLM to cite with `<#ID>`
4. Backend replaces `<#ID>` with `<URL>`; frontend renders magnifier icons

## Summarization Features

All summarization tasks use Llama model:
- **get_contextual_chunks**: Summarizes conversation history and RAG chunks
- **chunk_response**: Chunks and summarizes bot responses
- **summarize_documents**: Summarizes web search results

### Text Processing Pipeline
1. **Clean Text**: Remove conversational elements and normalize
2. **Extract Key Phrases**: Identify medical terms and concepts
3. **Summarize**: Create concise, focused summaries
4. **Validate**: Ensure quality and relevance

## Folders (overview)
- `api/` FastAPI app, routes, chatbot orchestration
- `models/` Llama, Llama Guard, summarizer, warmup
- `memory/` Memory manager and FAISS interfaces
- `search/` Web search + extraction + chunking
- `utils/` Translation, VLM, data utilities

## Dependencies

See `requirements.txt` for complete list. Key additions:
- `requests` - Web search functionality
- `beautifulsoup4` - HTML content extraction
- API integration for Llama and Llama Guard