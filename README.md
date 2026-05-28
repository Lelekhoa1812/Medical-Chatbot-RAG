# Medical Chatbot Platform

A production-oriented medical chatbot platform built around Retrieval-Augmented Generation (RAG), Azure AI Foundry-hosted models, FastAPI, FAISS-backed retrieval, medical-image support, multilingual UX, and source-aware answer rendering.

This repository delivers a browser-based medical assistant that combines:
- a curated medical QA knowledge base,
- diagnosis-oriented symptom retrieval,
- optional query-focused web search,
- vision-assisted image workflows,
- safety/validation layers, and
- persistent client-side session UX.

> **Important**
> This software is an informational medical assistant, not a substitute for licensed clinical judgment, diagnosis, or emergency care.

## Access
- **Frontend (Vercel)**: https://medical-chatbot-henna.vercel.app/
- **Backend (Hugging Face Space endpoint used by frontend in production)**: https://khoaliamle-Medical-Chatbot.hf.space
- **Hugging Face Space page**: https://huggingface.co/spaces/BinKhoaLe1812/Medical-Chatbot

---

## System Overview

At a high level, the platform consists of:

1. **Frontend web client**
   - Static/browser-based UI
   - Multilingual strings (EN / VI / ZH)
   - Theme toggle (light/dark)
   - Chat persistence in browser storage
   - Search, upload, and video interaction modes
   - Markdown rendering and citation UX

2. **Backend API (FastAPI)**
   - Chat orchestration
   - Retrieval over medical knowledge sources
   - Query-focused web search and summarization
   - Safety validation and response filtering
   - Citation normalization for frontend rendering
   - Async/pending request restoration support

3. **Knowledge + retrieval layer**
   - FAISS similarity search
   - curated medical QA corpus
   - diagnosis/symptom retrieval data
   - MongoDB-backed storage/index artifacts

4. **Model layer**
   - Azure AI Foundry-hosted LLM/SLM endpoints
   - stronger model for primary reasoning tasks
   - lightweight model for translation, validation, reranking-style support, and guard workflows

---

## Core Technical Capabilities

### RAG Medical Answering
The application uses retrieval-augmented generation to ground responses in a curated medical corpus before composing answers. The retrieval stack is designed to reduce hallucination and prioritize medically relevant context.

### Query-Focused Web Search
When search mode is enabled, the backend can augment local retrieval with external web information. The documented backend structure indicates that the search pipeline:
- aggregates result pages,
- crawls and chunks long content,
- summarizes only query-relevant facts,
- merges distilled snippets per source,
- and preserves URL mappings for source citations.

### Medical Image Workflow
The UI supports upload-driven workflows for medical image analysis/description. The README cannot assert exact model internals beyond repository evidence, but the platform clearly exposes an image-enabled interaction path in both product copy and frontend behavior.

### Source Transparency
Model outputs may contain citation placeholders which are normalized server-side and rendered client-side as source-linked UI elements. This allows users to inspect supporting sources rather than receiving unsupported free text alone.

### Session Persistence and Recovery
The frontend persists conversation-related state in browser storage, including:
- chat history,
- pending requests,
- session-scoped conversation identifiers,
- theme preference,
- user identifier,
- and per-conversation stored video results.

This enables recovery after refreshes and restoration of completed async responses from previous sessions.

---

## Frontend Architecture and Behavior

Repository evidence from `static/src/script.js` shows the frontend is more sophisticated than a simple chat box.

### Runtime Endpoint Selection
The browser chooses the API base dynamically:
- **Local development**: `http://0.0.0.0:8000`
- **Production**: `https://khoaliamle-Medical-Chatbot.hf.space`

This means local frontend testing expects a backend listening on port `8000`.

### Language Support
The frontend currently includes built-in translations for:
- English (`EN`)
- Vietnamese (`VI`)
- Chinese (`ZH`)

Translated UI elements include:
- header/title,
- tooltips,
- welcome text,
- author/license labels,
- navigation items,
- input placeholder,
- message role labels,
- loader copy.

### Theme Support
A user-selectable light/dark theme is implemented in the browser and persisted to `localStorage`.

### Input Modes
The UI exposes independent mode toggles for:
- `search` mode,
- `upload` mode,
- `video` mode.

These are tracked independently in client state rather than being treated as one mutually exclusive selector.

### Markdown Rendering
Frontend responses are rendered with `marked` configured for:
- GitHub-flavored markdown,
- line breaks,
- disabled header IDs,
- disabled mangling.

This allows backend responses to include rich formatting such as:
- headings,
- lists,
- emphasis,
- tables,
- code-like blocks where needed.

### Chat Persistence
The frontend stores chat history under browser storage keys including:
- `medical_chatbot_history`
- `medical_chatbot_pending_requests`
- session conversation identifier `chat_conversation_id`
- user identifier `chat_user_id`

Current chat history logic:
- serializes rendered message content,
- retains only the most recent 100 messages,
- restores messages on reload,
- can clear history and restore the welcome screen.

### Pending Request Recovery
The frontend tracks pending requests and periodically polls the backend every 30 seconds for completion. Completed responses are restored into the UI after refresh/session interruption.

Operational details visible in the client logic:
- at most 5 pending requests are retained,
- stale pending requests older than 1 hour are cleaned up,
- successful recoveries trigger a notification,
- failed pending requests are removed from storage.

### Conversation Scoping
The UI generates a session-scoped conversation ID using `crypto.randomUUID()` when available. This is used to isolate conversation-specific persisted artifacts such as stored videos.

### Video Mode and Persistence
The frontend supports rendering related medical videos in a dedicated container/card layout. Implemented behaviors include:
- duplicate render suppression by URL signature,
- per-conversation persistence of returned video lists,
- embedded playback for YouTube URLs,
- external link-out for source viewing,
- single-expanded-player interaction.

---

## Backend Architecture

Repository documentation under `backend/api/README.md` and `backend/README.md` indicates a modular FastAPI backend.

### Main Backend Modules

#### `backend/api/config.py`
Responsible for configuration and operational settings such as:
- environment variable validation,
- logging configuration,
- system/resource settings,
- memory optimization,
- CORS configuration.

#### `backend/api/database.py`
Handles data and retrieval infrastructure, including:
- MongoDB connection management,
- FAISS index lazy loading,
- SentenceTransformer initialization,
- symptom vector management,
- GridFS integration.

#### `backend/api/retrieval.py`
Implements retrieval behavior such as:
- medical information retrieval,
- symptom-based diagnosis retrieval,
- similarity scoring,
- deduplication and matching.

#### `backend/api/chatbot.py`
Core orchestration layer for the chatbot flow, including:
- RAG orchestration,
- search mode integration,
- citation processing,
- memory integration.

#### `backend/api/routes.py`
Defines HTTP endpoints such as:
- `POST /chat`
- `GET /health`
- `GET /`

#### `backend/api/app.py`
The FastAPI application entrypoint that wires:
- app initialization,
- middleware,
- database startup,
- route registration,
- server startup behavior.

### Supporting Backend Areas
Backend repository docs also identify these folders:
- `backend/models/` — model clients, guardrails, summarization, warmup
- `backend/memory/` — memory manager and FAISS interfaces
- `backend/search/` — web search, extraction, chunking
- `backend/utils/` — translation, VLM, and utility code

---

## AI / Model Configuration

The current repository-level documentation should reflect **Azure AI Foundry** as the active provider abstraction for language-model access.

### Required Environment Variables
Configure Azure AI access through environment variables:

```env
FOUNDRY_API_KEY=your_azure_ai_key
FOUNDRY_ENDPOINT=https://your-resource.openai.azure.com
LLM_MODEL=gpt-5.4
SLM_MODEL=gpt-5-nano
```

### Variable Semantics
- `FOUNDRY_API_KEY` — Azure AI Foundry / Azure OpenAI-compatible API key
- `FOUNDRY_ENDPOINT` — Azure AI Foundry-compatible endpoint base URL
- `LLM_MODEL` — primary stronger model for harder reasoning/generation tasks
- `SLM_MODEL` — smaller/faster model for lightweight tasks such as validation, translation, filtering, and auxiliary processing

### Generation Parameter Policy
For all LLM services, use provider/model defaults unless a runtime-specific requirement is introduced later. Do not define or send fixed generation-control parameters such as:
- `temperature`
- `max_tokens`
- `max_output_tokens`

This keeps configuration aligned with the request to remove initially defined sampling/output limits across services.

### Recommended Additional Infrastructure Variables
The backend docs also indicate non-model infrastructure dependencies. In practice, deployments commonly also require values such as:

```env
MONGO_URI=your_mongodb_connection_string
INDEX_URI=your_faiss_or_index_artifact_location
```

Use your actual deployment/runtime configuration for these values based on how the backend is provisioned.

---

## API Contract

Repository documentation shows the main application endpoint as:

### `POST /chat`
Primary chat endpoint for standard and search-augmented medical interaction.

#### Example Request
```json
{
  "query": "What are common causes of persistent dry cough?",
  "lang": "EN",
  "search": true,
  "user_id": "unique_user_id",
  "image_base64": "optional_base64_image",
  "img_desc": "optional_image_description"
}
```

#### Example Response
```json
{
  "response": "Medical response with citations <URL>",
  "response_time": "2.34s"
}
```

### Other Documented Endpoints
- `GET /health` — service health/status check
- `GET /` — root endpoint

### Pending Request Recovery Endpoint
Frontend evidence shows the client polling:
- `GET /check-request/{requestId}`

This endpoint is used to restore completed async responses after refreshes or interrupted sessions.

---

## Retrieval, Memory, and Ranking

### Knowledge Sources
The project documentation references:
- a curated medical QA dataset with over **256,916** QA items,
- a diagnosis retrieval agent with over **4,962** symptom scenarios.

These figures are retained here because they are explicitly described in the existing README, but should be updated if the underlying corpora have changed.

### Embeddings
Current docs reference SentenceTransformer-based embeddings, specifically:
- `all-MiniLM-L6-v2`

for encoding queries and knowledge-base entries.

### FAISS Retrieval
FAISS is used for similarity search and retrieval efficiency. The backend docs also indicate lazy loading and artifact/index handling through the database layer.

### Conversation Memory
Current project docs describe:
- short-term per-user memory,
- compacted or summarized history,
- prioritization using recency/time-decay and usage signals,
- context compression before final generation.

### Summarization Pipeline
Repository documentation indicates summarization is used across:
- conversation/context compression,
- search chunk extraction,
- response chunking,
- document synthesis.

---

## Safety and Trust Controls

The repository documents explicit safety behavior.

### Guard Validation
The backend includes guard logic that validates:
- user requests before response generation,
- model answers before they are returned.

### Safety Outcomes
Documented behavior includes:
- blocking unsafe requests,
- redacting or replacing unsafe answers,
- returning safer fallback responses where necessary.

### Citation-Aware Responses
Web-derived claims are intended to remain auditable through explicit source mapping and frontend citation rendering.

---

## Local Development

## Prerequisites
- Python 3.x for backend runtime
- MongoDB access if using the full retrieval stack
- FAISS/index artifacts available to the backend
- Azure AI Foundry credentials

## Backend Run Options
Based on repository docs:

```bash
# From the backend entrypoint
python main.py

# Or directly
python api/app.py
```

If you run the backend locally, ensure it is reachable at:

```text
http://0.0.0.0:8000
```

because the frontend script uses that URL automatically when the browser hostname is `localhost`.

## Frontend Local Behavior
The frontend is a static app whose script dynamically targets the local backend when served from localhost. Make sure any local static hosting setup preserves that behavior.

---

## Deployment Topology

| Component | Hosting | URL |
|---|---|---|
| Frontend | Vercel | `https://medical-chatbot-henna.vercel.app/` |
| Backend API | Hugging Face Spaces | `https://khoaliamle-Medical-Chatbot.hf.space` |
| HF Space page | Hugging Face | `https://huggingface.co/spaces/BinKhoaLe1812/Medical-Chatbot` |
| Database | MongoDB Atlas | deployment-specific |

> Note: older documentation in this repository references slightly different deployment URLs and older provider names. This README has been normalized to match the currently evidenced frontend production endpoint behavior and Azure AI environment model.

---

## Client-Side Storage Model

The frontend currently uses browser storage for several UX and recovery features.

### `localStorage`
Used for:
- theme preference,
- chat history,
- pending requests,
- persistent user ID,
- conversation-scoped stored videos.

### `sessionStorage`
Used for:
- current `chat_conversation_id`

This means:
- refreshing a tab can preserve significant state,
- opening a new browser session can create a new conversation scope,
- some artifacts persist across sessions while conversation ID itself is session-scoped.

---

## User Experience Features

### Welcome Screen
On empty history, the chat area displays a welcome container with:
- logo,
- welcome text,
- acknowledgement,
- author attribution,
- license link.

### Notifications
The frontend includes status notifications for events such as:
- loading previous messages,
- restoring previous responses,
- clearing history,
- error conditions.

### Submission Protection
The UI includes debounce and submission-state protections to reduce accidental rapid repeat submissions.

---

## Repository Structure

A concise high-level view from repository evidence:

```text
backend/
  api/
  memory/
  models/
  search/
  utils/
static/
  src/
README.md
```

### Important Documented Surfaces
- `README.md` — project-level architecture and deployment notes
- `backend/README.md` — backend capabilities and runtime notes
- `backend/api/README.md` — API module responsibilities
- `static/src/script.js` — concrete frontend behavior and persistence logic

---

## Known Documentation Corrections Applied

This README updates several outdated or inconsistent areas from the previous version:
- normalizes model-provider documentation toward **Azure AI Foundry**,
- removes mixed or conflicting top-level wording about frontend stack details not evidenced here,
- aligns production backend URL with the frontend runtime configuration,
- documents browser-side pending request recovery and conversation scoping,
- adds technical detail for multilingual, persistence, and video features,
- clarifies backend module responsibilities using repository-provided docs,
- documents removal of explicitly defined generation parameters such as temperature and token caps.

---

## Screenshots

### Chatbot Console Example
<img src="imgsrc/chatbot_console1.png" alt="Chatbot Medical Answer Example" style="width: 80%; max-width: 1000px;">

### Chatbot with Answer
<img src="imgsrc/src.png" alt="Chatbot UI 1" style="width: 80%; max-width: 1000px;">

### Flowchart
<img src="med-flow/flow.png" alt="Flowchart" style="width: 80%; max-width: 1000px;">

---

## License
This project is licensed under the [Apache 2.0 License](LICENSE).

---
Author: (Liam) Dang Khoa Le  
README technical refresh: 2026-05-28
