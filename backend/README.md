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

## Project Structure

The backend is organized into logical modules for better maintainability:

### 📁 **api/**
- **app.py** - Main FastAPI application with endpoints
- **__init__.py** - API package initialization

### 📁 **models/**
- **llama.py** - NVIDIA Llama model integration for search processing
- **summarizer.py** - Text summarization using NVIDIA Llama
- **download_model.py** - Model download utilities
- **warmup.py** - Model warmup scripts

### 📁 **memory/**
- **memory_updated.py** - Enhanced memory management with NVIDIA Llama summarization
- **memory.py** - Legacy memory implementation

### 📁 **search/**
- **search.py** - Web search and content extraction functionality

### 📁 **utils/**
- **translation.py** - Multi-language translation utilities
- **vlm.py** - Vision Language Model for medical image processing
- **diagnosis.py** - Symptom-based diagnosis utilities
- **connect_mongo.py** - MongoDB connection utilities
- **clear_mongo.py** - Database cleanup utilities
- **migrate.py** - Database migration scripts

## Key Features

### 🔍 **Search Integration**
- Web search with up to 10 resources
- NVIDIA Llama model for keyword generation and document summarization
- Citation system with URL mapping
- Smart content filtering and validation

### 🧠 **Enhanced Memory Management**
- NVIDIA Llama-powered summarization for all text processing
- Optimized chunking and context retrieval
- Smart deduplication and merging
- Conversation continuity with concise summaries

### 📝 **Summarization System**
- **Text Cleaning**: Removes conversational fillers and normalizes text
- **Key Phrase Extraction**: Identifies medical terms and concepts
- **Concise Summaries**: Preserves key ideas without fluff
- **NVIDIA Llama Integration**: All summarization uses NVIDIA model instead of Gemini

## Usage

### Running the Application
```bash
# Using main entry point
python main.py

# Or directly
python api/app.py
```

### Environment Variables
- `NVIDIA_URI` - NVIDIA API key for Llama model
- `FlashAPI` - Gemini API key
- `MONGO_URI` - MongoDB connection string
- `INDEX_URI` - FAISS index database URI

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
1. **Web Search**: Fetches up to 10 relevant medical resources
2. **Llama Processing**: Generates keywords and summarizes content
3. **Citation System**: Replaces `<#ID>` tags with actual URLs
4. **UI Integration**: Frontend displays magnifier icons for source links

## Summarization Features

All summarization tasks use NVIDIA Llama model:
- **get_contextual_chunks**: Summarizes conversation history and RAG chunks
- **chunk_response**: Chunks and summarizes bot responses
- **summarize_documents**: Summarizes web search results

### Text Processing Pipeline
1. **Clean Text**: Remove conversational elements and normalize
2. **Extract Key Phrases**: Identify medical terms and concepts
3. **Summarize**: Create concise, focused summaries
4. **Validate**: Ensure quality and relevance

## Dependencies

See `requirements.txt` for complete list. Key additions:
- `requests` - Web search functionality
- `beautifulsoup4` - HTML content extraction
- NVIDIA API integration for Llama model