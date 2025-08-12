# 🔄 Hybrid Context Retrieval System

## Overview

The Medical Chatbot now implements a **hybrid context retrieval system** that combines **semantic search (RAG)** with **recent chat history** to provide more intelligent and contextually aware responses. This addresses the limitation of pure RAG systems that can miss conversational context like "What's the diagnosis again?" or "Can you clarify that?"

## 🏗️ Architecture

### Before (Pure RAG)
```
User Query → Semantic Search → FAISS Index → Relevant Chunks → LLM Response
```

### After (Hybrid Approach)
```
User Query → Hybrid Context Retrieval → Intelligent Context Selection → LLM Response
                ↓
        ┌─────────────────┬─────────────────┐
        │   RAG Search    │ Recent History  │
        │ (Semantic)      │ (Conversational)│
        └─────────────────┴─────────────────┘
                ↓
        Gemini Flash Lite Contextual Analysis
                ↓
        Selected Relevant Context
```

## 🔧 Key Components

### 1. Memory Manager (`memory.py`)

#### New Method: `get_recent_chat_history()`
```python
def get_recent_chat_history(self, user_id: str, num_turns: int = 3) -> List[Dict]:
    """
    Get the most recent chat history with both user questions and bot responses.
    Returns: [{"user": "question", "bot": "response", "timestamp": time}, ...]
    """
```

**Features:**
- Stores last 3 conversations by default
- Maintains chronological order
- Includes both user questions and bot responses
- Accessible for conversational continuity

#### Existing Method: `get_relevant_chunks()`
- Semantic search using FAISS
- Cosine similarity-based retrieval
- Smart deduplication and scoring

### 2. Chatbot Class (`app.py`)

#### New Method: `_get_contextual_chunks()`
```python
def _get_contextual_chunks(self, user_id: str, current_query: str, 
                          recent_history: List[Dict], rag_chunks: List[str], 
                          lang: str) -> List[str]:
```

**Purpose:**
- Analyzes current query against available context
- Uses Gemini Flash Lite for intelligent context selection
- Combines RAG results with recent history
- Ensures conversational continuity

## 🚀 How It Works

### Step 1: Context Collection
```python
# Get both types of context
rag_context = memory.get_relevant_chunks(user_id, user_query, top_k=3)
recent_history = memory.get_recent_chat_history(user_id, num_turns=3)
```

### Step 2: Contextual Analysis
The system sends both context sources to Gemini Flash Lite with this prompt:

```
You are a medical assistant analyzing conversation context to provide relevant information.

Current user query: "{current_query}"

Available context information:
{recent_history + rag_chunks}

Task: Analyze the current query and determine which pieces of context are most relevant.

Consider:
1. Is the user asking for clarification about something mentioned before?
2. Is the user referencing a previous diagnosis or recommendation?
3. Are there any follow-up questions that build on previous responses?
4. Which chunks provide the most relevant medical information for the current query?

Output: Return only the most relevant context chunks that should be included in the response.
```

### Step 3: Intelligent Selection
Gemini Flash Lite analyzes the query and selects relevant context from:
- **Recent conversations** (for continuity)
- **Semantic chunks** (for topic relevance)
- **Combined insights** (for comprehensive understanding)

### Step 4: Context Integration
Selected context is integrated into the main LLM prompt, ensuring the response is both:
- **Semantically relevant** (from RAG)
- **Conversationally continuous** (from recent history)

## 📊 Benefits

### 1. **Conversational Continuity**
- Handles follow-up questions naturally
- Maintains context across multiple exchanges
- Understands references to previous responses

### 2. **Intelligent Context Selection**
- No more irrelevant context injection
- Gemini Flash Lite decides what's truly relevant
- Balances semantic relevance with conversational flow

### 3. **Fallback Mechanisms**
- If contextual analysis fails, falls back to RAG
- If RAG fails, falls back to recent history
- Ensures system reliability

### 4. **Performance Optimization**
- Uses lightweight Gemini Flash Lite for context analysis
- Maintains existing RAG performance
- Minimal additional latency

## 🧪 Example Scenarios

### Scenario 1: Follow-up Question
```
User: "I have a headache"
Bot: "This could be a tension headache. Try rest and hydration."

User: "What medication should I take?"
Bot: "For tension headaches, try acetaminophen or ibuprofen..."

User: "Can you clarify the dosage again?"
Bot: "For ibuprofen: 200-400mg every 4-6 hours, max 1200mg/day..."
```
**Result:** System retrieves ibuprofen dosage from recent conversation, not just semantic search.

### Scenario 2: Reference to Previous Diagnosis
```
User: "What was the diagnosis you mentioned?"
Bot: "I previously diagnosed this as a tension headache based on your symptoms..."
```
**Result:** System understands the reference and retrieves previous diagnosis.

### Scenario 3: Clarification Request
```
User: "I didn't understand the part about prevention"
Bot: "Let me clarify the prevention steps I mentioned earlier..."
```
**Result:** System identifies the clarification request and retrieves relevant previous response.

## ⚙️ Configuration

### Environment Variables
```bash
FlashAPI=your_gemini_api_key  # For both main LLM and contextual analysis
```

### Memory Settings
```python
memory = MemoryManager(
    max_users=1000,           # Maximum users in memory
    history_per_user=10,      # Chat history per user
    max_chunks=30             # Maximum chunks per user
)
```

### Context Parameters
```python
# Recent history retrieval
recent_history = memory.get_recent_chat_history(user_id, num_turns=3)

# RAG retrieval
rag_chunks = memory.get_relevant_chunks(user_id, query, top_k=3, min_sim=0.30)

# Contextual analysis
contextual_chunks = self._get_contextual_chunks(
    user_id, current_query, recent_history, rag_chunks, lang
)
```

## 🔍 Monitoring & Debugging

### Logging
The system provides comprehensive logging:
```python
logger.info(f"[Contextual] Gemini selected {len(relevant_chunks)} relevant chunks")
logger.warning(f"[Contextual] Gemini contextual analysis failed: {e}")
```

### Performance Metrics
- Context retrieval time
- Number of relevant chunks selected
- Fallback usage statistics

## 🚨 Error Handling

### Fallback Strategy
1. **Primary:** Gemini Flash Lite contextual analysis
2. **Secondary:** RAG semantic search
3. **Tertiary:** Recent chat history
4. **Final:** No context (minimal response)

### Error Scenarios
- Gemini API failure → Fall back to RAG
- RAG failure → Fall back to recent history
- Memory corruption → Reset user session

## 🔮 Future Enhancements

### 1. **Context Scoring**
- Implement confidence scores for context relevance
- Weight recent history vs. semantic chunks
- Dynamic threshold adjustment

### 2. **Multi-turn Context**
- Extend beyond 3 recent turns
- Implement conversation threading
- Handle multiple conversation topics

### 3. **Context Compression**
- Summarize long conversation histories
- Implement context pruning strategies
- Optimize memory usage

### 4. **Language-specific Context**
- Enhance context analysis for different languages
- Implement language-aware context selection
- Cultural context considerations

## 📝 Testing

Run the test script to verify functionality:
```bash
cd Medical-Chatbot
python test_hybrid_context.py
```

This will demonstrate:
- Memory management
- Context retrieval
- Hybrid approach simulation
- Expected behavior examples

## 🎯 Summary

The hybrid context retrieval system transforms the Medical Chatbot from a simple RAG system to an intelligent, contextually aware assistant that:

✅ **Maintains conversational continuity**  
✅ **Provides semantically relevant responses**  
✅ **Handles follow-up questions naturally**  
✅ **Uses AI for intelligent context selection**  
✅ **Maintains performance and reliability**  

This system addresses real-world conversational patterns that pure RAG systems miss, making the chatbot more human-like and useful in extended medical consultations.
