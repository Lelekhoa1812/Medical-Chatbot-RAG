# 🔄 Enhanced Memory System: STM + LTM + Hybrid Context Retrieval

## Overview

The Medical Chatbot now implements an **advanced memory system** with **Short-Term Memory (STM)** and **Long-Term Memory (LTM)** that intelligently manages conversation context, semantic knowledge, and conversational continuity. This system goes beyond simple RAG to provide truly intelligent, contextually aware responses that remember and build upon previous interactions.

## 🏗️ Architecture

### Memory Hierarchy
```
User Query → Enhanced Memory System → Intelligent Context Selection → LLM Response
                ↓
        ┌─────────────────┬─────────────────┬─────────────────┐
        │   STM (5 items) │   LTM (60 items)│   RAG Search    │
        │ (Recent Summaries)│ (Semantic Store)│ (Knowledge Base)│
        └─────────────────┴─────────────────┴─────────────────┘
                ↓
        Gemini Flash Lite Contextual Analysis
                ↓
        Summarized Context + Semantic Knowledge
```

### Memory Types

#### 1. **Short-Term Memory (STM)**
- **Capacity:** 5 recent conversation summaries
- **Content:** Chunked and summarized LLM responses with enriched topics
- **Features:** Semantic deduplication, intelligent merging, topic enrichment
- **Purpose:** Maintain conversational continuity and immediate context

#### 2. **Long-Term Memory (LTM)**
- **Capacity:** 60 semantic chunks (~20 conversational rounds)
- **Content:** FAISS-indexed medical knowledge chunks
- **Features:** Semantic similarity search, usage tracking, smart eviction
- **Purpose:** Provide deep medical knowledge and historical context

#### 3. **RAG Knowledge Base**
- **Content:** External medical knowledge and guidelines
- **Features:** Real-time retrieval, semantic matching
- **Purpose:** Supplement with current medical information

## 🔧 Key Components

### 1. Enhanced Memory Manager (`memory.py`)

#### STM Management
```python
def get_recent_chat_history(self, user_id: str, num_turns: int = 5) -> List[Dict]:
    """
    Get the most recent STM summaries (not raw Q/A).
    Returns: [{"user": "", "bot": "Topic: ...\n<summary>", "timestamp": time}, ...]
    """
```

**STM Features:**
- **Capacity:** 5 recent conversation summaries
- **Content:** Chunked and summarized LLM responses with enriched topics
- **Deduplication:** Semantic similarity-based merging (≥0.92 identical, ≥0.75 merge)
- **Topic Enrichment:** Uses user question context to generate detailed topics

#### LTM Management
```python
def get_relevant_chunks(self, user_id: str, query: str, top_k: int = 3, min_sim: float = 0.30) -> List[str]:
    """Return texts of chunks whose cosine similarity ≥ min_sim."""
```

**LTM Features:**
- **Capacity:** 60 semantic chunks (~20 conversational rounds)
- **Indexing:** FAISS-based semantic search
- **Smart Eviction:** Usage-based decay and recency scoring
- **Merging:** Intelligent deduplication and content fusion

#### Enhanced Chunking
```python
def chunk_response(self, response: str, lang: str, question: str = "") -> List[Dict]:
    """
    Enhanced chunking with question context for richer topics.
    Returns: [{"tag": "detailed_topic", "text": "summary"}, ...]
    """
```

**Chunking Features:**
- **Question Context:** Incorporates user's latest question for topic generation
- **Rich Topics:** Detailed topics (10-20 words) capturing context, condition, and action
- **Medical Focus:** Excludes disclaimers, includes exact medication names/doses
- **Semantic Grouping:** Groups by medical topic, symptom, assessment, plan, or instruction

### 2. Intelligent Context Retrieval

#### Contextual Summarization
```python
def get_contextual_chunks(self, user_id: str, current_query: str, lang: str = "EN") -> str:
    """
    Creates a single, coherent summary from STM + LTM + RAG.
    Returns: A single summary string for the main LLM.
    """
```

**Features:**
- **Unified Summary:** Combines STM (5 turns) + LTM (semantic) + RAG (knowledge)
- **Gemini Analysis:** Uses Gemini Flash Lite for intelligent context selection
- **Conversational Flow:** Maintains continuity while providing medical relevance
- **Fallback Strategy:** Graceful degradation if analysis fails

## 🚀 How It Works

### Step 1: Enhanced Memory Processing
```python
# Process new exchange through STM and LTM
chunks = memory.chunk_response(response, lang, question=query)
for chunk in chunks:
    memory._upsert_stm(user_id, chunk, lang)  # STM with dedupe/merge
memory._upsert_ltm(user_id, chunks, lang)     # LTM with semantic storage
```

### Step 2: Context Retrieval
```python
# Get STM summaries (5 recent turns)
recent_history = memory.get_recent_chat_history(user_id, num_turns=5)

# Get LTM semantic chunks
rag_chunks = memory.get_relevant_chunks(user_id, current_query, top_k=3)

# Get external RAG knowledge
external_rag = retrieve_medical_info(current_query)
```

### Step 3: Intelligent Context Summarization
The system sends all context sources to Gemini Flash Lite for unified summarization:

```
You are a medical assistant creating a concise summary of conversation context for continuity.

Current user query: "{current_query}"

Available context information:
Recent conversation history:
{recent_history}

Semantically relevant historical medical information:
{rag_chunks}

Task: Create a brief, coherent summary that captures the key points from the conversation history and relevant medical information that are important for understanding the current query.

Guidelines:
1. Focus on medical symptoms, diagnoses, treatments, or recommendations mentioned
2. Include any patient concerns or questions that are still relevant
3. Highlight any follow-up needs or pending clarifications
4. Keep the summary concise but comprehensive enough for context
5. Maintain conversational flow and continuity

Output: Provide a single, well-structured summary paragraph that can be used as context for the main LLM to provide a coherent response.
```

### Step 4: Unified Context Integration
The single, coherent summary is integrated into the main LLM prompt, providing:
- **Conversational continuity** (from STM summaries)
- **Medical knowledge** (from LTM semantic chunks)
- **Current information** (from external RAG)
- **Unified narrative** (single summary instead of multiple chunks)

## 📊 Benefits

### 1. **Advanced Memory Management**
- **STM:** Maintains 5 recent conversation summaries with intelligent deduplication
- **LTM:** Stores 60 semantic chunks (~20 rounds) with FAISS indexing
- **Smart Merging:** Combines similar content while preserving unique details
- **Topic Enrichment:** Detailed topics using user question context

### 2. **Intelligent Context Summarization**
- **Unified Summary:** Single coherent narrative instead of multiple chunks
- **Gemini Analysis:** AI-powered context selection and summarization
- **Medical Focus:** Prioritizes symptoms, diagnoses, treatments, and recommendations
- **Conversational Flow:** Maintains natural dialogue continuity

### 3. **Enhanced Chunking & Topics**
- **Question Context:** Incorporates user's latest question for richer topics
- **Detailed Topics:** 10-20 word descriptions capturing context, condition, and action
- **Medical Precision:** Includes exact medication names, doses, and clinical instructions
- **Semantic Grouping:** Organizes by medical topic, symptom, assessment, plan, or instruction

### 4. **Robust Fallback Strategy**
- **Primary:** Gemini Flash Lite contextual summarization
- **Secondary:** LTM semantic search with usage-based scoring
- **Tertiary:** STM recent summaries
- **Final:** External RAG knowledge base

### 5. **Performance & Scalability**
- **Efficient Storage:** Semantic deduplication reduces memory footprint
- **Fast Retrieval:** FAISS indexing for sub-millisecond LTM search
- **Smart Eviction:** Usage-based decay and recency scoring
- **Minimal Latency:** Optimized for real-time medical consultations

## 🧪 Example Scenarios

### Scenario 1: STM Deduplication & Merging
```
User: "I have chest pain"
Bot: "This could be angina. Symptoms include pressure, tightness, and shortness of breath."

User: "What about chest pain with shortness of breath?"
Bot: "Chest pain with shortness of breath is concerning for angina or heart attack..."

User: "Tell me more about the symptoms"
Bot: "Angina symptoms include chest pressure, tightness, shortness of breath, and may radiate to arms..."
```
**Result:** STM merges similar responses, creating a comprehensive summary: "Patient has chest pain symptoms consistent with angina, including pressure, tightness, shortness of breath, and potential radiation to arms. This represents a concerning cardiac presentation requiring immediate evaluation."

### Scenario 2: LTM Semantic Retrieval
```
User: "What medications should I avoid with my condition?"
Bot: "Based on your previous discussion about hypertension and the medications mentioned..."
```
**Result:** LTM retrieves relevant medical information about hypertension medications and contraindications from previous conversations, even if not in recent STM.

### Scenario 3: Enhanced Topic Generation
```
User: "I'm having trouble sleeping"
Bot: "Topic: Sleep disturbance evaluation and management for adult patient with insomnia symptoms"
```
**Result:** The topic incorporates the user's question context to create a detailed, medical-specific description instead of just "Sleep problems."

### Scenario 4: Unified Context Summarization
```
User: "Can you repeat the treatment plan?"
Bot: "Based on our conversation about your hypertension and sleep issues, your treatment plan includes..."
```
**Result:** The system creates a unified summary combining STM (recent sleep discussion), LTM (hypertension history), and RAG (current treatment guidelines) into a single coherent narrative.

## ⚙️ Configuration

### Environment Variables
```bash
FlashAPI=your_gemini_api_key  # For both main LLM and contextual analysis
```

### Enhanced Memory Settings
```python
memory = MemoryManager(
    max_users=1000,           # Maximum users in memory
    history_per_user=5,       # STM capacity (5 recent summaries)
    max_chunks=60             # LTM capacity (~20 conversational rounds)
)
```

### Memory Parameters
```python
# STM retrieval (5 recent turns)
recent_history = memory.get_recent_chat_history(user_id, num_turns=5)

# LTM semantic search
rag_chunks = memory.get_relevant_chunks(user_id, query, top_k=3, min_sim=0.30)

# Unified context summarization
contextual_summary = memory.get_contextual_chunks(user_id, current_query, lang)
```

### Similarity Thresholds
```python
# STM deduplication thresholds
IDENTICAL_THRESHOLD = 0.92    # Replace older with newer
MERGE_THRESHOLD = 0.75        # Merge similar content

# LTM semantic search
MIN_SIMILARITY = 0.30         # Minimum similarity for retrieval
TOP_K = 3                     # Number of chunks to retrieve
```

## 🔍 Monitoring & Debugging

### Enhanced Logging
The system provides comprehensive logging for all memory operations:
```python
# STM operations
logger.info(f"[Contextual] Retrieved {len(recent_history)} recent history items")
logger.info(f"[Contextual] Retrieved {len(rag_chunks)} RAG chunks")

# Chunking operations
logger.info(f"[Memory] 📦 Gemini summarized chunk output: {output}")
logger.warning(f"[Memory] ❌ Gemini chunking failed: {e}")

# Contextual summarization
logger.info(f"[Contextual] Gemini created summary: {summary[:100]}...")
logger.warning(f"[Contextual] Gemini summarization failed: {e}")
```

### Performance Metrics
- **STM Operations:** Deduplication rate, merge frequency, topic enrichment quality
- **LTM Operations:** FAISS search latency, semantic similarity scores, eviction patterns
- **Context Summarization:** Gemini response time, summary quality, fallback usage
- **Memory Usage:** Storage efficiency, retrieval hit rates, cache performance

## 🚨 Error Handling

### Enhanced Fallback Strategy
1. **Primary:** Gemini Flash Lite contextual summarization
2. **Secondary:** LTM semantic search with usage-based scoring
3. **Tertiary:** STM recent summaries
4. **Final:** External RAG knowledge base
5. **Emergency:** No context (minimal response)

### Error Scenarios & Recovery
- **Gemini API failure** → Fall back to LTM semantic search
- **LTM corruption** → Rebuild FAISS index from remaining chunks
- **STM corruption** → Reset to empty STM, continue with LTM
- **Memory corruption** → Reset user session, clear all memory
- **Chunking failure** → Store raw response as fallback chunk

## 🔮 Future Enhancements

### 1. **Persistent Memory Storage**
- **Database Integration:** Store LTM in PostgreSQL/SQLite with FAISS index persistence
- **Session Recovery:** Resume conversations after system restarts
- **Memory Export:** Allow users to export their conversation history
- **Cross-device Sync:** Synchronize memory across different devices

### 2. **Advanced Memory Features**
- **Fact Store:** Dedicated storage for critical medical facts (allergies, chronic conditions, medications)
- **Memory Compression:** Summarize older STM entries into LTM when STM overflows
- **Contextual Tags:** Add metadata tags (encounter type, modality, urgency) to bias retrieval
- **Memory Analytics:** Track memory usage patterns and optimize storage strategies

### 3. **Intelligent Memory Management**
- **Adaptive Thresholds:** Dynamically adjust similarity thresholds based on conversation context
- **Memory Prioritization:** Protect critical medical information from eviction
- **Usage-based Retention:** Keep frequently accessed information longer
- **Semantic Clustering:** Group related memories for better organization

### 4. **Enhanced Medical Context**
- **Clinical Decision Support:** Integrate with medical guidelines and protocols
- **Risk Assessment:** Track and alert on potential medical risks across conversations
- **Medication Reconciliation:** Maintain accurate medication lists across sessions
- **Follow-up Scheduling:** Track recommended follow-ups and reminders

### 5. **Multi-modal Memory**
- **Image Memory:** Store and retrieve medical images with descriptions
- **Voice Memory:** Convert voice interactions to text for memory storage
- **Document Memory:** Process and store medical documents and reports
- **Temporal Memory:** Track changes in symptoms and conditions over time

## 📝 Testing

### Memory System Testing
```bash
cd Medical-Chatbot
python test_memory_system.py
```

### Test Scenarios
1. **STM Deduplication Test:** Verify similar responses are merged correctly
2. **LTM Semantic Search Test:** Test FAISS retrieval with various queries
3. **Context Summarization Test:** Validate unified summary generation
4. **Topic Enrichment Test:** Check detailed topic generation with question context
5. **Memory Capacity Test:** Verify STM (5 items) and LTM (60 items) limits
6. **Fallback Strategy Test:** Test system behavior when Gemini API fails

### Expected Behaviors
- **STM:** Similar responses merge, unique details preserved
- **LTM:** Semantic search returns relevant chunks with usage tracking
- **Topics:** Detailed, medical-specific descriptions (10-20 words)
- **Summaries:** Coherent narratives combining STM + LTM + RAG
- **Performance:** Sub-second retrieval times for all operations

## 🎯 Summary

The enhanced memory system transforms the Medical Chatbot into a sophisticated, memory-aware medical assistant that:

✅ **Maintains Short-Term Memory (STM)** with 5 recent conversation summaries and intelligent deduplication  
✅ **Provides Long-Term Memory (LTM)** with 60 semantic chunks and FAISS-based retrieval  
✅ **Generates Enhanced Topics** using question context for detailed, medical-specific descriptions  
✅ **Creates Unified Summaries** combining STM + LTM + RAG into coherent narratives  
✅ **Implements Smart Merging** that preserves unique details while eliminating redundancy  
✅ **Ensures Conversational Continuity** across extended medical consultations  
✅ **Optimizes Performance** with sub-second retrieval and efficient memory management  

This advanced memory system addresses the limitations of simple RAG systems by providing:
- **Intelligent context management** that remembers and builds upon previous interactions
- **Medical precision** with detailed topics and exact clinical information
- **Scalable architecture** that can handle extended conversations without performance degradation
- **Robust fallback strategies** ensuring system reliability in all scenarios

The result is a medical chatbot that truly understands conversation context, remembers patient history, and provides increasingly relevant and personalized medical guidance over time.
