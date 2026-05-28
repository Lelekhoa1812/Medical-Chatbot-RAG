# API Module Structure

## 📁 **Module Overview**

### **config.py** - Configuration Management
- Environment variables validation
- Logging configuration
- System resource monitoring
- Memory optimization settings
- CORS configuration
- Azure AI Foundry provider settings

### **database.py** - Database Management
- MongoDB connection management
- FAISS index lazy loading
- SentenceTransformer model initialization
- Symptom vectors management
- GridFS integration

### **retrieval.py** - RAG Retrieval Engine
- Medical information retrieval from FAISS
- Symptom-based diagnosis retrieval
- Smart deduplication and similarity matching
- Vector similarity computations
- Optional reranking support for guideline-heavy content

### **chatbot.py** - Core Chatbot Logic
- RAGMedicalChatbot class
- Azure LLM client usage
- Search mode integration
- Citation processing
- Memory management integration

### **routes.py** - API Endpoints
- `/chat` - Main chat endpoint
- `/health` - Health check
- `/` - Root endpoint
- Request/response handling

### **app.py** - Main Application
- FastAPI app initialization
- Middleware configuration
- Database initialization
- Route registration
- Server startup

## 🔄 **Data Flow**

```
Request → routes.py → chatbot.py → retrieval.py → database.py
                ↓
         memory.py (context) + search.py (web search)
                ↓
         models/ (provider-backed model processing)
                ↓
         Response with citations
```

## 🚀 **Benefits of Modular Structure**

1. **Separation of Concerns**: Each module has a single responsibility
2. **Easier Testing**: Individual modules can be tested in isolation
3. **Better Maintainability**: Changes to one module don't affect others
4. **Improved Readability**: Smaller files are easier to understand
5. **Reusability**: Modules can be imported and used elsewhere
6. **Scalability**: Easy to add new features without affecting existing code

## 📊 **File Sizes Comparison**

| File | Lines | Purpose |
|------|-------|---------|
| **app_old.py** | 370 | Monolithic (everything) |
| **app.py** | 45 | Main app initialization |
| **config.py** | 65 | Configuration |
| **database.py** | 95 | Database management |
| **retrieval.py** | 85 | RAG retrieval |
| **chatbot.py** | 120 | Chatbot logic |
| **routes.py** | 55 | API endpoints |
| **Total** | 465 | Modular structure |

## 🔧 **Usage**

The modular structure maintains the same API interface:

```python
# All imports work the same way
from api.app import app
from api.chatbot import RAGMedicalChatbot
from api.retrieval import retrieval_engine
```

## 🛠 **Development Benefits**

- **Easier Debugging**: Issues can be isolated to specific modules
- **Parallel Development**: Multiple developers can work on different modules
- **Code Reviews**: Smaller files are easier to review
- **Documentation**: Each module can have focused documentation
- **Testing**: Unit tests can be written for each module independently
