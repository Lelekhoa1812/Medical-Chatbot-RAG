# api/app_new.py
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .config import setup_logging, check_system_resources, optimize_memory, CORS_ORIGINS, validate_environment
from .database import db_manager
from .routes import router

# ✅ Validate environment
validate_environment()

# ✅ Setup logging
logger = setup_logging()
logger.info("🚀 Starting Medical Chatbot API...")

# ✅ Monitor system resources
check_system_resources(logger)

# ✅ Optimize memory usage
optimize_memory()

# ✅ Initialize FastAPI app
app = FastAPI(
    title="Medical Chatbot API",
    description="AI-powered medical chatbot with RAG and search capabilities",
    version="3.0.0"
)

# ✅ Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Initialize database connections
try:
    db_manager.initialize_embedding_model()
    db_manager.initialize_mongodb()
    logger.info("✅ Database connections initialized successfully")
except Exception as e:
    logger.error(f"❌ Database initialization failed: {e}")
    raise

# ✅ Include routes
app.include_router(router)

# ✅ Run Uvicorn
if __name__ == "__main__":
    logger.info("[System] ✅ Starting FastAPI Server...")
    try:
        uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info")
    except Exception as e:
        logger.error(f"❌ Server Startup Failed: {e}")
        exit(1)
