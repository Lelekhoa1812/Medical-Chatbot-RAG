# api/config.py
import os
import logging
import psutil
from typing import List

# ✅ Environment Variables
mongo_uri = os.getenv("MONGO_URI")
index_uri = os.getenv("INDEX_URI")
# Legacy Gemini key kept for backward compatibility only; Azure AI Foundry is the primary provider.
gemini_flash_api_key = os.getenv("FlashAPI")
foundry_api_key = os.getenv("FOUNDRY_API_KEY")
foundry_endpoint = os.getenv("FOUNDRY_ENDPOINT")

# Validate environment endpoint (only when actually running the app)
def validate_environment():
    if not all([mongo_uri, index_uri]):
        raise ValueError("❌ Missing required infrastructure variables! Set MONGO_URI and INDEX_URI in environment.")

    # Prefer Azure AI Foundry, but do not fail startup solely because the legacy Gemini key is absent.
    if not all([foundry_api_key, foundry_endpoint]):
        logging.getLogger("medical-chatbot").warning(
            "⚠️ Azure AI Foundry credentials are not fully configured. "
            "Model-backed features may be unavailable until FOUNDRY_API_KEY and FOUNDRY_ENDPOINT are set."
        )

# ✅ Logging Configuration
def setup_logging():
    """Configure logging for the application"""
    # Silence noisy loggers
    for name in [
        "uvicorn.error", "uvicorn.access",
        "fastapi", "starlette",
        "pymongo", "gridfs",
        "sentence_transformers", "faiss",
        "google", "google.auth",
    ]:
        logging.getLogger(name).setLevel(logging.WARNING)
    
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s — %(name)s — %(levelname)s — %(message)s", 
        force=True
    )
    
    logger = logging.getLogger("medical-chatbot")
    logger.setLevel(logging.DEBUG)
    return logger

# ✅ System Resource Monitoring
def check_system_resources(logger):
    """Monitor system resources and log warnings"""
    memory = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=1)
    disk = psutil.disk_usage("/")
    
    logger.info(f"[System] 🔍 System Resources - RAM: {memory.percent}%, CPU: {cpu}%, Disk: {disk.percent}%")
    
    if memory.percent > 85:
        logger.warning("⚠️ High RAM usage detected!")
    if cpu > 90:
        logger.warning("⚠️ High CPU usage detected!")
    if disk.percent > 90:
        logger.warning("⚠️ High Disk usage detected!")

# ✅ Memory Optimization
def optimize_memory():
    """Set environment variables for memory optimization"""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ✅ CORS Configuration
CORS_ORIGINS = [
    "http://localhost:5173",                    # Vite dev server
    "http://localhost:3000",                    # Another vercel local dev
    "https://medical-chatbot-henna.vercel.app", # ✅ Vercel frontend production URL
]

# ✅ Model Configuration
MODEL_CACHE_DIR = "/app/model_cache"
EMBEDDING_MODEL_DEVICE = "cpu"
