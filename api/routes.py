# api/routes.py
import time
import logging
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from .chatbot import RAGMedicalChatbot
from .retrieval import retrieval_engine
from utils import process_medical_image

logger = logging.getLogger("medical-chatbot")

# Create router
router = APIRouter()

# Initialize chatbot
chatbot = RAGMedicalChatbot(
    model_name="gemini-2.5-flash", 
    retrieve_function=retrieval_engine.retrieve_medical_info
)

@router.post("/chat")
async def chat_endpoint(req: Request):
    """Main chat endpoint with search mode support"""
    body = await req.json()
    user_id = body.get("user_id", "anonymous")
    query_raw = body.get("query")
    query = query_raw.strip() if isinstance(query_raw, str) else ""
    lang = body.get("lang", "EN")
    search_mode = body.get("search", False)
    image_base64 = body.get("image_base64", None)
    img_desc = body.get("img_desc", "Describe and investigate any clinical findings from this medical image.")
    
    start = time.time()
    image_diagnosis = ""
    
    # LLM Only
    if not image_base64:
        logger.info(f"[BOT] LLM scenario. Search mode: {search_mode}")
    # LLM+VLM
    else:
        # If image is present → diagnose first
        safe_load = len(image_base64.encode("utf-8"))
        if safe_load > 5_000_000:  # Img size safe processor
            return JSONResponse({"response": "⚠️ Image too large. Please upload smaller images (<5MB)."})
        logger.info(f"[BOT] VLM+LLM scenario. Search mode: {search_mode}")
        logger.info(f"[VLM] Process medical image size: {safe_load}, desc: {img_desc}, {lang}.")
        image_diagnosis = process_medical_image(image_base64, img_desc, lang)
    
    answer = chatbot.chat(user_id, query, lang, image_diagnosis, search_mode)
    elapsed = time.time() - start
    
    # Final
    return JSONResponse({"response": f"{answer}\n\n(Response time: {elapsed:.2f}s)"})

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "medical-chatbot"}

@router.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Medical Chatbot API", "version": "1.0.0"}
