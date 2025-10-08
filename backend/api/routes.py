# api/routes.py
import time, os, re, json
import logging
import uuid
from datetime import datetime, timedelta
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from .chatbot import RAGMedicalChatbot
from .retrieval import retrieval_engine
from .database import db_manager
from utils import process_medical_image

logger = logging.getLogger("routes")

# Create router
router = APIRouter()

# Initialize chatbot
chatbot = RAGMedicalChatbot(
    model_name="gemini-2.5-flash", 
    retrieve_function=retrieval_engine.retrieve_medical_info
)

@router.post("/chat")
async def chat_endpoint(req: Request):
    """Main chat endpoint with search mode support and request persistence"""
    body = await req.json()
    user_id = body.get("user_id", "anonymous")
    query_raw = body.get("query")
    query = query_raw.strip() if isinstance(query_raw, str) else ""
    lang = body.get("lang", "EN")
    search_mode = body.get("search", False)
    image_base64 = body.get("image_base64", None)
    img_desc = body.get("img_desc", "Describe and investigate any clinical findings from this medical image.")
    
    # Generate unique request ID
    request_id = str(uuid.uuid4())
    
    # Store pending request in database
    pending_request = {
        "request_id": request_id,
        "user_id": user_id,
        "query": query,
        "lang": lang,
        "search_mode": search_mode,
        "image_base64": image_base64,
        "img_desc": img_desc,
        "status": "pending",
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    
    try:
        # Get requests collection
        requests_collection = db_manager.get_qa_collection().database["chat_requests"]
        requests_collection.insert_one(pending_request)
        logger.info(f"[REQUEST] Stored pending request {request_id} for user {user_id}")
    except Exception as e:
        logger.error(f"[REQUEST] Failed to store pending request: {e}")
        # Continue processing even if storage fails
    
    start = time.time()
    image_diagnosis = ""
    
    # LLM Only
    if not image_base64:
        logger.info(f"[BOT] LLM scenario. Search mode: {search_mode}")
    # LLM+VLM
    else:
        # If image is present → diagnose first
        safe_load = len(image_base64.encode("utf-8"))
        if safe_load > 6_000_000:  # Img size safe processor
            # Update request status to failed
            try:
                requests_collection.update_one(
                    {"request_id": request_id},
                    {"$set": {"status": "failed", "error": "Image too large", "updated_at": datetime.utcnow()}}
                )
            except:
                pass
            return JSONResponse({"response": "⚠️ Image too large. Please upload smaller images (<5MB).", "request_id": request_id})
        logger.info(f"[BOT] VLM+LLM scenario. Search mode: {search_mode}")
        logger.info(f"[VLM] Process medical image size: {safe_load}, desc: {img_desc}, {lang}.")
        image_diagnosis = process_medical_image(image_base64, img_desc, lang)
    
    try:
        answer = chatbot.chat(user_id, query, lang, image_diagnosis, search_mode)
        elapsed = time.time() - start
        
        # Store completed response
        completed_response = {
            "request_id": request_id,
            "user_id": user_id,
            "query": query,
            "response": f"{answer}\n\n(Response time: {elapsed:.2f}s)",
            "status": "completed",
            "created_at": pending_request["created_at"],
            "completed_at": datetime.utcnow(),
            "response_time": elapsed
        }
        
        try:
            requests_collection.update_one(
                {"request_id": request_id},
                {"$set": {
                    "status": "completed",
                    "response": completed_response["response"],
                    "completed_at": completed_response["completed_at"],
                    "response_time": elapsed,
                    "updated_at": datetime.utcnow()
                }}
            )
            logger.info(f"[REQUEST] Stored completed response for request {request_id}")
        except Exception as e:
            logger.error(f"[REQUEST] Failed to store completed response: {e}")
        
        # Final response
        return JSONResponse({
            "response": completed_response["response"],
            "request_id": request_id
        })
        
    except Exception as e:
        logger.error(f"[REQUEST] Error processing request {request_id}: {e}")
        # Update request status to failed
        try:
            requests_collection.update_one(
                {"request_id": request_id},
                {"$set": {"status": "failed", "error": str(e), "updated_at": datetime.utcnow()}}
            )
        except:
            pass
        return JSONResponse({"response": "❌ Failed to get a response. Please try again.", "request_id": request_id})

@router.get("/check-request/{request_id}")
async def check_request_status(request_id: str):
    """Check the status of a specific request"""
    try:
        requests_collection = db_manager.get_qa_collection().database["chat_requests"]
        request_data = requests_collection.find_one({"request_id": request_id})
        
        if not request_data:
            return JSONResponse({"status": "not_found", "message": "Request not found"})
        
        # Remove sensitive data before returning
        request_data.pop("_id", None)
        request_data.pop("image_base64", None)  # Don't return large image data
        
        return JSONResponse(request_data)
    except Exception as e:
        logger.error(f"[REQUEST] Error checking request {request_id}: {e}")
        return JSONResponse({"status": "error", "message": "Failed to check request status"})

@router.get("/pending-requests/{user_id}")
async def get_pending_requests(user_id: str):
    """Get all pending requests for a user"""
    try:
        requests_collection = db_manager.get_qa_collection().database["chat_requests"]
        pending_requests = list(requests_collection.find({
            "user_id": user_id,
            "status": {"$in": ["pending", "completed"]}
        }).sort("created_at", -1).limit(10))
        
        # Remove sensitive data and MongoDB ObjectId
        for req in pending_requests:
            req.pop("_id", None)
            req.pop("image_base64", None)
        
        return JSONResponse({"requests": pending_requests})
    except Exception as e:
        logger.error(f"[REQUEST] Error getting pending requests for user {user_id}: {e}")
        return JSONResponse({"requests": []})

@router.delete("/cleanup-requests")
async def cleanup_old_requests():
    """Clean up old completed requests (older than 24 hours)"""
    try:
        requests_collection = db_manager.get_qa_collection().database["chat_requests"]
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        
        result = requests_collection.delete_many({
            "status": "completed",
            "completed_at": {"$lt": cutoff_time}
        })
        
        logger.info(f"[CLEANUP] Removed {result.deleted_count} old requests")
        return JSONResponse({"deleted_count": result.deleted_count})
    except Exception as e:
        logger.error(f"[CLEANUP] Error cleaning up old requests: {e}")
        return JSONResponse({"deleted_count": 0, "error": str(e)})

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "medical-chatbot"}

@router.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Medical Chatbot API", "version": "3.0.0"}
