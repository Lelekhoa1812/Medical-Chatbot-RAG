# api/routes.py
import time
import os
import re
import json
import logging
import uuid
from datetime import datetime, timedelta
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, HTMLResponse
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
    video_mode = body.get("video", False)
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
        "video_mode": video_mode,
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
        answer = chatbot.chat(user_id, query, lang, image_diagnosis, search_mode, video_mode)
        elapsed = time.time() - start
        
        # Handle response format (might be string or dict with videos)
        if isinstance(answer, dict):
            response_text = answer.get('text', '')
            video_data = answer.get('videos', [])
        else:
            response_text = answer
            video_data = []
        
        # Store completed response
        completed_response = {
            "request_id": request_id,
            "user_id": user_id,
            "query": query,
            "response": f"{response_text}\n\n(Response time: {elapsed:.2f}s)",
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
        response_data = {
            "response": completed_response["response"],
            "request_id": request_id
        }
        
        # Include video data if available
        if video_data:
            response_data["videos"] = video_data
        
        return JSONResponse(response_data)
        
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
    """Root endpoint - Landing page with redirect to main app"""
    
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Medical Chatbot API</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Inter', sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                overflow: hidden;
                position: relative;
            }
            
            /* Animated background particles */
            .particles {
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                overflow: hidden;
                z-index: 1;
            }
            
            .particle {
                position: absolute;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 50%;
                animation: float 6s ease-in-out infinite;
            }
            
            .particle:nth-child(1) { width: 80px; height: 80px; top: 20%; left: 10%; animation-delay: 0s; }
            .particle:nth-child(2) { width: 120px; height: 120px; top: 60%; left: 80%; animation-delay: 2s; }
            .particle:nth-child(3) { width: 60px; height: 60px; top: 80%; left: 20%; animation-delay: 4s; }
            .particle:nth-child(4) { width: 100px; height: 100px; top: 10%; left: 70%; animation-delay: 1s; }
            .particle:nth-child(5) { width: 90px; height: 90px; top: 40%; left: 50%; animation-delay: 3s; }
            
            @keyframes float {
                0%, 100% { transform: translateY(0px) rotate(0deg); opacity: 0.7; }
                50% { transform: translateY(-20px) rotate(180deg); opacity: 1; }
            }
            
            .container {
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(20px);
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 24px;
                padding: 3rem 2rem;
                text-align: center;
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
                max-width: 500px;
                width: 90%;
                position: relative;
                z-index: 2;
                animation: slideUp 0.8s ease-out;
            }
            
            @keyframes slideUp {
                from {
                    opacity: 0;
                    transform: translateY(50px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            
            .logo {
                width: 80px;
                height: 80px;
                background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                border-radius: 20px;
                display: flex;
                align-items: center;
                justify-content: center;
                margin: 0 auto 1.5rem;
                animation: pulse 2s ease-in-out infinite;
            }
            
            @keyframes pulse {
                0%, 100% { transform: scale(1); }
                50% { transform: scale(1.05); }
            }
            
            .logo i {
                font-size: 2rem;
                color: white;
            }
            
            h1 {
                color: white;
                font-size: 2.5rem;
                font-weight: 700;
                margin-bottom: 0.5rem;
                background: linear-gradient(135deg, #ffffff 0%, #f0f9ff 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }
            
            .subtitle {
                color: rgba(255, 255, 255, 0.8);
                font-size: 1.1rem;
                margin-bottom: 2rem;
                font-weight: 400;
            }
            
            .version {
                color: rgba(255, 255, 255, 0.6);
                font-size: 0.9rem;
                margin-bottom: 2rem;
                font-weight: 300;
            }
            
            .redirect-btn {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                padding: 1rem 2rem;
                border-radius: 12px;
                font-size: 1.1rem;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                text-decoration: none;
                display: inline-flex;
                align-items: center;
                gap: 0.5rem;
                box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
                position: relative;
                overflow: hidden;
            }
            
            .redirect-btn::before {
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
                transition: left 0.5s;
            }
            
            .redirect-btn:hover::before {
                left: 100%;
            }
            
            .redirect-btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 12px 30px rgba(102, 126, 234, 0.4);
            }
            
            .redirect-btn:active {
                transform: translateY(0);
            }
            
            .redirect-btn i {
                font-size: 1.2rem;
                transition: transform 0.3s ease;
            }
            
            .redirect-btn:hover i {
                transform: translateX(3px);
            }
            
            .features {
                margin-top: 2rem;
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
                gap: 1rem;
            }
            
            .feature {
                color: rgba(255, 255, 255, 0.7);
                font-size: 0.9rem;
                font-weight: 500;
            }
            
            .feature i {
                display: block;
                font-size: 1.5rem;
                margin-bottom: 0.5rem;
                color: rgba(255, 255, 255, 0.9);
            }
            
            @media (max-width: 768px) {
                .container {
                    padding: 2rem 1.5rem;
                    margin: 1rem;
                }
                
                h1 {
                    font-size: 2rem;
                }
                
                .subtitle {
                    font-size: 1rem;
                }
                
                .redirect-btn {
                    padding: 0.8rem 1.5rem;
                    font-size: 1rem;
                }
            }
        </style>
    </head>
    <body>
        <div class="particles">
            <div class="particle"></div>
            <div class="particle"></div>
            <div class="particle"></div>
            <div class="particle"></div>
            <div class="particle"></div>
        </div>
        
        <div class="container">
            <div class="logo">
                <i class="fas fa-stethoscope"></i>
            </div>
            
            <h1>Medical Chatbot</h1>
            <p class="subtitle">AI-Powered Health Assistant</p>
            <p class="version">API Version 3.0.0</p>
            
            <a href="https://medical-chatbot-henna.vercel.app/" class="redirect-btn" target="_blank">
                <i class="fas fa-external-link-alt"></i>
                Launch Application
            </a>
            
            <div class="features">
                <div class="feature">
                    <i class="fas fa-brain"></i>
                    AI-Powered
                </div>
                <div class="feature">
                    <i class="fas fa-shield-alt"></i>
                    Secure
                </div>
                <div class="feature">
                    <i class="fas fa-globe"></i>
                    Multi-Language
                </div>
            </div>
        </div>
        
        <script>
            // Add some interactive effects
            document.addEventListener('DOMContentLoaded', function() {
                const btn = document.querySelector('.redirect-btn');
                const particles = document.querySelectorAll('.particle');
                
                // Add click animation
                btn.addEventListener('click', function(e) {
                    // Create ripple effect
                    const ripple = document.createElement('span');
                    const rect = this.getBoundingClientRect();
                    const size = Math.max(rect.width, rect.height);
                    const x = e.clientX - rect.left - size / 2;
                    const y = e.clientY - rect.top - size / 2;
                    
                    ripple.style.cssText = `
                        position: absolute;
                        width: ${size}px;
                        height: ${size}px;
                        left: ${x}px;
                        top: ${y}px;
                        background: rgba(255, 255, 255, 0.3);
                        border-radius: 50%;
                        transform: scale(0);
                        animation: ripple 0.6s ease-out;
                        pointer-events: none;
                    `;
                    
                    this.appendChild(ripple);
                    
                    setTimeout(() => {
                        ripple.remove();
                    }, 600);
                });
                
                // Add CSS for ripple animation
                const style = document.createElement('style');
                style.textContent = `
                    @keyframes ripple {
                        to {
                            transform: scale(2);
                            opacity: 0;
                        }
                    }
                `;
                document.head.appendChild(style);
                
                // Animate particles on mouse move
                document.addEventListener('mousemove', function(e) {
                    const x = e.clientX / window.innerWidth;
                    const y = e.clientY / window.innerHeight;
                    
                    particles.forEach((particle, index) => {
                        const speed = (index + 1) * 0.5;
                        const xOffset = (x - 0.5) * speed * 20;
                        const yOffset = (y - 0.5) * speed * 20;
                        
                        particle.style.transform = `translate(${xOffset}px, ${yOffset}px)`;
                    });
                });
            });
        </script>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html_content)
