# main.py - Entry point for the Medical Chatbot API
from api.app import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info")
