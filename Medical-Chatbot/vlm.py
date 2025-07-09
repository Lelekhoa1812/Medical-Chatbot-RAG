# vlm.py
import os, logging, traceback, json
from huggingface_hub import InferenceClient
from translation import translate_query

# Initialise once
HF_TOKEN = os.getenv("HF_TOKEN")
client = InferenceClient(provider="auto", api_key=HF_TOKEN)

logger = logging.getLogger("vlm-agent")
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(name)s — %(levelname)s — %(message)s", force=True) # Change INFO to DEBUG for full-ctx JSON loader

def process_medical_image(base64_image: str, prompt: str = None, lang: str = "EN") -> str:
    """
    Send base64 image + prompt to MedGEMMA and return output.
    """
    if not prompt:
        prompt = "Describe and investigate any clinical findings from this medical image."
    elif prompt and (lang.upper() in {"VI", "ZH"}):
        user_query = translate_query(user_query, lang.lower())
    # Send over API
    try:
        response = client.chat.completions.create(
            model="google/medgemma-4b-it",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }]
        )
        # Validate response
        if not response or not hasattr(response, "choices") or not response.choices:
            raise ValueError("Empty or malformed response from MedGEMMA.")
        # Get choice resp
        message = response.choices[0].message
        if not message or not hasattr(message, "content"):
            raise ValueError("MedGEMMA response missing `.content`.")
        # Beautify
        result = message.content.strip()
        logger.info(f"[VLM] MedGemma returned {result}")
        return result
    except Exception as e:
        logger.error(f"[VLM] ❌ Exception: {e}")
        logger.error(f"[VLM] 🔍 Traceback:\n{traceback.format_exc()}")
        try:
            logger.error(f"[VLM] ⚠️ Raw response: {json.dumps(response, default=str, indent=2)}")
        except:
            logger.warning("[VLM] ⚠️ Response not serializable.")
        return f"[VLM] ⚠️ Image diagnosis failed: {str(e)}"
