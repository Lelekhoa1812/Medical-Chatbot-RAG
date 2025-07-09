# vlm.py
import os, logging, traceback, json, base64
from io import BytesIO
from PIL import Image
# from huggingface_hub import InferenceClient # Render model on HF hub
from transformers import pipeline             # Render model on transformers
from translation import translate_query

# Initialise once
HF_TOKEN = os.getenv("HF_TOKEN")
# client = InferenceClient(provider="auto", api_key=HF_TOKEN) # comment in back

logger = logging.getLogger("vlm-agent")
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(name)s — %(levelname)s — %(message)s", force=True) # Change INFO to DEBUG for full-ctx JSON loader

# ✅ Load VLM pipeline once (lazy load allowed)
vlm_pipe = None
def load_vlm():
    global vlm_pipe
    if vlm_pipe is None:
        logger.info("⏳ Loading MedGEMMA model via Transformers pipeline...")
        vlm_pipe = pipeline("image-to-text", model="google/medgemma-4b", use_auth_token=HF_TOKEN, device_map="auto")
        logger.info("✅ MedGEMMA model ready.")
    return vlm_pipe

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
        # HF hub
        # response = client.chat.completions.create(
        #     model="google/medgemma-4b-it",
        #     messages=[{
        #         "role": "user",
        #         "content": [
        #             {"type": "text", "text": prompt},
        #             {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        #         ]
        #     }]
        # )
        # Transformers
        image_data = base64.b64decode(base64_image) # Decode base64 to PIL Image
        image = Image.open(BytesIO(image_data)).convert("RGB")
        pipe = load_vlm()
        response = pipe(image, prompt=prompt, max_new_tokens=100)[0]["generated_text"]
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
