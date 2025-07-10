import os, logging, traceback, json, base64
from io import BytesIO
from PIL import Image
from translation import translate_query
from gradio_client import Client, handle_file
import tempfile

logger = logging.getLogger("vlm-agent")
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(name)s — %(levelname)s — %(message)s", force=True)

# ✅ Load Gradio client once
gr_client = None
def load_gradio_client():
    global gr_client
    if gr_client is None:
        logger.info("[VLM] ⏳ Connecting to MedGEMMA Gradio Space...")
        gr_client = Client("warshanks/medgemma-4b-it")
        logger.info("[VLM] Gradio MedGEMMA client ready.")
    return gr_client

def process_medical_image(base64_image: str, prompt: str = None, lang: str = "EN") -> str:
    if not prompt:
        prompt = "Describe and investigate any clinical findings from this medical image."
    elif lang.upper() in {"VI", "ZH"}:
        prompt = translate_query(prompt, lang.lower())

    try:
        # 1️⃣ Decode base64 image to temp file
        image_data = base64.b64decode(base64_image)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(image_data)
            tmp.flush()
            image_path = tmp.name

        # 2️⃣ Send to Gradio MedGEMMA
        client = load_gradio_client()
        logger.info(f"[VLM] Sending prompt: {prompt}")
        result = client.predict(
            message={"text": prompt, "files": [handle_file(image_path)]},
            param_2 = "You analyze medical images and report abnormalities, diseases with clear diagnostic insight.",
            param_3=2048,
            api_name="/chat"
        )
        if isinstance(result, str):
            logger.info(f"[VLM] ✅ Response: {result}")
            return result.strip()
        else:
            logger.warning(f"[VLM] ⚠️ Unexpected result type: {type(result)} — {result}")
            return str(result)

    except Exception as e:
        logger.error(f"[VLM] ❌ Exception: {e}")
        logger.error(f"[VLM] 🔍 Traceback:\n{traceback.format_exc()}")
        return f"[VLM] ⚠️ Failed to process image: {e}"
