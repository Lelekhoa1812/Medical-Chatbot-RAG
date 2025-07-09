# vlm.py
import os
from huggingface_hub import InferenceClient
from translation import translate_query
# Initialise once
HF_TOKEN = os.getenv("HF_TOKEN")
client = InferenceClient(provider="auto", api_key=HF_TOKEN)

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
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Error from image diagnosis model: {e}"
