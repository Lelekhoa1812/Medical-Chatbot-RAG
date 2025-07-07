# translation.py
from transformers import pipeline
import logging

logger = logging.getLogger("translation-agent")
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(name)s — %(levelname)s — %(message)s", force=True) # Change INFO to DEBUG for full-ctx JSON loader

# To use lazy model loader
vi_en = None
zh_en = None

def translate_query(text: str, lang_code: str) -> str:
    global vi_en, zh_en
    if lang_code == "vi":
        if vi_en is None:
            vi_en = pipeline("translation", model="VietAI/envit5-translation", src_lang="vi", tgt_lang="en", device=-1)
        result = vi_en(text, max_length=512)[0]["translation_text"]
        logger.info(f"[En-Vi] Query in `{lang_code}` translated to: {result}")
        return result
    elif lang_code == "zh":
        if zh_en is None:
            zh_en = pipeline("translation", model="Helsinki-NLP/opus-mt-zh-en", device=-1)
        result = zh_en(text, max_length=512)[0]["translation_text"]
        logger.info(f"[En-Zh] Query in `{lang_code}` translated to: {result}")
        return result
    return text
