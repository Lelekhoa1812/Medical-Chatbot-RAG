# translation.py
from transformers import pipeline
import logging
import re
from collections import Counter

logger = logging.getLogger("translation-agent")
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(name)s — %(levelname)s — %(message)s", force=True) # Change INFO to DEBUG for full-ctx JSON loader

# To use lazy model loader
vi_en = None
zh_en = None

def _dedupe_repeats(s: str, n_min: int = 3, n_max: int = 7) -> str:
    """Collapse excessive repeated n-grams (3..7) and repeated phrases."""
    if not s:
        return s
    # Collapse repeated spaces/newlines
    s = re.sub(r"\s+", " ", s).strip()
    # Heuristic: remove runs of identical tokens
    tokens = s.split()
    out = []
    last = None
    for t in tokens:
        if last is None or t.lower() != last.lower():
            out.append(t)
        last = t
    s2 = " ".join(out)
    # Limit consecutive duplicate n-grams
    for n in range(n_max, n_min - 1, -1):
        pattern = re.compile(r"(\b(?:\w+\s+){%d}\w+\b)(?:\s+\1){2,}" % (n - 1), flags=re.IGNORECASE)
        s2 = pattern.sub(r"\1", s2)
    return s2


def _normalize_and_cap(s: str, cap: int = 512) -> str:
    if not s:
        return s
    s = s.strip()
    if len(s) > cap:
        s = s[:cap]
    return s


def _is_too_repetitive(s: str, threshold: float = 0.4) -> bool:
    if not s:
        return False
    tokens = [t.lower() for t in s.split()]
    if len(tokens) < 10:
        return False
    counts = Counter(tokens)
    top = counts.most_common(1)[0][1]
    return (top / max(1, len(tokens))) >= threshold


def translate_query(text: str, lang_code: str) -> str:
    global vi_en, zh_en
    if lang_code == "vi":
        if vi_en is None:
            vi_en = pipeline("translation", model="VietAI/envit5-translation", src_lang="vi", tgt_lang="en", device=-1)
        raw = vi_en(text, max_length=512)[0]["translation_text"]
        cleaned = _dedupe_repeats(raw)
        norm = _normalize_and_cap(cleaned, cap=512)
        if _is_too_repetitive(norm):
            logger.warning("[En-Vi] Translation repetitive; falling back to original text")
            norm = text
        logger.info(f"[En-Vi] Query in `{lang_code}` translated to: {norm}")
        return norm
    elif lang_code == "zh":
        if zh_en is None:
            zh_en = pipeline("translation", model="Helsinki-NLP/opus-mt-zh-en", device=-1)
        raw = zh_en(text, max_length=512)[0]["translation_text"]
        cleaned = _dedupe_repeats(raw)
        norm = _normalize_and_cap(cleaned, cap=512)
        if _is_too_repetitive(norm):
            logger.warning("[En-Zh] Translation repetitive; falling back to original text")
            norm = text
        logger.info(f"[En-Zh] Query in `{lang_code}` translated to: {norm}")
        return norm
    return text
