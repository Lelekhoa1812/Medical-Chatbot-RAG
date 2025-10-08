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
    """Collapse excessive repeated n-grams and repeated phrases with improved logic."""
    if not s:
        return s
    
    # Collapse repeated spaces/newlines
    s = re.sub(r"\s+", " ", s).strip()
    
    # More aggressive repetition detection
    # Check for simple word repetition (like "a lot of people do not" repeated)
    words = s.split()
    if len(words) > 20:  # Only check if text is long enough
        # Look for repeated sequences of 3-8 words
        for seq_len in range(8, 2, -1):
            if len(words) < seq_len * 3:  # Need at least 3 repetitions
                continue
            
            # Check each possible starting position
            for start in range(len(words) - seq_len * 2):
                sequence = words[start:start + seq_len]
                # Count how many times this sequence repeats
                repeat_count = 1
                pos = start + seq_len
                while pos + seq_len <= len(words):
                    if words[pos:pos + seq_len] == sequence:
                        repeat_count += 1
                        pos += seq_len
                    else:
                        break
                
                # If we found 3+ repetitions, remove the excess
                if repeat_count >= 3:
                    # Keep only the first occurrence
                    new_words = words[:start + seq_len] + words[start + seq_len * repeat_count:]
                    s = " ".join(new_words)
                    words = s.split()
                    break
            else:
                continue
            break  # Break outer loop if we found and fixed a repetition
    
    # Additional cleanup for remaining patterns
    # Remove consecutive identical word
    tokens = s.split()
    out = []
    last = None
    for t in tokens:
        if last is None or t.lower() != last.lower():
            out.append(t)
        last = t
    s = " ".join(out)
    
    # Limit consecutive duplicate n-grams
    for n in range(n_max, n_min - 1, -1):
        pattern = re.compile(r"(\b(?:\w+\s+){%d}\w+\b)(?:\s+\1){2,}" % (n - 1), flags=re.IGNORECASE)
        s = pattern.sub(r"\1", s)
    
    return s


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
    
    if not text or not text.strip():
        return text
    
    try:
        if lang_code == "vi":
            if vi_en is None:
                logger.info("[Translation] Loading Vietnamese-English model...")
                vi_en = pipeline("translation", model="VietAI/envit5-translation", src_lang="vi", tgt_lang="en", device=-1)
            
            # Limit input length to prevent model issues
            input_text = text[:1000] if len(text) > 1000 else text
            raw = vi_en(input_text, max_length=512)[0]["translation_text"]
            cleaned = _dedupe_repeats(raw)
            norm = _normalize_and_cap(cleaned, cap=512)
            
            if _is_too_repetitive(norm) or len(norm.strip()) < 10:
                logger.warning("[En-Vi] Translation repetitive or too short; falling back to original text")
                return text
                
            logger.info(f"[En-Vi] Query in `{lang_code}` translated to: {norm[:100]}...")
            return norm
            
        elif lang_code == "zh":
            if zh_en is None:
                logger.info("[Translation] Loading Chinese-English model...")
                zh_en = pipeline("translation", model="Helsinki-NLP/opus-mt-zh-en", device=-1)
            
            # Limit input length to prevent model issues
            input_text = text[:1000] if len(text) > 1000 else text
            raw = zh_en(input_text, max_length=512)[0]["translation_text"]
            cleaned = _dedupe_repeats(raw)
            norm = _normalize_and_cap(cleaned, cap=512)
            
            if _is_too_repetitive(norm) or len(norm.strip()) < 10:
                logger.warning("[En-Zh] Translation repetitive or too short; falling back to original text")
                return text
                
            logger.info(f"[En-Zh] Query in `{lang_code}` translated to: {norm[:100]}...")
            return norm
            
    except Exception as e:
        logger.error(f"[Translation] Translation failed for {lang_code}: {e}")
        return text  # Fallback to original text
    
    return text
