# translation.py
import logging
import re
from collections import Counter

from backend.models.llama import AzureAIClient

logger = logging.getLogger("translation-agent")
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(name)s — %(levelname)s — %(message)s", force=True) # Change INFO to DEBUG for full-ctx JSON loader

_translation_client = None


def _get_translation_client() -> AzureAIClient:
    global _translation_client
    if _translation_client is None:
        _translation_client = AzureAIClient(model_env_var="SLM_MODEL", default_model="gpt-5-nano")
    return _translation_client


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


def _translate_with_slm(text: str, source_language: str, source_label: str) -> str:
    client = _get_translation_client()
    input_text = text[:1000] if len(text) > 1000 else text

    prompt = f"""Translate the following medical user query from {source_label} to English.
Preserve the full medical meaning, symptoms, medications, dosage words, body parts, and urgency.
Do not answer the question.
Do not summarize.
Return only the English translation text.

Text: {input_text}

English translation:"""

    raw = client.chat_completion(
        messages=[
            {
                "role": "system",
                "content": "You are a precise medical translation assistant. Translate only and return only the translated English text.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.0,
        max_tokens=512,
        timeout=30,
        max_retries=3,
    )

    cleaned = _dedupe_repeats(raw)
    norm = _normalize_and_cap(cleaned, cap=512)

    if _is_too_repetitive(norm) or len(norm.strip()) < 2:
        logger.warning(f"[Translation-{source_language}] Translation repetitive or too short; falling back to original text")
        return text

    logger.info(f"[Translation-{source_language}] Query translated to: {norm[:100]}...")
    return norm


def translate_query(text: str, lang_code: str) -> str:
    if not text or not text.strip():
        return text
    
    try:
        if lang_code == "vi":
            return _translate_with_slm(text, "vi", "Vietnamese")
        elif lang_code == "zh":
            return _translate_with_slm(text, "zh", "Chinese")
            
    except Exception as e:
        logger.error(f"[Translation] Translation failed for {lang_code}: {e}")
        return text  # Fallback to original text
    
    return text
