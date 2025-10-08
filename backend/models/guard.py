import os
import requests
import logging
from typing import Tuple, List, Dict


logger = logging.getLogger(__name__)


class SafetyGuard:
    """
    Wrapper around NVIDIA Llama Guard (meta/llama-guard-4-12b) hosted at
    https://integrate.api.nvidia.com/v1/chat/completions

    Exposes helpers to validate:
      - user input safety
      - model output safety (in context of the user question)
    """

    def __init__(self):
        self.api_key = os.getenv("NVIDIA_URI")
        if not self.api_key:
            raise ValueError("NVIDIA_URI environment variable not set for SafetyGuard")
        self.base_url = "https://integrate.api.nvidia.com/v1/chat/completions"
        self.model = "meta/llama-guard-4-12b"
        self.timeout_s = 30

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 2800, overlap: int = 200) -> List[str]:
        """Chunk long text to keep request payloads small enough for the guard.
        Uses character-based approximation with small overlap.
        """
        if not text:
            return [""]
        n = len(text)
        if n <= chunk_size:
            return [text]
        chunks: List[str] = []
        start = 0
        while start < n:
            end = min(start + chunk_size, n)
            chunks.append(text[start:end])
            if end == n:
                break
            start = max(0, end - overlap)
        return chunks

    def _call_guard(self, messages: List[Dict], max_tokens: int = 512) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        # Try OpenAI-compatible schema first
        payload_chat = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "top_p": 0.7,
            "max_tokens": max_tokens,
            "stream": False,
        }
        # Alternative schema (some NVIDIA deployments require message content objects)
        alt_messages = []
        for m in messages:
            content = m.get("content", "")
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            alt_messages.append({"role": m.get("role", "user"), "content": content})
        payload_alt = {
            "model": self.model,
            "messages": alt_messages,
            "temperature": 0.2,
            "top_p": 0.7,
            "max_tokens": max_tokens,
            "stream": False,
        }
        # Attempt primary, then fallback
        for payload in (payload_chat, payload_alt):
            try:
                resp = requests.post(self.base_url, headers=headers, json=payload, timeout=self.timeout_s)
                if resp.status_code >= 400:
                    # Log server message for debugging payload issues
                    try:
                        logger.error(f"[SafetyGuard] HTTP {resp.status_code}: {resp.text[:400]}")
                    except Exception:
                        pass
                    resp.raise_for_status()
                data = resp.json()
                content = (
                    data.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                        .strip()
                )
                if content:
                    return content
            except Exception as e:
                # Try next payload shape
                logger.error(f"[SafetyGuard] Guard API call failed: {e}")
                continue
        # All attempts failed
        return ""

    @staticmethod
    def _parse_guard_reply(text: str) -> Tuple[bool, str]:
        """Parse guard reply; expect 'SAFE' or 'UNSAFE: <reason>' (case-insensitive)."""
        if not text:
            # Fail-open: treat as SAFE if guard unavailable to avoid false blocks
            return True, "guard_unavailable"
        t = text.strip()
        upper = t.upper()
        if upper.startswith("SAFE") and not upper.startswith("SAFEGUARD"):
            return True, ""
        if upper.startswith("UNSAFE"):
            # Extract reason after the first colon if present
            parts = t.split(":", 1)
            reason = parts[1].strip() if len(parts) > 1 else "policy violation"
            return False, reason
        # Fallback: treat unknown response as unsafe
        return False, t[:180]

    def check_user_query(self, user_query: str) -> Tuple[bool, str]:
        """Validate the user query is safe to process (provider-style single user message)."""
        text = user_query or ""
        # If too long, validate each chunk; any UNSAFE makes overall UNSAFE
        for part in self._chunk_text(text):
            messages = [{"role": "user", "content": part}]
            reply = self._call_guard(messages, max_tokens=64)
            ok, reason = self._parse_guard_reply(reply)
            if not ok:
                return False, reason
        return True, ""

    def check_model_answer(self, user_query: str, model_answer: str) -> Tuple[bool, str]:
        """Validate the model's answer is safe using provider's example schema (user + assistant turns)."""
        uq = user_query or ""
        ans = model_answer or ""
        # Chunk assistant answer; if user query is huge, use first chunk of it as context
        user_parts = self._chunk_text(uq, chunk_size=2000)
        user_context = user_parts[0] if user_parts else ""
        for ans_part in self._chunk_text(ans):
            messages = [
                {"role": "user", "content": user_context},
                {"role": "assistant", "content": ans_part},
            ]
            reply = self._call_guard(messages, max_tokens=96)
            ok, reason = self._parse_guard_reply(reply)
            if not ok:
                return False, reason
        return True, ""


# Global instance (optional convenience)
safety_guard = SafetyGuard()