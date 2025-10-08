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

    def _call_guard(self, messages: List[Dict], max_tokens: int = 512) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }
        try:
            resp = requests.post(self.base_url, headers=headers, json=payload, timeout=self.timeout_s)
            resp.raise_for_status()
            data = resp.json()
            content = (
                data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                    .strip()
            )
            return content
        except Exception as e:
            logger.error(f"[SafetyGuard] Guard API call failed: {e}")
            # Fail closed: if guard is unavailable, treat as unsafe by returning empty
            return ""

    @staticmethod
    def _parse_guard_reply(text: str) -> Tuple[bool, str]:
        """Parse guard reply; expect 'SAFE' or 'UNSAFE: <reason>' (case-insensitive)."""
        if not text:
            return False, "safety check unavailable"
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
        """Validate the user query is safe to process."""
        system = (
            "You are a safety classifier. Analyze the USER message for safety policy violations "
            "(violence, self-harm, child sexual content, explicit sexual content, hate, abuse, "
            "illegal activities, medical harm, personal data leakage).\n"
            "Respond with exactly 'SAFE' if compliant, or 'UNSAFE: <brief reason>'."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_query or ""},
        ]
        reply = self._call_guard(messages, max_tokens=64)
        return self._parse_guard_reply(reply)

    def check_model_answer(self, user_query: str, model_answer: str) -> Tuple[bool, str]:
        """Validate the model's answer is safe in the context of the user question."""
        system = (
            "You are a safety classifier. Given a user question and an assistant answer, "
            "determine if the ASSISTANT answer violates safety policy (e.g., dangerous "
            "medical instructions, illegal guidance, explicit sexual content, hate, abuse, "
            "privacy violations).\n"
            "If safe, answer 'SAFE'. If not, answer 'UNSAFE: <brief reason>'."
        )
        content = (
            f"USER: {user_query or ''}\n\nASSISTANT: {model_answer or ''}"
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]
        reply = self._call_guard(messages, max_tokens=96)
        return self._parse_guard_reply(reply)


# Global instance (optional convenience)
safety_guard = SafetyGuard()


