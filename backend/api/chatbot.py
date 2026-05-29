"""Chatbot wiring for the medical RAG API.

Root Cause vs Logic: startup failed because this layer passed an `api_key` argument into `AzureLLMClient`, but the shared Azure client already reads `FOUNDRY_API_KEY` / `FOUNDRY_ENDPOINT` from the environment. Keep auth configuration centralized in the client and only pass runtime behavior options from the chatbot layer.
"""

from __future__ import annotations

import logging
from typing import Any

from models import AzureLLMClient

logger = logging.getLogger(__name__)


class RAGMedicalChatbot:
    def __init__(self, *args: Any, **kwargs: Any):
        """Initialize the chatbot and its shared Azure LLM client.

        Motivation vs Logic: the chatbot should stay thin and reuse the common
        Azure client so there is a single provider configuration path across the
        application.
        """
        self.llm_client = AzureLLMClient()
        self._init_args = args
        self._init_kwargs = kwargs
        logger.info("RAGMedicalChatbot initialized with shared AzureLLMClient")

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(name)
