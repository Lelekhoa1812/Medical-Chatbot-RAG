import os
import requests
import logging
import time
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)


class AzureAIClient:
    def __init__(self, model_env_var: str = "LLM_MODEL", default_model: str = "gpt-5.4"):
        self.api_key = os.getenv("FOUNDRY_API_KEY")
        self.endpoint = (os.getenv("FOUNDRY_ENDPOINT") or "").rstrip("/")
        self.api_version = os.getenv("FOUNDRY_API_VERSION", "2024-05-01-preview")
        self.model = os.getenv(model_env_var, default_model)

        if not self.api_key:
            raise ValueError("FOUNDRY_API_KEY environment variable not set")
        if not self.endpoint:
            raise ValueError("FOUNDRY_ENDPOINT environment variable not set")

        self.base_url = self._build_chat_completions_url(self.endpoint)

    @staticmethod
    def _build_chat_completions_url(endpoint: str) -> str:
        endpoint = endpoint.rstrip("/")
        if endpoint.endswith("/chat/completions"):
            return endpoint
        if "/openai/deployments/" in endpoint:
            return f"{endpoint}/chat/completions"
        if endpoint.endswith("/openai/v1"):
            return f"{endpoint}/chat/completions"
        if endpoint.endswith("/openai"):
            return f"{endpoint}/chat/completions"
        return f"{endpoint}/openai/chat/completions"

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        timeout: int = 30,
        max_retries: int = 3,
        model: Optional[str] = None,
    ) -> str:
        for attempt in range(max_retries):
            try:
                headers = {
                    "Content-Type": "application/json",
                    "api-key": self.api_key,
                }

                payload = {
                    "model": model or self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }

                response = requests.post(
                    f"{self.base_url}?api-version={self.api_version}",
                    headers=headers,
                    json=payload,
                    timeout=timeout,
                )
                response.raise_for_status()
                result = response.json()

                content = (
                    result.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                    .strip()
                )
                if not content:
                    raise ValueError("Empty response from Azure AI chat completion API")

                return content

            except requests.exceptions.Timeout:
                logger.warning(
                    f"Azure AI timeout (attempt {attempt + 1}/{max_retries})"
                )
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)

            except requests.exceptions.RequestException as e:
                logger.warning(
                    f"Azure AI request failed (attempt {attempt + 1}/{max_retries}): {e}"
                )
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)

            except Exception as e:
                logger.error(f"Azure AI chat completion failed: {e}")
                raise


class AzureLLMClient:
    def __init__(self):
        self.client = AzureAIClient(model_env_var="LLM_MODEL", default_model="gpt-5.4")

    def generate_keywords(self, user_query: str) -> List[str]:
        """Use Azure AI LLM to generate search keywords from user query"""
        try:
            prompt = f"""Given this medical question: \"{user_query}\"

Generate 3-5 specific search keywords that would help find relevant medical information online.
Focus on medical terms, symptoms, conditions, treatments, or procedures mentioned.
Return only the keywords separated by commas, no explanations.

Keywords:"""

            response = self._call_llm(prompt)
            keywords = [kw.strip() for kw in response.split(',') if kw.strip()]
            logger.info(f"Generated keywords: {keywords}")
            return keywords[:5]

        except Exception as e:
            logger.error(f"Failed to generate keywords: {e}")
            return [user_query]

    def summarize_documents(self, documents: List[Dict], user_query: str) -> Tuple[str, Dict[int, str]]:
        """Use Azure AI LLM to summarize documents and return summary with URL mapping"""
        try:
            from .summarizer import summarizer

            combined_summary, url_mapping = summarizer.summarize_documents(documents, user_query)
            return combined_summary, url_mapping

        except Exception as e:
            logger.error(f"Failed to summarize documents: {e}")
            return "", {}

    def _call_llm(self, prompt: str, max_retries: int = 3) -> str:
        """Make API call to Azure AI LLM"""
        return self.client.chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0.7,
            max_tokens=1000,
            timeout=30,
            max_retries=max_retries,
        )


# Backwards-compatible alias for existing imports
NVIDIALLamaClient = AzureLLMClient


def process_search_query(user_query: str, search_results: List[Dict]) -> Tuple[str, Dict[int, str]]:
    """Process search results using Azure AI LLM"""
    try:
        llm_client = AzureLLMClient()

        keywords = llm_client.generate_keywords(user_query)
        logger.debug(f"Search keywords for query processing: {keywords}")

        summary, url_mapping = llm_client.summarize_documents(search_results, user_query)
        return summary, url_mapping

    except Exception as e:
        logger.error(f"Failed to process search query: {e}")
        return "", {}
