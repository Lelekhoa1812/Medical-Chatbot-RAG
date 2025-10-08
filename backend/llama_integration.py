import os
import requests
import json
import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)

class NVIDIALLamaClient:
    def __init__(self):
        self.api_key = os.getenv("NVIDIA_URI")
        if not self.api_key:
            raise ValueError("NVIDIA_URI environment variable not set")
        
        self.base_url = "https://api.nvcf.nvidia.com/v2/nvcf/chat/completions"
        self.model = "meta/llama-3.1-8b-instruct"
        
    def generate_keywords(self, user_query: str) -> List[str]:
        """Use Llama to generate search keywords from user query"""
        try:
            prompt = f"""Given this medical question: "{user_query}"

Generate 3-5 specific search keywords that would help find relevant medical information online. 
Focus on medical terms, symptoms, conditions, treatments, or procedures mentioned.
Return only the keywords separated by commas, no explanations.

Keywords:"""

            response = self._call_llama(prompt)
            
            # Extract keywords from response
            keywords = [kw.strip() for kw in response.split(',') if kw.strip()]
            logger.info(f"Generated keywords: {keywords}")
            return keywords[:5]  # Limit to 5 keywords
            
        except Exception as e:
            logger.error(f"Failed to generate keywords: {e}")
            return [user_query]  # Fallback to original query
    
    def summarize_documents(self, documents: List[Dict], user_query: str) -> Tuple[str, Dict[int, str]]:
        """Use Llama to summarize documents and return summary with URL mapping"""
        try:
            # Create document summaries
            doc_summaries = []
            url_mapping = {}
            
            for doc in documents:
                doc_id = doc['id']
                url_mapping[doc_id] = doc['url']
                
                # Create a summary prompt for each document
                summary_prompt = f"""Summarize this medical information in 2-3 sentences, focusing on details relevant to: "{user_query}"

Document: {doc['title']}
Content: {doc['content'][:1000]}...

Summary:"""
                
                summary = self._call_llama(summary_prompt)
                doc_summaries.append(f"Document {doc_id}: {summary}")
            
            # Combine all summaries
            combined_summary = "\n\n".join(doc_summaries)
            
            return combined_summary, url_mapping
            
        except Exception as e:
            logger.error(f"Failed to summarize documents: {e}")
            return "", {}
    
    def _call_llama(self, prompt: str) -> str:
        """Make API call to NVIDIA Llama model"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 1000
            }
            
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            response.raise_for_status()
            result = response.json()
            
            return result['choices'][0]['message']['content'].strip()
            
        except Exception as e:
            logger.error(f"Llama API call failed: {e}")
            raise

def process_search_query(user_query: str, search_results: List[Dict]) -> Tuple[str, Dict[int, str]]:
    """Process search results using Llama model"""
    try:
        llama_client = NVIDIALLamaClient()
        
        # Generate search keywords
        keywords = llama_client.generate_keywords(user_query)
        
        # Summarize documents
        summary, url_mapping = llama_client.summarize_documents(search_results, user_query)
        
        return summary, url_mapping
        
    except Exception as e:
        logger.error(f"Failed to process search query: {e}")
        return "", {}
