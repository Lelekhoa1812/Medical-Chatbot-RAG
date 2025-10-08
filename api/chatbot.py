# api/chatbot.py
import re
import logging
from typing import Dict
from google import genai
from api.config import gemini_flash_api_key
from api.retrieval import retrieval_engine
from memory import MemoryManager
from utils import translate_query, process_medical_image
from search import search_web
from models import process_search_query

logger = logging.getLogger("medical-chatbot")

class GeminiClient:
    """Gemini API client for generating responses"""
    
    def __init__(self):
        self.client = genai.Client(api_key=gemini_flash_api_key)
    
    def generate_content(self, prompt: str, model: str = "gemini-2.5-flash", temperature: float = 0.7) -> str:
        """Generate content using Gemini API"""
        try:
            response = self.client.models.generate_content(model=model, contents=prompt)
            return response.text
        except Exception as e:
            logger.error(f"[LLM] ❌ Error calling Gemini API: {e}")
            return "Error generating response from Gemini."

class RAGMedicalChatbot:
    """Main chatbot class with RAG capabilities"""
    
    def __init__(self, model_name: str, retrieve_function):
        self.model_name = model_name
        self.retrieve = retrieve_function
        self.gemini_client = GeminiClient()
        self.memory = MemoryManager()
    
    def chat(self, user_id: str, user_query: str, lang: str = "EN", image_diagnosis: str = "", search_mode: bool = False) -> str:
        """Main chat method with RAG and search capabilities"""
        
        # 0. Translate query if not EN, this help our RAG system
        if lang.upper() in {"VI", "ZH"}:
            user_query = translate_query(user_query, lang.lower())

        # 1. Fetch knowledge
        ## a. KB for generic QA retrieval
        retrieved_info = self.retrieve(user_query)
        knowledge_base = "\n".join(retrieved_info)
        ## b. Diagnosis RAG from symptom query
        diagnosis_guides = retrieval_engine.retrieve_diagnosis_from_symptoms(user_query)
        
        # 1.5. Search mode - web search and Llama processing
        search_context = ""
        url_mapping = {}
        if search_mode:
            logger.info(f"[SEARCH] Starting web search mode for query: {user_query}")
            try:
                # Search the web with max 10 resources
                search_results = search_web(user_query, num_results=10)
                if search_results:
                    logger.info(f"[SEARCH] Retrieved {len(search_results)} web resources")
                    # Process with Llama
                    search_context, url_mapping = process_search_query(user_query, search_results)
                    logger.info(f"[SEARCH] Processed with Llama, generated {len(url_mapping)} URL mappings")
                else:
                    logger.warning("[SEARCH] No search results found")
            except Exception as e:
                logger.error(f"[SEARCH] Search failed: {e}")
                search_context = ""

        # 2. Hybrid Context Retrieval: RAG + Recent History + Intelligent Selection
        contextual_chunks = self.memory.get_contextual_chunks(user_id, user_query, lang)

        # 3. Build prompt parts
        parts = ["You are a medical chatbot, designed to answer medical questions."]
        parts.append("Please format your answer using MarkDown.")
        parts.append("**Bold for titles**, *italic for emphasis*, and clear headings.")
        
        # 4. Append image diagnosis from VLM
        if image_diagnosis:
            parts.append(
                "A user medical image is diagnosed by our VLM agent:\n"
                f"{image_diagnosis}\n\n"
                "Please incorporate the above findings in your response if medically relevant.\n\n"
            )
        
        # Append contextual chunks from hybrid approach
        if contextual_chunks:
            parts.append("Relevant context from conversation history:\n" + contextual_chunks)
        # Load up guideline (RAG over medical knowledge base)
        if knowledge_base:
            parts.append(f"Example Q&A medical scenario knowledge-base: {knowledge_base}")
        # Symptom-Diagnosis prediction RAG
        if diagnosis_guides:
            parts.append("Symptom-based diagnosis guidance (if applicable):\n" + "\n".join(diagnosis_guides))
        
        # 5. Search context with citation instructions
        if search_context:
            parts.append("Additional information from web search:\n" + search_context)
            parts.append("IMPORTANT: When you use information from the web search results above, you MUST add a citation tag <#ID> immediately after the relevant content, where ID is the document number (1, 2, 3, etc.). For example: 'According to recent studies <#1>, this condition affects...'")
        
        parts.append(f"User's question: {user_query}")
        parts.append(f"Language to generate answer: {lang}")
        prompt = "\n\n".join(parts)
        logger.info(f"[LLM] Question query in `prompt`: {prompt}") # Debug out checking RAG on kb and history
        response = self.gemini_client.generate_content(prompt, model=self.model_name, temperature=0.7)
        
        # 6. Process citations and replace with URLs
        if search_mode and url_mapping:
            response = self._process_citations(response, url_mapping)
        
         # Store exchange + chunking
        if user_id:
            self.memory.add_exchange(user_id, user_query, response, lang=lang)
        logger.info(f"[LLM] Response on `prompt`: {response.strip()}") # Debug out base response
        return response.strip()
    
    def _process_citations(self, response: str, url_mapping: Dict[int, str]) -> str:
        """Replace citation tags with actual URLs"""
        
        # Find all citation tags like <#1>, <#2>, etc.
        citation_pattern = r'<#(\d+)>'
        citations_found = re.findall(citation_pattern, response)
        
        def replace_citation(match):
            doc_id = int(match.group(1))
            if doc_id in url_mapping:
                url = url_mapping[doc_id]
                logger.info(f"[CITATION] Replacing <#{doc_id}> with {url}")
                return f'<{url}>'
            else:
                logger.warning(f"[CITATION] No URL mapping found for document ID {doc_id}")
                return match.group(0)  # Keep original if URL not found
        
        # Replace citations with URLs
        processed_response = re.sub(citation_pattern, replace_citation, response)
        
        logger.info(f"[CITATION] Processed {len(citations_found)} citations, {len(url_mapping)} URL mappings available")
        return processed_response
