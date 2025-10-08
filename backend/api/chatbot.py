# api/chatbot.py
import re
import logging
from typing import Dict
from google import genai
from .config import gemini_flash_api_key
from .retrieval import retrieval_engine
from memory import MemoryManager
from utils import translate_query, process_medical_image
from search import search_web
from models import summarizer
from models import process_search_query
from models.guard import safety_guard

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

        # 0.1 Safety check on user query
        is_safe_user, reason_user = safety_guard.check_user_query(user_query or "")
        if not is_safe_user:
            logger.warning(f"[SAFETY] Blocked unsafe user query: {reason_user}")
            return "⚠️ Unable to process this request safely. Please rephrase your question."

        # 1. Fetch knowledge
        ## a. KB for generic QA retrieval
        retrieved_info = self.retrieve(user_query)
        knowledge_base = "\n".join(retrieved_info)
        ## b. Diagnosis RAG from symptom query
        diagnosis_guides = retrieval_engine.retrieve_diagnosis_from_symptoms(user_query)
        # c. Hybrid Context Retrieval: RAG + Recent History + Intelligent Selection
        contextual_chunks = self.memory.get_contextual_chunks(user_id, user_query, lang)

        # 2. Search mode - web search and Llama processing
        search_context = ""
        url_mapping = {}
        if search_mode:
            logger.info(f"[SEARCH] Starting web search mode for query: {user_query}")
            try:
                # Build enriched search query from user's question + recent memory (no KB retrieval)
                recent_memory_chunk = self.memory.get_context(user_id, num_turns=3) or ""
                # If contextual_chunks available, use it instead of recent_memory_chunk
                recent_memory_ctx = contextual_chunks if contextual_chunks else recent_memory_chunk[:600]
                memory_focus = summarizer.summarize_for_query(recent_memory_ctx, user_query, max_length=180) if recent_memory_ctx else ""
                final_search_query = user_query if not memory_focus else f"{user_query}. {memory_focus}"
                logger.info(f"[SEARCH] Final search query: {final_search_query}")

                # Search the web with max 10 resources using enriched query
                search_results = search_web(final_search_query, num_results=10)
                if search_results:
                    logger.info(f"[SEARCH] Retrieved {len(search_results)} web resources")
                    # Compose a compact context strictly from query-relevant snippets
                    # Also build URL mapping for citations
                    url_mapping = {doc['id']: doc['url'] for doc in search_results if doc.get('id') and doc.get('url')}
                    # Create per-doc short summaries (already relevant snippets), further compress
                    summaries = []
                    for doc in search_results:
                        content = (doc.get('content') or '').strip()
                        if not content:
                            continue
                        concise = summarizer.summarize_for_query(content, user_query, max_length=320)
                        if concise:
                            summaries.append(f"Document {doc['id']}: {concise}")
                    search_context = "\n\n".join(summaries)
                    logger.info(f"[SEARCH] Processed with Llama, generated {len(url_mapping)} URL mappings")
                else:
                    logger.warning("[SEARCH] No search results found")
            except Exception as e:
                logger.error(f"[SEARCH] Search failed: {e}")
                search_context = ""

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
            parts.append("IMPORTANT: When you use information from the web search results above, you MUST add citation tags immediately after the relevant content. Use single citations like <#1> or multiple citations like <#1, #2, #5> when information comes from multiple sources. For example: 'According to recent studies <#1>, this condition affects...' or 'Multiple sources <#1, #3, #7> suggest that...'")
        
        parts.append(f"User's question: {user_query}")
        parts.append(f"Language to generate answer: {lang}")
        prompt = "\n\n".join(parts)
        logger.info(f"[LLM] Question query in `prompt`: {prompt}") # Debug out checking RAG on kb and history
        response = self.gemini_client.generate_content(prompt, model=self.model_name, temperature=0.7)
        
        # 6. Process citations and replace with URLs
        if search_mode and url_mapping:
            response = self._process_citations(response, url_mapping)
        
        # 7. Safety check on model answer
        is_safe_ans, reason_ans = safety_guard.check_model_answer(user_query, response or "")
        if not is_safe_ans:
            logger.warning(f"[SAFETY] Withholding unsafe model answer: {reason_ans}")
            response = "⚠️ I cannot share that information. Let's discuss this topic at a high level or try a different question."

         # Store exchange + chunking
        if user_id:
            self.memory.add_exchange(user_id, user_query, response, lang=lang)
        logger.info(f"[LLM] Response on `prompt`: {response.strip()}") # Debug out base response
        return response.strip()
    
    def _process_citations(self, response: str, url_mapping: Dict[int, str]) -> str:
        """Replace citation tags with actual URLs, handling both single and multiple references"""
        
        # Pattern to match both single citations <#1> and multiple citations <#1, #2, #5, #7, #9>
        citation_pattern = r'<#([^>]+)>'
        
        def replace_citation(match):
            citation_content = match.group(1)
            # Split by comma and clean up each citation ID
            citation_ids = [id_str.strip() for id_str in citation_content.split(',')]
            
            urls = []
            for citation_id in citation_ids:
                try:
                    doc_id = int(citation_id)
                    if doc_id in url_mapping:
                        url = url_mapping[doc_id]
                        urls.append(f'<{url}>')
                        logger.info(f"[CITATION] Replacing <#{doc_id}> with {url}")
                    else:
                        logger.warning(f"[CITATION] No URL mapping found for document ID {doc_id}")
                        urls.append(f'<#{doc_id}>')  # Keep original if URL not found
                except ValueError:
                    logger.warning(f"[CITATION] Invalid citation ID: {citation_id}")
                    urls.append(f'<#{citation_id}>')  # Keep original if invalid
            
            # Join multiple URLs with spaces
            return ' '.join(urls)
        
        # Replace citations with URLs
        processed_response = re.sub(citation_pattern, replace_citation, response)
        
        # Count total citations processed
        citations_found = re.findall(citation_pattern, response)
        total_citations = sum(len([id_str.strip() for id_str in citation_content.split(',')]) 
                            for citation_content in citations_found)
        
        logger.info(f"[CITATION] Processed {total_citations} citations from {len(citations_found)} citation groups, {len(url_mapping)} URL mappings available")
        return processed_response
