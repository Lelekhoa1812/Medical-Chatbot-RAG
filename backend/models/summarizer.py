import re
import logging
from typing import List, Dict, Tuple
from .llama import NVIDIALLamaClient

logger = logging.getLogger(__name__)

class TextSummarizer:
    def __init__(self):
        self.llama_client = NVIDIALLamaClient()
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text for summarization"""
        if not text:
            return ""
        
        # Remove common conversation starters and fillers
        conversation_patterns = [
            r'\b(hi|hello|hey|sure|okay|yes|no|thanks|thank you)\b',
            r'\b(here is|this is|let me|i will|i can|i would)\b',
            r'\b(summarize|summary|here\'s|here is)\b',
            r'\b(please|kindly|would you|could you)\b',
            r'\b(um|uh|er|ah|well|so|like|you know)\b'
        ]
        
        # Remove excessive whitespace and normalize
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', ' ', text)
        
        # Remove conversation patterns
        for pattern in conversation_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Remove extra punctuation and normalize
        text = re.sub(r'[.]{2,}', '.', text)
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        
        return text.strip()
    
    def extract_key_phrases(self, text: str) -> List[str]:
        """Extract key medical phrases and terms"""
        if not text:
            return []
        
        # Medical term patterns
        medical_patterns = [
            r'\b(?:symptoms?|diagnosis|treatment|therapy|medication|drug|disease|condition|syndrome)\b',
            r'\b(?:patient|doctor|physician|medical|clinical|healthcare)\b',
            r'\b(?:blood pressure|heart rate|temperature|pulse|respiration)\b',
            r'\b(?:acute|chronic|severe|mild|moderate|serious|critical)\b',
            r'\b(?:pain|ache|discomfort|swelling|inflammation|infection)\b'
        ]
        
        key_phrases = []
        for pattern in medical_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            key_phrases.extend(matches)
        
        return list(set(key_phrases))  # Remove duplicates
    
    def summarize_text(self, text: str, max_length: int = 200) -> str:
        """Summarize text using NVIDIA Llama model"""
        try:
            if not text or len(text.strip()) < 50:
                return text
            
            # Clean the text first
            cleaned_text = self.clean_text(text)
            
            # Extract key phrases for context
            key_phrases = self.extract_key_phrases(cleaned_text)
            key_phrases_str = ", ".join(key_phrases[:5]) if key_phrases else "medical information"
            
            # Create optimized prompt
            prompt = f"""Summarize this medical text in {max_length} characters or less. Focus only on key medical facts, symptoms, treatments, and diagnoses. Do not include greetings, confirmations, or conversational elements.

Key terms: {key_phrases_str}

Text: {cleaned_text[:1500]}

Summary:"""

            summary = self.llama_client._call_llama(prompt)
            
            # Post-process summary
            summary = self.clean_text(summary)
            
            # Ensure it's within length limit
            if len(summary) > max_length:
                summary = summary[:max_length-3] + "..."
            
            return summary
            
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            # Fallback to simple truncation
            return self.clean_text(text)[:max_length]

    def summarize_for_query(self, text: str, query: str, max_length: int = 220) -> str:
        """Summarize text focusing strictly on information relevant to the query.
        Returns an empty string if nothing relevant is found.
        """
        try:
            if not text:
                return ""
            cleaned_text = self.clean_text(text)
            if not cleaned_text:
                return ""

            # Short, strict prompt to avoid verbosity; instruct to output NOTHING if irrelevant
            prompt = (
                f"You extract only medically relevant facts that help answer: '{query}'. "
                f"Respond with a concise bullet list (<= {max_length} chars total). "
                "If the content is irrelevant, respond with EXACTLY: NONE.\n\n"
                f"Content: {cleaned_text[:1600]}\n\nRelevant facts:"
            )

            summary = self.llama_client._call_llama(prompt)
            summary = self.clean_text(summary)
            if not summary or summary.upper().strip() == "NONE":
                return ""
            if len(summary) > max_length:
                summary = summary[:max_length-3] + "..."
            return summary
        except Exception as e:
            logger.warning(f"Query-focused summarization failed: {e}")
            return ""
    
    def summarize_documents(self, documents: List[Dict], user_query: str) -> Tuple[str, Dict[int, str]]:
        """Summarize multiple documents with URL mapping"""
        try:
            doc_summaries = []
            url_mapping = {}
            
            for doc in documents:
                doc_id = doc['id']
                url_mapping[doc_id] = doc['url']
                
                # Create focused summary for each document
                summary_prompt = f"""Summarize this medical document in 2-3 sentences, focusing on information relevant to: "{user_query}"

Document: {doc['title']}
Content: {doc['content'][:800]}

Key medical information:"""

                summary = self.llama_client._call_llama(summary_prompt)
                summary = self.clean_text(summary)
                
                doc_summaries.append(f"Document {doc_id}: {summary}")
            
            combined_summary = "\n\n".join(doc_summaries)
            return combined_summary, url_mapping
            
        except Exception as e:
            logger.error(f"Document summarization failed: {e}")
            return "", {}
    
    def summarize_conversation_chunk(self, chunk: str) -> str:
        """Summarize a conversation chunk for memory"""
        try:
            if not chunk or len(chunk.strip()) < 30:
                return chunk
            
            cleaned_chunk = self.clean_text(chunk)
            
            prompt = f"""Summarize this medical conversation in 1-2 sentences. Focus only on medical facts, symptoms, treatments, or diagnoses discussed. Remove greetings and conversational elements.

Conversation: {cleaned_chunk[:1000]}

Medical summary:"""

            summary = self.llama_client._call_llama(prompt)
            return self.clean_text(summary)
            
        except Exception as e:
            logger.error(f"Conversation summarization failed: {e}")
            return self.clean_text(chunk)[:150]
    
    def chunk_response(self, response: str, max_chunk_size: int = 500) -> List[str]:
        """Split response into chunks and summarize each"""
        try:
            if not response or len(response) <= max_chunk_size:
                return [response]
            
            # Split by sentences first
            sentences = re.split(r'[.!?]+', response)
            chunks = []
            current_chunk = ""
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                # Check if adding this sentence would exceed limit
                if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
                    chunks.append(self.summarize_conversation_chunk(current_chunk))
                    current_chunk = sentence
                else:
                    current_chunk += sentence + ". "
            
            # Add the last chunk
            if current_chunk:
                chunks.append(self.summarize_conversation_chunk(current_chunk))
            
            return chunks
            
        except Exception as e:
            logger.error(f"Response chunking failed: {e}")
            return [response]

# Global summarizer instance
summarizer = TextSummarizer()
