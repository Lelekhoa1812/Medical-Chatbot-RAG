import os
import requests
import logging
from typing import List, Dict
import re

logger = logging.getLogger(__name__)

class MedicalReranker:
    """Rerank search results based on medical relevance and source quality"""
    
    def __init__(self):
        self.api_key = os.getenv("FOUNDRY_API_KEY")
        self.endpoint = (os.getenv("FOUNDRY_ENDPOINT") or "").rstrip("/")
        self.api_version = os.getenv("FOUNDRY_API_VERSION", "2024-05-01-preview")
        self.model = os.getenv("SLM_MODEL", "gpt-5-nano")
        self.base_url = self._build_chat_completions_url(self.endpoint) if self.endpoint else ""
        self.timeout = 30
        
        # Medical domain priority scoring
        self.domain_scores = {
            # High priority medical domains
            'mayoclinic.org': 0.95,
            'webmd.com': 0.90,
            'healthline.com': 0.88,
            'medlineplus.gov': 0.95,
            'nih.gov': 0.98,
            'cdc.gov': 0.98,
            'who.int': 0.97,
            'pubmed.ncbi.nlm.nih.gov': 0.96,
            'uptodate.com': 0.94,
            'merckmanuals.com': 0.92,
            'medscape.com': 0.89,
            'clevelandclinic.org': 0.93,
            'hopkinsmedicine.org': 0.94,
            'harvard.edu': 0.96,
            'stanford.edu': 0.95,
            
            # Medium priority
            'youtube.com': 0.60,
            'wikipedia.org': 0.70,
            
            # Low priority (generic health sites)
            'generic_health_site': 0.30
        }
        
        self.irrelevant_patterns = [
            r'quiz|test|assessment|survey',
            r'homepage|main page|index',
            r'login|sign up|register',
            r'contact|about us|privacy',
            r'subscribe|newsletter|rss',
            r'sitemap|search results',
            r'healthtopics\.html',
            r'healthy-sleep/quiz',
        ]

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
    
    def rerank_results(self, query: str, results: List[Dict], min_score: float = 0.05) -> List[Dict]:
        """Rerank search results based on medical relevance"""
        if not results:
            return []
        
        filtered_results = self._filter_irrelevant_results(results)
        if not filtered_results:
            return []
        
        domain_scored = self._score_by_domain(filtered_results)
        
        try:
            semantic_scored = self._semantic_rerank(query, domain_scored)
        except Exception as e:
            logger.warning(f"Semantic reranking failed: {e}")
            semantic_scored = domain_scored
        
        diversity_scored = self._apply_diversity_scoring(semantic_scored)
        
        final_results = [r for r in diversity_scored if r.get('composite_score', 0) >= min_score]
        final_results.sort(key=lambda x: x.get('composite_score', 0), reverse=True)
        
        return final_results

    def _filter_irrelevant_results(self, results: List[Dict]) -> List[Dict]:
        """Filter out obviously irrelevant results"""
        filtered = []
        
        for result in results:
            url = result.get('url', '').lower()
            title = result.get('title', '').lower()
            content = result.get('content', '').lower()
            
            is_irrelevant = False
            for pattern in self.irrelevant_patterns:
                if re.search(pattern, url) or re.search(pattern, title):
                    is_irrelevant = True
                    break
            
            if is_irrelevant:
                logger.debug(f"Filtered irrelevant result: {url}")
                continue
            
            if content and len(content) < 20:
                logger.debug(f"Filtered result with very short content: {url}")
                continue
            
            filtered.append(result)
        
        return filtered
    
    def _score_by_domain(self, results: List[Dict]) -> List[Dict]:
        """Score results based on domain credibility"""
        scored_results = []
        
        for result in results:
            url = result.get('url', '')
            domain = self._extract_domain(url)
            domain_score = self.domain_scores.get(domain, 0.70)
            
            title = result.get('title', '').lower()
            content = result.get('content', '').lower()
            
            medical_boost = 0.0
            medical_keywords = [
                'treatment', 'diagnosis', 'symptoms', 'therapy', 'medication',
                'clinical', 'medical', 'health', 'disease', 'condition'
            ]
            
            for keyword in medical_keywords:
                if keyword in title:
                    medical_boost += 0.05
                if keyword in content[:500]:
                    medical_boost += 0.02
            
            composite_score = min(domain_score + medical_boost, 1.0)
            
            result['domain_score'] = domain_score
            result['medical_boost'] = medical_boost
            result['composite_score'] = composite_score
            result['domain'] = domain
            
            scored_results.append(result)
        
        return scored_results
    
    def _semantic_rerank(self, query: str, results: List[Dict]) -> List[Dict]:
        """Use Azure AI lightweight model for semantic relevance with title prioritization"""
        if not self.api_key or not self.base_url:
            return self._fallback_title_rerank(query, results)
        
        documents = []
        for idx, result in enumerate(results):
            title = result.get('title', '')
            content = result.get('content', '')[:600]
            documents.append(f"[{idx}] TITLE: {title}\nCONTENT: {content}")
        
        if not documents:
            return self._fallback_title_rerank(query, results)
        
        prompt = (
            "You are a medical search reranker. Score each candidate document for how useful it is in answering the user query. "
            "Return strict JSON only as an array of objects with fields index and score, where score is a float between 0 and 1. "
            "Consider medical relevance, factual usefulness, and title alignment.\n\n"
            f"QUERY: {query}\n\n"
            "DOCUMENTS:\n" + "\n\n".join(documents)
        )
        
        try:
            headers = {
                "api-key": self.api_key,
                "Content-Type": "application/json",
            }
            
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a precise JSON-only reranking assistant."},
                    {"role": "user", "content": prompt},
                ],
            }
            
            response = requests.post(
                f"{self.base_url}?api-version={self.api_version}",
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            content = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
            
            match = re.search(r'\[[\s\S]*\]', content)
            if not match:
                return self._fallback_title_rerank(query, results)
            
            import json
            entries = json.loads(match.group(0))
            score_map = {
                int(entry.get("index")): float(entry.get("score", 0.5))
                for entry in entries
                if isinstance(entry, dict) and str(entry.get("index", "")).isdigit()
            }
            
            reranked_results = []
            for i, result in enumerate(results):
                result = result.copy()
                semantic_score = max(0.0, min(score_map.get(i, 0.5), 1.0))
                domain_score = result.get('domain_score', 0.3)
                title_relevance = self._calculate_title_relevance(query, result.get('title', ''))
                final_score = (domain_score * 0.3) + (semantic_score * 0.5) + (title_relevance * 0.2)
                
                result['semantic_score'] = semantic_score
                result['title_relevance'] = title_relevance
                result['composite_score'] = final_score
                reranked_results.append(result)
            
            return reranked_results
            
        except Exception as e:
            logger.warning(f"Azure AI reranking failed: {e}")
            return self._fallback_title_rerank(query, results)
    
    def _fallback_title_rerank(self, query: str, results: List[Dict]) -> List[Dict]:
        """Fallback reranking based on title relevance when AI API fails"""
        query_words = set(query.lower().split())
        
        for result in results:
            title = result.get('title', '').lower()
            title_words = set(title.split())
            
            if query_words and title_words:
                overlap = len(query_words.intersection(title_words))
                title_relevance = overlap / len(query_words)
            else:
                title_relevance = 0.0
            
            if query.lower() in title:
                title_relevance = min(title_relevance + 0.3, 1.0)
            
            domain_score = result.get('domain_score', 0.7)
            result['title_relevance'] = title_relevance
            result['composite_score'] = (domain_score * 0.3) + (title_relevance * 0.7)
        
        return results
    
    def _calculate_title_relevance(self, query: str, title: str) -> float:
        """Calculate relevance score based on title and query matching"""
        if not title or not query:
            return 0.0
        
        query_lower = query.lower()
        title_lower = title.lower()
        
        if query_lower in title_lower:
            return 1.0
        
        query_words = set(query_lower.split())
        title_words = set(title_lower.split())
        
        if not query_words:
            return 0.0
        
        overlap = len(query_words.intersection(title_words))
        base_score = overlap / len(query_words)
        
        medical_terms = ['treatment', 'diagnosis', 'symptoms', 'therapy', 'medication', 'medical', 'health']
        medical_boost = sum(0.15 for term in medical_terms if term in title_lower)
        
        return min(base_score + medical_boost, 1.0)
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            if domain.startswith('www.'):
                domain = domain[4:]
            
            return domain
        except:
            return 'unknown'
    
    def filter_youtube_results(self, results: List[Dict], query: str) -> List[Dict]:
        """Filter and improve YouTube results for medical queries"""
        filtered = []
        
        for result in results:
            url = result.get('url', '')
            title = result.get('title', '')
            
            if 'results?search_query=' in url:
                continue
            
            if not self._is_medical_video(title, query):
                continue
            
            video_id = self._extract_video_id(url)
            if video_id:
                result['url'] = f"https://www.youtube.com/watch?v={video_id}"
                result['video_id'] = video_id
                result['source_type'] = 'video'
                filtered.append(result)
        
        return filtered
    
    def _is_medical_video(self, title: str, query: str) -> bool:
        """Check if video title is medically relevant"""
        title_lower = title.lower()
        query_lower = query.lower()
        
        medical_keywords = [
            'medical', 'health', 'doctor', 'treatment', 'diagnosis',
            'symptoms', 'therapy', 'medicine', 'clinical', 'patient'
        ]
        
        has_medical = any(keyword in title_lower for keyword in medical_keywords)
        has_query = any(word in title_lower for word in query_lower.split())
        
        return has_medical or has_query
    
    def _extract_video_id(self, url: str) -> str:
        """Extract YouTube video ID from URL"""
        patterns = [
            r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
            r'(?:embed\/)([0-9A-Za-z_-]{11})',
            r'(?:watch\?v=)([0-9A-Za-z_-]{11})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        return None
    
    def _apply_diversity_scoring(self, results: List[Dict]) -> List[Dict]:
        """Apply diversity scoring to avoid too many results from same domain"""
        if not results:
            return results
        
        from urllib.parse import urlparse
        from collections import defaultdict
        
        domain_counts = defaultdict(int)
        max_per_domain = 3
        
        for result in results:
            url = result.get('url', '')
            try:
                domain = urlparse(url).netloc.lower()
                if domain.startswith('www.'):
                    domain = domain[4:]
                
                domain_counts[domain] += 1
                
                if domain_counts[domain] > max_per_domain:
                    current_score = result.get('composite_score', 0)
                    penalty = 0.15 * (domain_counts[domain] - max_per_domain)
                    result['composite_score'] = max(0, current_score - penalty)
                    result['diversity_penalty'] = penalty
                    logger.debug(f"Applied diversity penalty {penalty} to {domain}")
                else:
                    result['diversity_penalty'] = 0
                    
            except Exception as e:
                logger.debug(f"Error parsing domain for diversity scoring: {e}")
                result['diversity_penalty'] = 0
        
        total_domains = len(domain_counts)
        over_represented = sum(1 for count in domain_counts.values() if count > max_per_domain)
        logger.info(f"Diversity scoring: {total_domains} domains, {over_represented} over-represented")
        
        return results
