import logging
from typing import List, Dict, Tuple
import time
import hashlib
from .engines.duckduckgo import DuckDuckGoEngine
from .engines.video import VideoSearchEngine
from .coordinator import SearchCoordinator
from models.reranker import MedicalReranker
from models import summarizer

logger = logging.getLogger(__name__)

# Global instances
_duckduckgo_engine = None
_video_engine = None
_reranker = None
_search_coordinator = None

# Simple in-memory cache for search results
_search_cache = {}
_cache_ttl = 300  # 5 minutes TTL

def get_duckduckgo_engine() -> DuckDuckGoEngine:
    """Get or create the global DuckDuckGo engine instance"""
    global _duckduckgo_engine
    if _duckduckgo_engine is None:
        _duckduckgo_engine = DuckDuckGoEngine()
    return _duckduckgo_engine

def get_video_engine() -> VideoSearchEngine:
    """Get or create the global video engine instance"""
    global _video_engine
    if _video_engine is None:
        _video_engine = VideoSearchEngine()
    return _video_engine

def get_reranker() -> MedicalReranker:
    """Get or create the global reranker instance"""
    global _reranker
    if _reranker is None:
        _reranker = MedicalReranker()
    return _reranker

def get_search_coordinator() -> SearchCoordinator:
    """Get or create the global search coordinator instance"""
    global _search_coordinator
    if _search_coordinator is None:
        _search_coordinator = SearchCoordinator()
    return _search_coordinator

def _clean_search_query(query: str) -> str:
    """Clean search query by removing bullet points and special characters"""
    if not query:
        return ""
    
    import re
    # Remove bullet points and special characters
    cleaned = re.sub(r'[•·▪▫‣⁃]', ' ', query)
    cleaned = re.sub(r'[^\w\s\-\.]', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = cleaned.strip()
    
    # Remove common prefixes that might confuse search
    prefixes_to_remove = [
        r'^(en|vi|zh)\s*:\s*',
        r'^(search|find|look for)\s+',
        r'^(how to|what is|what are)\s+',
    ]
    
    for prefix in prefixes_to_remove:
        cleaned = re.sub(prefix, '', cleaned, flags=re.IGNORECASE)
    
    return cleaned.strip()

def _boost_medical_keywords(query: str) -> str:
    """Add medical context keywords to improve search relevance"""
    if not query:
        return ""
    
    # Medical keywords that boost relevance
    medical_boosters = [
        'medical', 'health', 'treatment', 'therapy', 'diagnosis', 'symptoms',
        'clinical', 'patient', 'disease', 'condition', 'medicine', 'healthcare'
    ]
    
    query_lower = query.lower()
    
    # If query doesn't contain medical terms, add context
    has_medical = any(term in query_lower for term in medical_boosters)
    
    if not has_medical:
        # Add medical context without being too verbose
        if len(query.split()) <= 3:
            return f"{query} medical treatment"
        else:
            return f"{query} healthcare"
    
    return query

def _get_cache_key(query: str, num_results: int, target_language: str = None, include_videos: bool = True) -> str:
    """Generate cache key for search results"""
    cache_data = f"{query}_{num_results}_{target_language}_{include_videos}"
    return hashlib.md5(cache_data.encode()).hexdigest()

def _get_cached_results(cache_key: str) -> Tuple[str, Dict[int, str], Dict]:
    """Get cached search results if available and not expired"""
    if cache_key not in _search_cache:
        return None, None, None
    
    cached_data = _search_cache[cache_key]
    if time.time() - cached_data['timestamp'] > _cache_ttl:
        # Cache expired
        del _search_cache[cache_key]
        return None, None, None
    
    logger.info(f"Using cached search results for key: {cache_key[:8]}...")
    return cached_data['search_context'], cached_data['url_mapping'], cached_data['source_aggregation']

def _cache_results(cache_key: str, search_context: str, url_mapping: Dict[int, str], source_aggregation: Dict):
    """Cache search results"""
    _search_cache[cache_key] = {
        'search_context': search_context,
        'url_mapping': url_mapping,
        'source_aggregation': source_aggregation,
        'timestamp': time.time()
    }
    logger.info(f"Cached search results for key: {cache_key[:8]}...")

class WebSearcher:
    """Legacy wrapper for backward compatibility"""
    def __init__(self):
        self.coordinator = get_search_coordinator()
        self.max_results = 10
        self.timeout = 10
        
    def search_google(self, query: str, num_results: int = 10) -> List[Dict]:
        """Search using the new coordinator system"""
        try:
            cleaned_query = _clean_search_query(query)
            return self.coordinator.quick_search(cleaned_query, num_results)
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def search_duckduckgo(self, query: str, num_results: int = 10) -> List[Dict]:
        """Search using DuckDuckGo engine"""
        try:
            cleaned_query = _clean_search_query(query)
            return self.coordinator.quick_search(cleaned_query, num_results)
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
            return []
    
    def extract_content(self, url: str) -> str:
        """Extract content using the new content extractor"""
        try:
            return self.coordinator.content_extractor.extract(url)
        except Exception as e:
            logger.error(f"Content extraction failed: {e}")
            return ""
    
    def search_and_extract(self, query: str, num_results: int = 10) -> List[Dict]:
        """Search and extract content using the new system"""
        try:
            # Clean the query first
            cleaned_query = _clean_search_query(query)
            # Get search results
            results = self.coordinator.quick_search(cleaned_query, num_results)
            
            # Extract content for each result
            enriched_results = []
            for result in results:
                content = self.extract_content(result['url'])
                if content:
                    enriched_result = result.copy()
                    enriched_result['content'] = content
                    enriched_results.append(enriched_result)
            return enriched_results
        except Exception as e:
            logger.error(f"Search and extract failed: {e}")
            return []
        
# Main search function for backward compatibility
def search_web(query: str, num_results: int = 10) -> List[Dict]:
    """Main search function using the new coordinator system"""
    try:
        # Clean the query first
        cleaned_query = _clean_search_query(query)
        coordinator = get_search_coordinator()
        return coordinator.quick_search(cleaned_query, num_results)
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return []

# Enhanced search function with content extraction
def search_web_with_content(query: str, num_results: int = 10) -> Tuple[str, Dict[int, str]]:
    """Enhanced search with content extraction and summarization"""
    try:
        # Clean the query first
        cleaned_query = _clean_search_query(query)
        coordinator = get_search_coordinator()
        return coordinator.search(cleaned_query, num_results)
    except Exception as e:
        logger.error(f"Enhanced web search failed: {e}")
        return "", {}

# Medical-focused search function
def search_medical(query: str, num_results: int = 8) -> Tuple[str, Dict[int, str]]:
    """Medical-focused search with enhanced processing"""
    try:
        # Clean the query first
        cleaned_query = _clean_search_query(query)
        coordinator = get_search_coordinator()
        return coordinator.medical_focus_search(cleaned_query, num_results)
    except Exception as e:
        logger.error(f"Medical search failed: {e}")
        return "", {}

# Multilingual medical search function
def search_multilingual_medical(query: str, num_results: int = 10, target_language: str = None) -> Tuple[str, Dict[int, str]]:
    """Comprehensive multilingual medical search supporting English, Vietnamese, and Chinese"""
    try:
        # Clean the query first
        cleaned_query = _clean_search_query(query)
        coordinator = get_search_coordinator()
        return coordinator.multilingual_medical_search(cleaned_query, num_results, target_language)
    except Exception as e:
        logger.error(f"Multilingual medical search failed: {e}")
        return "", {}

# Video search function
def search_videos(query: str, num_results: int = 2, target_language: str = None) -> List[Dict]:
    """Search for medical videos across multiple platforms"""
    try:
        # Clean the query first
        cleaned_query = _clean_search_query(query)
        coordinator = get_search_coordinator()
        return coordinator.video_search(cleaned_query, num_results, target_language)
    except Exception as e:
        logger.error(f"Video search failed: {e}")
        return []

# Comprehensive search function with maximum information extraction
def search_comprehensive(query: str, num_results: int = 15, target_language: str = None, include_videos: bool = True) -> Tuple[str, Dict[int, str], Dict]:
    """Comprehensive search with maximum information extraction and detailed references"""
    logger.info(f"Starting comprehensive search for: {query} (target: {target_language})")
    
    # Check cache first
    cache_key = _get_cache_key(query, num_results, target_language, include_videos)
    cached_context, cached_mapping, cached_aggregation = _get_cached_results(cache_key)
    if cached_context is not None:
        return cached_context, cached_mapping, cached_aggregation
    
    # Clean and boost the query for better medical relevance
    cleaned_query = _clean_search_query(query)
    boosted_query = _boost_medical_keywords(cleaned_query)
    logger.info(f"Query processing: '{query}' -> '{cleaned_query}' -> '{boosted_query}'")
    
    # Get engines
    duckduckgo_engine = get_duckduckgo_engine()
    video_engine = get_video_engine()
    reranker = get_reranker()
    
    # Optimized search strategy: get just enough results for good filtering
    # Calculate optimal initial count based on expected filtering ratio
    expected_filter_ratio = 0.4  # Expect to keep ~40% after filtering
    optimal_initial_count = max(num_results * 2, int(num_results / expected_filter_ratio))
    
    # Search for text results with optimized count
    text_results = duckduckgo_engine.search(boosted_query, optimal_initial_count)
    logger.info(f"Found {len(text_results)} text results (requested {optimal_initial_count})")
    
    # If no text results, try simple fallback search
    if not text_results:
        logger.warning("No text results found, trying simple fallback search")
        try:
            # Try with a very simple query
            simple_query = " ".join(cleaned_query.split()[:3])  # First 3 words only
            text_results = duckduckgo_engine.search(simple_query, num_results)
            logger.info(f"Simple fallback found {len(text_results)} results")
        except Exception as e:
            logger.warning(f"Simple fallback search failed: {e}")
    
    # Search for videos if requested (limit to avoid over-fetching)
    video_results = []
    if include_videos:
        try:
            # Map language codes for video search
            lang_mapping = {
                'EN': 'en',
                'VI': 'vi', 
                'ZH': 'zh',
                'en': 'en',
                'vi': 'vi',
                'zh': 'zh'
            }
            search_language = lang_mapping.get(target_language, 'en')
            # Limit video results to avoid over-fetching
            max_video_results = min(5, num_results // 3)  # Max 5 or 1/3 of total
            video_results = video_engine.search(boosted_query, num_results=max_video_results, language=search_language)
            logger.info(f"Found {len(video_results)} video results")
        except Exception as e:
            logger.warning(f"Video search failed: {e}")
    
    # Combine all results
    all_results = text_results + video_results
    
    # Use reranker to improve overall quality and relevance
    if all_results:
        reranked_results = reranker.rerank_results(boosted_query, all_results, min_score=0.1)  # Much more lenient
        logger.info(f"Reranked {len(all_results)} total results to {len(reranked_results)} high-quality results")
        
        # If reranking filtered out too many results, use original results
        if len(reranked_results) < max(1, len(all_results) // 4):  # If less than 25% remain
            logger.warning(f"Reranking too strict ({len(reranked_results)}/{len(all_results)}), using original results")
            all_results = all_results[:num_results]  # Just take top N original results
        else:
            all_results = reranked_results
    
    # Limit final results to requested count
    all_results = all_results[:num_results]
    
    # Final safety check - ensure we have at least some results
    if not all_results and text_results:
        logger.warning("No results after processing, using original text results as fallback")
        all_results = text_results[:num_results]
    
    # Create URL mapping
    url_mapping = {}
    for i, result in enumerate(all_results, 1):
        url_mapping[i] = result.get('url', '')
    
    # Create search context using summarizer (only for top results)
    search_context = ""
    if all_results:
        summaries = []
        # Only summarize top results to avoid over-processing
        top_results = all_results[:min(10, len(all_results))]
        for i, result in enumerate(top_results, 1):
            content = result.get('content', '') or result.get('title', '')
            if content:
                # Use query-focused summarization
                summary = summarizer.summarize_for_query(content, boosted_query, max_length=300)
                if summary:
                    summaries.append(f"Document {i}: {summary}")
        
        search_context = "\n\n".join(summaries)
    
    # Create source aggregation
    source_aggregation = {
        'total_sources': len(all_results),
        'text_sources': len(text_results),
        'video_sources': len(video_results),
        'sources': all_results
    }
    
    logger.info(f"Comprehensive search completed: {len(all_results)} total sources")
    
    # Cache the results
    _cache_results(cache_key, search_context, url_mapping, source_aggregation)
    
    return search_context, url_mapping, source_aggregation
