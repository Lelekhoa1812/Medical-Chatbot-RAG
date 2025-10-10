import logging
from typing import List, Dict, Tuple
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

class WebSearcher:
    """Legacy wrapper for backward compatibility"""
    def __init__(self):
        self.coordinator = get_search_coordinator()
        self.max_results = 10
        self.timeout = 10
        
    def search_google(self, query: str, num_results: int = 10) -> List[Dict]:
        """Search using the new coordinator system"""
        try:
            return self.coordinator.quick_search(query, num_results)
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def search_duckduckgo(self, query: str, num_results: int = 10) -> List[Dict]:
        """Search using DuckDuckGo engine"""
        try:
            return self.coordinator.quick_search(query, num_results)
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
            # Get search results
            results = self.coordinator.quick_search(query, num_results)
            
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
        coordinator = get_search_coordinator()
        return coordinator.quick_search(query, num_results)
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return []

# Enhanced search function with content extraction
def search_web_with_content(query: str, num_results: int = 10) -> Tuple[str, Dict[int, str]]:
    """Enhanced search with content extraction and summarization"""
    try:
        coordinator = get_search_coordinator()
        return coordinator.search(query, num_results)
    except Exception as e:
        logger.error(f"Enhanced web search failed: {e}")
        return "", {}

# Medical-focused search function
def search_medical(query: str, num_results: int = 8) -> Tuple[str, Dict[int, str]]:
    """Medical-focused search with enhanced processing"""
    try:
        coordinator = get_search_coordinator()
        return coordinator.medical_focus_search(query, num_results)
    except Exception as e:
        logger.error(f"Medical search failed: {e}")
        return "", {}

# Multilingual medical search function
def search_multilingual_medical(query: str, num_results: int = 10, target_language: str = None) -> Tuple[str, Dict[int, str]]:
    """Comprehensive multilingual medical search supporting English, Vietnamese, and Chinese"""
    try:
        coordinator = get_search_coordinator()
        return coordinator.multilingual_medical_search(query, num_results, target_language)
    except Exception as e:
        logger.error(f"Multilingual medical search failed: {e}")
        return "", {}

# Video search function
def search_videos(query: str, num_results: int = 2, target_language: str = None) -> List[Dict]:
    """Search for medical videos across multiple platforms"""
    try:
        coordinator = get_search_coordinator()
        return coordinator.video_search(query, num_results, target_language)
    except Exception as e:
        logger.error(f"Video search failed: {e}")
        return []

# Comprehensive search function with maximum information extraction
def search_comprehensive(query: str, num_results: int = 15, target_language: str = None, include_videos: bool = True) -> Tuple[str, Dict[int, str], Dict]:
    """Comprehensive search with maximum information extraction and detailed references"""
    logger.info(f"Starting comprehensive search for: {query} (target: {target_language})")
    
    # Get engines
    duckduckgo_engine = get_duckduckgo_engine()
    video_engine = get_video_engine()
    reranker = get_reranker()
    
    # Search for text results with higher initial count for better filtering
    text_results = duckduckgo_engine.search(query, num_results * 2)  # Get more to filter
    logger.info(f"Found {len(text_results)} text results")
    
    # Search for videos if requested
    video_results = []
    if include_videos:
        try:
            video_results = video_engine.search(query, num_results=5, language=target_language or 'en')
            logger.info(f"Found {len(video_results)} video results")
        except Exception as e:
            logger.warning(f"Video search failed: {e}")
    
    # Combine all results
    all_results = text_results + video_results
    
    # Use reranker to improve overall quality and relevance
    if all_results:
        reranked_results = reranker.rerank_results(query, all_results, min_score=0.4)
        logger.info(f"Reranked {len(all_results)} total results to {len(reranked_results)} high-quality results")
        all_results = reranked_results
    
    # Create URL mapping
    url_mapping = {}
    for i, result in enumerate(all_results, 1):
        url_mapping[i] = result.get('url', '')
    
    # Create search context using summarizer
    search_context = ""
    if all_results:
        summaries = []
        for i, result in enumerate(all_results, 1):
            content = result.get('content', '') or result.get('title', '')
            if content:
                # Use query-focused summarization
                summary = summarizer.summarize_for_query(content, query, max_length=300)
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
    return search_context, url_mapping, source_aggregation