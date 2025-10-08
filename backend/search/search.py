import logging
from typing import List, Dict, Tuple
from .coordinator import SearchCoordinator

logger = logging.getLogger(__name__)

# Global search coordinator instance
_search_coordinator = None

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