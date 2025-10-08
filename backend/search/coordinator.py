import logging
from typing import List, Dict, Tuple
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from .engines.duckduckgo import DuckDuckGoEngine
from .engines.medical import MedicalSearchEngine
from .engines.multilingual import MultilingualMedicalEngine
from .extractors.content import ContentExtractor
from .processors.medical import MedicalSearchProcessor
from .processors.language import LanguageProcessor

logger = logging.getLogger(__name__)

class SearchCoordinator:
    """Coordinate multiple search strategies for comprehensive medical information"""
    
    def __init__(self, max_workers: int = 3):
        self.max_workers = max_workers
        
        # Initialize search engines
        self.duckduckgo_engine = DuckDuckGoEngine()
        self.medical_engine = MedicalSearchEngine()
        self.multilingual_engine = MultilingualMedicalEngine()
        
        # Initialize processors
        self.content_extractor = ContentExtractor()
        self.medical_processor = MedicalSearchProcessor()
        self.language_processor = LanguageProcessor()
        
        # Search strategies
        self.strategies = [
            self._search_multilingual,
            self._search_duckduckgo,
            self._search_medical_sources
        ]
    
    def search(self, query: str, num_results: int = 10, target_language: str = None) -> Tuple[str, Dict[int, str]]:
        """Execute comprehensive multilingual search with multiple strategies"""
        logger.info(f"Starting comprehensive multilingual search for: {query}")
        
        # Detect and enhance query for multiple languages
        enhanced_queries = self.language_processor.enhance_query(query, target_language)
        logger.info(f"Enhanced queries: {list(enhanced_queries.keys())}")
        
        # Execute search strategies in parallel
        all_results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit search tasks for each language
            future_to_strategy = {}
            
            for lang, enhanced_query in enhanced_queries.items():
                for strategy in self.strategies:
                    future = executor.submit(strategy, enhanced_query, num_results // len(enhanced_queries), lang)
                    future_to_strategy[future] = f"{strategy.__name__}_{lang}"
            
            # Collect results
            for future in as_completed(future_to_strategy):
                strategy_name = future_to_strategy[future]
                try:
                    results = future.result()
                    if results:
                        all_results.extend(results)
                        logger.info(f"{strategy_name} found {len(results)} results")
                except Exception as e:
                    logger.error(f"{strategy_name} failed: {e}")
        
        # Remove duplicates and filter by language preference
        unique_results = self._remove_duplicates(all_results)
        if target_language:
            unique_results = self.language_processor.filter_by_language(unique_results, target_language)
        
        logger.info(f"Total unique results: {len(unique_results)}")
        
        # Extract content from URLs
        enriched_results = self._enrich_with_content(unique_results)
        
        # Process results into comprehensive summary
        summary, url_mapping = self.medical_processor.process_results(enriched_results, query)
        
        logger.info(f"Multilingual search completed: {len(url_mapping)} sources processed")
        return summary, url_mapping
    
    def _search_multilingual(self, query: str, num_results: int, language: str = None) -> List[Dict]:
        """Search using multilingual medical engine"""
        try:
            if language:
                results = self.multilingual_engine.search_by_language(query, language, num_results)
            else:
                results = self.multilingual_engine.search(query, num_results)
            return results
        except Exception as e:
            logger.error(f"Multilingual search failed: {e}")
            return []
    
    def _search_duckduckgo(self, query: str, num_results: int, language: str = None) -> List[Dict]:
        """Search using DuckDuckGo engine"""
        try:
            results = self.duckduckgo_engine.search(query, num_results)
            return results
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
            return []
    
    def _search_medical_sources(self, query: str, num_results: int, language: str = None) -> List[Dict]:
        """Search using medical sources engine"""
        try:
            results = self.medical_engine.search(query, num_results)
            return results
        except Exception as e:
            logger.error(f"Medical sources search failed: {e}")
            return []
    
    def _remove_duplicates(self, results: List[Dict]) -> List[Dict]:
        """Remove duplicate results based on URL"""
        seen_urls = set()
        unique_results = []
        
        for result in results:
            url = result.get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(result)
        
        return unique_results
    
    def _enrich_with_content(self, results: List[Dict]) -> List[Dict]:
        """Enrich results with extracted content"""
        enriched_results = []
        
        # Extract content in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit content extraction tasks
            future_to_result = {
                executor.submit(self.content_extractor.extract, result['url']): result
                for result in results
            }
            
            # Collect enriched results
            for future in as_completed(future_to_result):
                original_result = future_to_result[future]
                try:
                    content = future.result()
                    if content:
                        enriched_result = original_result.copy()
                        enriched_result['content'] = content
                        enriched_results.append(enriched_result)
                except Exception as e:
                    logger.warning(f"Content extraction failed for {original_result['url']}: {e}")
                    # Still include result without content
                    enriched_results.append(original_result)
        
        return enriched_results
    
    def quick_search(self, query: str, num_results: int = 5) -> List[Dict]:
        """Quick search for basic results without content extraction"""
        logger.info(f"Quick search for: {query}")
        
        # Use only DuckDuckGo for speed
        results = self.duckduckgo_engine.search(query, num_results)
        
        # Remove duplicates
        unique_results = self._remove_duplicates(results)
        
        logger.info(f"Quick search completed: {len(unique_results)} results")
        return unique_results
    
    def medical_focus_search(self, query: str, num_results: int = 8) -> Tuple[str, Dict[int, str]]:
        """Medical-focused search with enhanced processing"""
        logger.info(f"Medical focus search for: {query}")
        
        # Use medical engine primarily
        medical_results = self.medical_engine.search(query, num_results)
        
        # Add some general results for context
        general_results = self.duckduckgo_engine.search(query, 3)
        
        # Combine and deduplicate
        all_results = self._remove_duplicates(medical_results + general_results)
        
        # Enrich with content
        enriched_results = self._enrich_with_content(all_results)
        
        # Process with medical focus
        summary, url_mapping = self.medical_processor.process_results(enriched_results, query)
        
        logger.info(f"Medical focus search completed: {len(url_mapping)} sources")
        return summary, url_mapping
    
    def multilingual_medical_search(self, query: str, num_results: int = 10, target_language: str = None) -> Tuple[str, Dict[int, str]]:
        """Comprehensive multilingual medical search"""
        logger.info(f"Multilingual medical search for: {query} (target: {target_language})")
        
        # Detect source language
        source_language = self.language_processor.detect_language(query)
        logger.info(f"Detected source language: {source_language}")
        
        # Use multilingual search with language preference
        summary, url_mapping = self.search(query, num_results, target_language)
        
        logger.info(f"Multilingual medical search completed: {len(url_mapping)} sources")
        return summary, url_mapping
