import logging
from typing import List, Dict, Tuple
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from .engines.duckduckgo import DuckDuckGoEngine
from .engines.medical import MedicalSearchEngine
from .engines.multilingual import MultilingualMedicalEngine
from .engines.video import VideoSearchEngine
from .extractors.content import ContentExtractor
from .processors.medical import MedicalSearchProcessor
from .processors.language import LanguageProcessor
from .processors.sources import SourceAggregator
from .processors.enhanced import EnhancedContentProcessor
from models.reranker import MedicalReranker

logger = logging.getLogger(__name__)

class SearchCoordinator:
    """Coordinate multiple search strategies for comprehensive medical information"""
    
    def __init__(self, max_workers: int = 3):
        self.max_workers = max_workers
        
        # Initialize search engines
        self.duckduckgo_engine = DuckDuckGoEngine()
        self.medical_engine = MedicalSearchEngine()
        self.multilingual_engine = MultilingualMedicalEngine()
        self.video_engine = VideoSearchEngine()
        
        # Initialize processors
        self.content_extractor = ContentExtractor()
        self.medical_processor = MedicalSearchProcessor()
        self.language_processor = LanguageProcessor()
        self.source_aggregator = SourceAggregator()
        self.enhanced_processor = EnhancedContentProcessor()
        self.reranker = MedicalReranker()
        
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
        
        # Use reranker to improve quality and relevance
        if enriched_results:
            reranked_results = self.reranker.rerank_results(query, enriched_results, min_score=0.4)
            logger.info(f"Reranked {len(enriched_results)} results to {len(reranked_results)} high-quality results")
            enriched_results = reranked_results
        
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
        
        # If no results, try with simplified query
        if not results:
            logger.warning("No results from DuckDuckGo, trying simplified query")
            simplified_query = self._simplify_query(query)
            if simplified_query != query:
                results = self.duckduckgo_engine.search(simplified_query, num_results)
                logger.info(f"Simplified query '{simplified_query}' found {len(results)} results")
        
        # If still no results, try medical engine as fallback
        if not results:
            logger.warning("Still no results, trying medical engine fallback")
            try:
                medical_results = self.medical_engine.search(query, num_results)
                if medical_results:
                    results = medical_results
                    logger.info(f"Medical engine fallback found {len(results)} results")
            except Exception as e:
                logger.warning(f"Medical engine fallback failed: {e}")
        
        # Remove duplicates
        unique_results = self._remove_duplicates(results)
        
        # If we still have no results, create a basic fallback
        if not unique_results:
            logger.warning("No search results found, creating basic fallback")
            unique_results = self._create_fallback_results(query)
        
        logger.info(f"Quick search completed: {len(unique_results)} results")
        return unique_results
    
    def _simplify_query(self, query: str) -> str:
        """Simplify query to core medical terms"""
        if not query:
            return ""
        
        # Extract key medical terms
        import re
        words = query.split()
        
        # Keep medical keywords and important terms
        medical_keywords = [
            'migraine', 'headache', 'pain', 'treatment', 'therapy', 'medication', 'drug',
            'chronic', 'acute', 'symptoms', 'diagnosis', 'prevention', 'management',
            'disease', 'condition', 'syndrome', 'disorder', 'infection', 'inflammation',
            'blood', 'heart', 'lung', 'brain', 'liver', 'kidney', 'diabetes', 'cancer',
            'covid', 'flu', 'cold', 'fever', 'cough', 'breathing', 'chest', 'stomach'
        ]
        
        # Keep words that are medical keywords or are important (longer than 3 chars)
        important_words = []
        for word in words:
            word_lower = word.lower()
            if word_lower in medical_keywords or len(word) > 3:
                important_words.append(word)
        
        # If we have important words, use them; otherwise use first few words
        if important_words:
            return ' '.join(important_words[:5])  # Max 5 words
        else:
            return ' '.join(words[:3])  # Max 3 words
    
    def _create_fallback_results(self, query: str) -> List[Dict]:
        """Create basic fallback results when search fails"""
        # Create some basic medical information URLs as fallback
        fallback_urls = [
            "https://www.mayoclinic.org",
            "https://www.webmd.com",
            "https://www.healthline.com",
            "https://medlineplus.gov",
            "https://www.cdc.gov"
        ]
        
        results = []
        for i, url in enumerate(fallback_urls[:3]):  # Limit to 3 fallback results
            results.append({
                'url': url,
                'title': f"Medical Information - {query}",
                'source': 'fallback',
                'composite_score': 0.3 - (i * 0.05)  # Decreasing score
            })
        
        return results
    
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
        
        # Use reranker to improve quality and relevance
        if enriched_results:
            reranked_results = self.reranker.rerank_results(query, enriched_results, min_score=0.5)
            logger.info(f"Reranked {len(enriched_results)} medical results to {len(reranked_results)} high-quality results")
            enriched_results = reranked_results
        
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
    
    def comprehensive_search(self, query: str, num_results: int = 15, target_language: str = None, include_videos: bool = True) -> Tuple[str, Dict[int, str], Dict]:
        """Comprehensive search with maximum information extraction and detailed references"""
        logger.info(f"Starting comprehensive search for: {query} (target: {target_language})")
        
        # Detect source language
        source_language = self.language_processor.detect_language(query)
        logger.info(f"Detected source language: {source_language}")
        
        # Execute comprehensive search
        search_results = []
        video_results = []
        
        # 1. Multilingual text search
        text_summary, text_url_mapping = self.search(query, num_results, target_language)
        
        # 2. Video search if requested
        if include_videos:
            try:
                video_results = self.video_search(query, num_results=5, target_language=target_language)
                logger.info(f"Video search found {len(video_results)} videos")
            except Exception as e:
                logger.warning(f"Video search failed: {e}")
        
        # 3. Aggregate all sources
        all_sources = []
        
        # Add text sources
        for i, url in text_url_mapping.items():
            # Find corresponding source data
            source_data = self._find_source_data(url, text_url_mapping)
            if source_data:
                all_sources.append(source_data)
        
        # Add video sources
        for video in video_results:
            all_sources.append(video)
        
        # 4. Process with enhanced content processor
        if all_sources:
            comprehensive_summary, detailed_mapping = self.enhanced_processor.process_comprehensive_content(all_sources, query)
        else:
            comprehensive_summary = text_summary
            detailed_mapping = text_url_mapping
        
        # 5. Create comprehensive source aggregation
        source_aggregation = self.source_aggregator.aggregate_sources(all_sources, video_results)
        
        # 6. Generate comprehensive references
        comprehensive_references = self.source_aggregator.create_comprehensive_references(all_sources, max_references=20)
        
        # 7. Add inline citations
        final_summary = self.enhanced_processor.create_inline_citations(comprehensive_summary, detailed_mapping)
        
        # 8. Add source statistics
        source_stats = self.enhanced_processor.generate_source_statistics(all_sources)
        
        # 9. Combine everything
        final_response = f"{final_summary}\n\n{comprehensive_references}\n\n{source_stats}"
        
        logger.info(f"Comprehensive search completed: {len(all_sources)} total sources processed")
        
        return final_response, detailed_mapping, source_aggregation
    
    def _find_source_data(self, url: str, url_mapping: Dict[int, str]) -> Dict:
        """Find source data for a given URL"""
        # This is a simplified version - ensure required fields always exist
        return {
            'url': url,
            'title': f"Source: {url}",
            'content': '',
            'domain': self._extract_domain(url),
            'type': 'text',
            'source_type': 'text',
            'language': 'en',
            'source_name': '',
            'platform': ''
        }
    
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
            return ''
    
    def video_search(self, query: str, num_results: int = 3, target_language: str = None) -> List[Dict]:
        """Search for medical videos across multiple platforms"""
        logger.info(f"Video search for: {query} (target: {target_language})")
        
        # Detect language if not provided
        if not target_language:
            target_language = self.language_processor.detect_language(query)
        
        # Map language codes
        lang_mapping = {
            'EN': 'en',
            'VI': 'vi', 
            'ZH': 'zh',
            'en': 'en',
            'vi': 'vi',
            'zh': 'zh'
        }
        search_language = lang_mapping.get(target_language, 'en')
        
        # Search for videos
        raw_results = self.video_engine.search(query, num_results, search_language)
        
        # Use reranker to filter and improve YouTube results
        filtered_video_results = self.reranker.filter_youtube_results(raw_results, query)
        
        # Validate and normalize results to avoid corrupted cards/links
        video_results = self._sanitize_video_results(filtered_video_results, limit=num_results)
        
        logger.info(f"Video search completed: {len(video_results)} videos found")
        return video_results

    def _sanitize_video_results(self, results: List[Dict], limit: int = 4) -> List[Dict]:
        """Ensure each video has a valid absolute https URL, reasonable title, and platform metadata.
        Drop unreachable/broken items and deduplicate by URL.
        """
        from urllib.parse import urlparse
        import requests
        clean: List[Dict] = []
        seen = set()
        for item in results or []:
            url = (item or {}).get('url', '')
            title = (item or {}).get('title', '').strip()
            if not url or not title:
                continue
            try:
                parsed = urlparse(url)
                if parsed.scheme not in ('http', 'https'):
                    continue
                if not parsed.netloc:
                    continue
                # Quick reachability check; YouTube often blocks HEAD, so skip strict checks for youtube domain
                host = parsed.netloc.lower()
                norm_url = url
                if 'youtube.com' not in host:
                    try:
                        r = requests.head(url, allow_redirects=True, timeout=3)
                        if r.status_code >= 400:
                            continue
                        norm_url = getattr(r, 'url', url) or url
                    except Exception:
                        # If HEAD blocked, try a light GET with small timeout
                        try:
                            r = requests.get(url, stream=True, timeout=4)
                            if r.status_code >= 400:
                                continue
                            norm_url = getattr(r, 'url', url) or url
                        except Exception:
                            continue
                if norm_url in seen:
                    continue
                seen.add(norm_url)
                platform = parsed.netloc.lower()
                if platform.startswith('www.'):
                    platform = platform[4:]
                clean.append({
                    'title': title,
                    'url': norm_url,
                    'thumbnail': item.get('thumbnail', ''),
                    'source': item.get('source', platform.split('.')[0]),
                    'platform': platform,
                    'language': item.get('language', 'en')
                })
                if len(clean) >= limit:
                    break
            except Exception:
                continue
        return clean



