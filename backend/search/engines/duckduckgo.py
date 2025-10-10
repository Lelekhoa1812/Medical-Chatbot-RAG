import requests
from bs4 import BeautifulSoup
import logging
from typing import List, Dict
import time
from models.reranker import MedicalReranker

logger = logging.getLogger(__name__)

class DuckDuckGoEngine:
    """DuckDuckGo search engine with multiple strategies"""
    
    def __init__(self, timeout: int = 15):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        self.timeout = timeout
        self.reranker = MedicalReranker()
    
    def search(self, query: str, num_results: int = 10) -> List[Dict]:
        """Search with multiple DuckDuckGo strategies and medical focus"""
        # Clean and simplify the query first
        clean_query = self._clean_query(query)
        logger.info(f"Cleaned query: '{query}' -> '{clean_query}'")
        
        results = []
        min_score = 0.3  # Adjustable
        
        # Strategy 1: HTML Interface with medical focus
        html_results = self._search_html(clean_query, num_results * 3)  # Get more to filter
        if html_results:
            results.extend(html_results)
            logger.info(f"DuckDuckGo HTML found {len(html_results)} results")
        
        # Strategy 2: Instant Answer API
        if len(results) < num_results * 2:
            api_results = self._search_api(clean_query, num_results)
            if api_results:
                results.extend(api_results)
                logger.info(f"DuckDuckGo API found {len(api_results)} results")
        
        # Strategy 3: Lite Interface (mobile-friendly)
        if len(results) < num_results * 2:
            lite_results = self._search_lite(clean_query, num_results)
            if lite_results:
                results.extend(lite_results)
                logger.info(f"DuckDuckGo Lite found {len(lite_results)} results")
        
        # If still no results, try with even simpler query
        if not results:
            simple_query = self._simplify_query(clean_query)
            if simple_query != clean_query:
                logger.info(f"Trying simplified query: '{simple_query}'")
                html_results = self._search_html(simple_query, num_results * 2)
                if html_results:
                    results.extend(html_results)
                    logger.info(f"Simplified query found {len(html_results)} results")
        
        # If still no results, try fallback search engines
        if not results:
            logger.warning("DuckDuckGo failed, trying fallback search engines")
            fallback_results = self._fallback_search(clean_query, num_results)
            if fallback_results:
                results.extend(fallback_results)
                logger.info(f"Fallback search found {len(fallback_results)} results")
        
        # Filter out irrelevant results first (less aggressive)
        filtered_results = self._filter_irrelevant_sources(results)
        logger.info(f"Filtered {len(results)} results to {len(filtered_results)} relevant results")
        
        # If we have results, use reranker; otherwise return what we have
        if filtered_results:
            try:
                reranked_results = self.reranker.rerank_results(clean_query, filtered_results, min_score)
                logger.info(f"Reranked {len(filtered_results)} results to {len(reranked_results)} high-quality results")
                return reranked_results[:num_results]
            except Exception as e:
                logger.warning(f"Reranking failed: {e}, returning filtered results")
                return filtered_results[:num_results]
        
        return filtered_results[:num_results]
    
    def _clean_query(self, query: str) -> str:
        """Clean and normalize search query"""
        if not query:
            return ""
        
        # Remove bullet points and special characters
        import re
        cleaned = re.sub(r'[•·▪▫‣⁃]', ' ', query)  # Remove bullet points
        cleaned = re.sub(r'[^\w\s\-\.]', ' ', cleaned)  # Keep only alphanumeric, spaces, hyphens, dots
        cleaned = re.sub(r'\s+', ' ', cleaned)  # Normalize whitespace
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
    
    def _filter_irrelevant_sources(self, results: List[Dict]) -> List[Dict]:
        """Filter out irrelevant sources like generic health pages, quizzes, etc."""
        import re
        filtered = []
        
        # Only exclude obvious non-medical content
        exclude_patterns = [
            r'/quiz$',  # Quiz pages (end of URL)
            r'/test$',  # Test pages (end of URL)
            r'/assessment',  # Assessment pages
            r'/survey',  # Survey pages
            r'homepage|main page|index',  # Homepage/index pages
            r'login|sign.up|register',  # Auth pages
            r'contact|about.us|privacy',  # Info pages
            r'subscribe|newsletter|rss',  # Subscription pages
            r'sitemap',  # Navigation pages
        ]
        
        for result in results:
            url = result.get('url', '').lower()
            title = result.get('title', '').lower()
            
            # Skip if matches exclude patterns
            should_exclude = False
            for pattern in exclude_patterns:
                if re.search(pattern, url) or re.search(pattern, title):
                    should_exclude = True
                    logger.debug(f"Excluding irrelevant source: {url}")
                    break
            
            if not should_exclude:
                filtered.append(result)
        
        # If we filtered out too many, be less aggressive
        if len(filtered) < len(results) * 0.3:  # If we kept less than 30%
            logger.warning(f"Filtering too aggressive, keeping more results: {len(results)} -> {len(filtered)}")
            # Return original results with minimal filtering
            minimal_filtered = []
            for result in results:
                url = result.get('url', '').lower()
                if not any(re.search(pattern, url) for pattern in [r'login', r'sign.up', r'register']):
                    minimal_filtered.append(result)
            return minimal_filtered
        
        return filtered
    
    def _search_html(self, query: str, num_results: int) -> List[Dict]:
        """Search using DuckDuckGo HTML interface with better error handling"""
        try:
            # Try multiple DuckDuckGo endpoints
            endpoints = [
                {
                    'url': 'https://html.duckduckgo.com/html/',
                    'params': {
                'q': query,
                'kl': 'us-en',
                        's': '0',
                        'dc': '1',
                        'v': 'l'
                    }
                },
                {
                    'url': 'https://lite.duckduckgo.com/lite/',
                    'params': {
                        'q': query,
                        'kl': 'us-en'
                    }
                },
                {
                    'url': 'https://duckduckgo.com/html/',
                    'params': {
                        'q': query,
                        'kl': 'us-en'
                    }
                }
            ]
            
            for endpoint in endpoints:
                try:
                    # Add random delay to avoid rate limiting
                    import time
                    time.sleep(0.5)
                    
                    # Update headers to look more like a real browser
                    headers = self.session.headers.copy()
                    headers.update({
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                        'Accept-Language': 'en-US,en;q=0.5',
                        'Accept-Encoding': 'gzip, deflate',
                        'DNT': '1',
                        'Connection': 'keep-alive',
                        'Upgrade-Insecure-Requests': '1',
                    })
                    
                    response = self.session.get(
                        endpoint['url'], 
                        params=endpoint['params'], 
                        headers=headers,
                        timeout=self.timeout
                    )
                    
                    if response.status_code == 403:
                        logger.warning(f"DuckDuckGo endpoint {endpoint['url']} returned 403, trying next...")
                        continue
                    elif response.status_code == 429:
                        logger.warning(f"DuckDuckGo rate limited, waiting...")
                        time.sleep(2)
                        continue
                    
            response.raise_for_status()
                    break
                    
                except Exception as e:
                    logger.warning(f"DuckDuckGo endpoint {endpoint['url']} failed: {e}")
                    if endpoint == endpoints[-1]:  # Last endpoint
                        raise e
                    continue
            else:
                # All endpoints failed
                logger.error("All DuckDuckGo endpoints failed")
                return []
            
            soup = BeautifulSoup(response.content, 'html.parser')
            results = []
            
            # Multiple selectors for different DDG layouts
            selectors = [
                'a.result__a',
                'a[data-testid="result-title-a"]',
                '.result__title a',
                '.web-result a',
                '.result a',
                'a[href*="http"]:not([href*="duckduckgo.com"])'
            ]
            
            for selector in selectors:
                links = soup.select(selector)
                if links:
                    logger.info(f"Using selector: {selector} - found {len(links)} links")
                    break
            
            for link in links[:num_results]:
                try:
                    href = link.get('href')
                    if not href or href.startswith('#') or 'duckduckgo.com' in href:
                        continue
                    
                    # Clean up DDG redirect URLs
                    if href.startswith('/l/?uddg='):
                        import urllib.parse
                        href = urllib.parse.unquote(href.split('uddg=')[1])
                    
                    title = link.get_text(strip=True)
                    if title and href.startswith('http'):
                        results.append({
                            'url': href,
                            'title': title,
                            'source': 'duckduckgo_html'
                        })
                except Exception as e:
                    logger.debug(f"Error parsing link: {e}")
                    continue
            
            return results
            
        except Exception as e:
            logger.warning(f"DuckDuckGo HTML search failed: {e}")
            return []
    
    def _search_api(self, query: str, num_results: int) -> List[Dict]:
        """Search using DuckDuckGo Instant Answer API"""
        try:
            url = "https://api.duckduckgo.com/"
            params = {
                'q': query,
                'format': 'json',
                'no_html': '1',
                'skip_disambig': '1',
                't': 'MedicalChatbot'
            }
            
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            
            results = []
            
            # Abstract result
            if data.get('AbstractURL') and data.get('Abstract'):
                results.append({
                    'url': data['AbstractURL'],
                    'title': data.get('Heading', query),
                    'content': data.get('Abstract', ''),
                    'source': 'duckduckgo_api'
                })
            
            # Related topics
            for topic in data.get('RelatedTopics', []):
                if len(results) >= num_results:
                    break
                    
                if isinstance(topic, dict) and topic.get('FirstURL'):
                    text = topic.get('Text', '')
                    title = text.split(' - ')[0] if ' - ' in text else text[:50]
                    
                    results.append({
                        'url': topic['FirstURL'],
                        'title': title,
                        'content': text,
                        'source': 'duckduckgo_api'
                    })
            
            return results
            
        except Exception as e:
            logger.warning(f"DuckDuckGo API search failed: {e}")
            return []
    
    def _search_lite(self, query: str, num_results: int) -> List[Dict]:
        """Search using DuckDuckGo Lite interface"""
        try:
            url = "https://lite.duckduckgo.com/lite/"
            params = {
                'q': query,
                'kl': 'us-en'
            }
            
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            results = []
            
            # Lite interface selectors
            links = soup.select('a[href*="http"]:not([href*="duckduckgo.com"])')
            
            for link in links[:num_results]:
                try:
                    href = link.get('href')
                    title = link.get_text(strip=True)
                    
                    if href and title and href.startswith('http'):
                        results.append({
                            'url': href,
                            'title': title,
                            'source': 'duckduckgo_lite'
                        })
                except Exception as e:
                    logger.debug(f"Error parsing lite link: {e}")
                    continue
            
            return results
            
        except Exception as e:
            logger.warning(f"DuckDuckGo Lite search failed: {e}")
            return []
    
    def _fallback_search(self, query: str, num_results: int) -> List[Dict]:
        """Fallback search using alternative methods when DuckDuckGo fails"""
        results = []
        
        # Try Bing search as fallback
        try:
            bing_results = self._search_bing(query, num_results)
            if bing_results:
                results.extend(bing_results)
                logger.info(f"Bing fallback found {len(bing_results)} results")
        except Exception as e:
            logger.warning(f"Bing fallback failed: {e}")
        
        # Try Startpage search as fallback
        try:
            startpage_results = self._search_startpage(query, num_results)
            if startpage_results:
                results.extend(startpage_results)
                logger.info(f"Startpage fallback found {len(startpage_results)} results")
        except Exception as e:
            logger.warning(f"Startpage fallback failed: {e}")
        
        # Try Searx instances as fallback
        try:
            searx_results = self._search_searx(query, num_results)
            if searx_results:
                results.extend(searx_results)
                logger.info(f"Searx fallback found {len(searx_results)} results")
        except Exception as e:
            logger.warning(f"Searx fallback failed: {e}")
        
        return results
    
    def _search_bing(self, query: str, num_results: int) -> List[Dict]:
        """Search using Bing as fallback"""
        try:
            url = "https://www.bing.com/search"
            params = {
                'q': query,
                'count': min(num_results, 50),
                'first': 1
            }
            
            headers = self.session.headers.copy()
            headers.update({
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            })
            
            response = self.session.get(url, params=params, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            results = []
            
            # Bing result selectors
            selectors = [
                'h2 a',
                '.b_title a',
                '.b_algo a'
            ]
            
            for selector in selectors:
                links = soup.select(selector)
                if links:
                    logger.info(f"Bing found {len(links)} links with selector: {selector}")
                    break
            
            for link in links[:num_results]:
                try:
                    href = link.get('href')
                    if not href or href.startswith('#') or 'bing.com' in href:
                        continue
                    
                    title = link.get_text(strip=True)
                    if title and href.startswith('http'):
                        results.append({
                            'url': href,
                            'title': title,
                            'source': 'bing_fallback'
                        })
                except Exception as e:
                    logger.debug(f"Error parsing Bing link: {e}")
                    continue
            
            return results
            
        except Exception as e:
            logger.warning(f"Bing search failed: {e}")
            return []
    
    def _search_startpage(self, query: str, num_results: int) -> List[Dict]:
        """Search using Startpage as fallback"""
        try:
            url = "https://www.startpage.com/sp/search"
            params = {
                'query': query,
                'cat': 'web',
                'pl': 'opensearch'
            }
            
            headers = self.session.headers.copy()
            headers.update({
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5'
            })
            
            response = self.session.get(url, params=params, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            results = []
            
            # Startpage result selectors
            links = soup.select('a[href*="http"]:not([href*="startpage.com"])')
            
            for link in links[:num_results]:
                try:
                    href = link.get('href')
                    if not href or href.startswith('#') or 'startpage.com' in href:
                        continue
                    
                    title = link.get_text(strip=True)
                    if title and href.startswith('http'):
                        results.append({
                            'url': href,
                            'title': title,
                            'source': 'startpage_fallback'
                        })
                except Exception as e:
                    logger.debug(f"Error parsing Startpage link: {e}")
                    continue
            
            return results
            
        except Exception as e:
            logger.warning(f"Startpage search failed: {e}")
            return []
    
    def _search_searx(self, query: str, num_results: int) -> List[Dict]:
        """Search using public Searx instances as fallback"""
        searx_instances = [
            "https://searx.be",
            "https://searx.tiekoetter.com",
            "https://searx.xyz"
        ]
        
        for instance in searx_instances:
            try:
                url = f"{instance}/search"
                params = {
                    'q': query,
                    'format': 'json'
                }
                
                response = self.session.get(url, params=params, timeout=self.timeout)
                response.raise_for_status()
                
                data = response.json()
                results = []
                
                for result in data.get('results', [])[:num_results]:
                    try:
                        url = result.get('url', '')
                        title = result.get('title', '')
                        content = result.get('content', '')
                        
                        if url and title and url.startswith('http'):
                            results.append({
                                'url': url,
                                'title': title,
                                'content': content,
                                'source': 'searx_fallback'
                            })
                    except Exception as e:
                        logger.debug(f"Error parsing Searx result: {e}")
                        continue
                
                if results:
                    logger.info(f"Searx instance {instance} found {len(results)} results")
                    return results
                    
            except Exception as e:
                logger.debug(f"Searx instance {instance} failed: {e}")
                continue
        
            return []