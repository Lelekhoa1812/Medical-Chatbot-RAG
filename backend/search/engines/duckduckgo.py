import requests
from bs4 import BeautifulSoup
import logging
from typing import List, Dict
import time

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
    
    def search(self, query: str, num_results: int = 10) -> List[Dict]:
        """Search with multiple DuckDuckGo strategies"""
        results = []
        
        # Strategy 1: HTML Interface
        html_results = self._search_html(query, num_results)
        if html_results:
            results.extend(html_results)
            logger.info(f"DuckDuckGo HTML found {len(html_results)} results")
        
        # Strategy 2: Instant Answer API
        if len(results) < num_results:
            api_results = self._search_api(query, num_results - len(results))
            if api_results:
                results.extend(api_results)
                logger.info(f"DuckDuckGo API found {len(api_results)} results")
        
        # Strategy 3: Lite Interface (mobile-friendly)
        if len(results) < num_results:
            lite_results = self._search_lite(query, num_results - len(results))
            if lite_results:
                results.extend(lite_results)
                logger.info(f"DuckDuckGo Lite found {len(lite_results)} results")
        
        return results[:num_results]
    
    def _search_html(self, query: str, num_results: int) -> List[Dict]:
        """Search using DuckDuckGo HTML interface"""
        try:
            url = "https://html.duckduckgo.com/html/"
            params = {
                'q': query,
                'kl': 'us-en',
                's': '0',  # Start from first result
                'dc': '1',  # Disable auto-complete
                'v': 'l',   # Lite version
                'o': 'json', # JSON output
                'api': 'd.js'  # API format
            }
            
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
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
