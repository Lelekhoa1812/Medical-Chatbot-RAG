import requests
from bs4 import BeautifulSoup
import logging
from typing import List, Dict
import time

logger = logging.getLogger(__name__)

class MedicalSearchEngine:
    """Specialized medical search engine with curated sources"""
    
    def __init__(self, timeout: int = 15):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        self.timeout = timeout
        
        # Curated medical sources
        self.medical_sources = {
            'mayo_clinic': {
                'base_url': 'https://www.mayoclinic.org',
                'search_url': 'https://www.mayoclinic.org/search/search-results',
                'domains': ['mayoclinic.org']
            },
            'webmd': {
                'base_url': 'https://www.webmd.com',
                'search_url': 'https://www.webmd.com/search/search_results/default.aspx',
                'domains': ['webmd.com']
            },
            'healthline': {
                'base_url': 'https://www.healthline.com',
                'search_url': 'https://www.healthline.com/search',
                'domains': ['healthline.com']
            },
            'medlineplus': {
                'base_url': 'https://medlineplus.gov',
                'search_url': 'https://medlineplus.gov/search',
                'domains': ['medlineplus.gov']
            },
            'nih': {
                'base_url': 'https://www.nih.gov',
                'search_url': 'https://search.nih.gov/search',
                'domains': ['nih.gov', 'nlm.nih.gov']
            }
        }
    
    def search(self, query: str, num_results: int = 10) -> List[Dict]:
        """Search medical sources for relevant information"""
        results = []
        
        # Strategy 1: Direct medical source searches
        for source_name, source_config in self.medical_sources.items():
            if len(results) >= num_results:
                break
                
            source_results = self._search_medical_source(query, source_name, source_config)
            results.extend(source_results)
            
            # Add delay between requests
            time.sleep(0.5)
        
        # Strategy 2: Medical fallback sources
        if len(results) < num_results:
            fallback_results = self._get_fallback_sources(query, num_results - len(results))
            results.extend(fallback_results)
        
        return results[:num_results]
    
    def _search_medical_source(self, query: str, source_name: str, source_config: Dict) -> List[Dict]:
        """Search a specific medical source"""
        try:
            search_url = source_config.get('search_url')
            if not search_url:
                return []
            
            params = {
                'q': query,
                'query': query,
                'search': query
            }
            
            response = self.session.get(search_url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            results = []
            
            # Source-specific selectors
            selectors = self._get_source_selectors(source_name)
            
            for selector in selectors:
                links = soup.select(selector)
                if links:
                    logger.info(f"{source_name} found {len(links)} results with selector: {selector}")
                    break
            
            for link in links[:3]:  # Limit per source
                try:
                    href = link.get('href')
                    if not href:
                        continue
                    
                    # Make absolute URL
                    if href.startswith('/'):
                        href = source_config['base_url'] + href
                    
                    title = link.get_text(strip=True)
                    if title and href.startswith('http'):
                        results.append({
                            'url': href,
                            'title': title,
                            'source': source_name,
                            'domain': source_config['domains'][0]
                        })
                except Exception as e:
                    logger.debug(f"Error parsing {source_name} link: {e}")
                    continue
            
            return results
            
        except Exception as e:
            logger.warning(f"Medical source {source_name} search failed: {e}")
            return []
    
    def _get_source_selectors(self, source_name: str) -> List[str]:
        """Get CSS selectors for specific medical sources"""
        selectors_map = {
            'mayo_clinic': [
                'a[href*="/diseases-conditions/"]',
                'a[href*="/symptoms/"]',
                '.search-result a',
                '.result-title a'
            ],
            'webmd': [
                'a[href*="/default.htm"]',
                '.search-result a',
                '.result-title a',
                'a[href*="/content/"]'
            ],
            'healthline': [
                'a[href*="/health/"]',
                '.search-result a',
                '.result-title a',
                'a[href*="/conditions/"]'
            ],
            'medlineplus': [
                'a[href*="/healthtopics/"]',
                '.search-result a',
                '.result-title a'
            ],
            'nih': [
                'a[href*="/health/"]',
                '.search-result a',
                '.result-title a'
            ]
        }
        return selectors_map.get(source_name, ['a[href*="http"]'])
    
    def _get_fallback_sources(self, query: str, num_results: int) -> List[Dict]:
        """Get fallback medical sources when direct search fails"""
        fallback_sources = [
            {
                'url': 'https://www.mayoclinic.org/diseases-conditions',
                'title': f'Mayo Clinic: {query}',
                'source': 'mayo_fallback',
                'domain': 'mayoclinic.org'
            },
            {
                'url': 'https://www.webmd.com/default.htm',
                'title': f'WebMD: {query}',
                'source': 'webmd_fallback',
                'domain': 'webmd.com'
            },
            {
                'url': 'https://www.healthline.com/health',
                'title': f'Healthline: {query}',
                'source': 'healthline_fallback',
                'domain': 'healthline.com'
            },
            {
                'url': 'https://medlineplus.gov/healthtopics.html',
                'title': f'MedlinePlus: {query}',
                'source': 'medlineplus_fallback',
                'domain': 'medlineplus.gov'
            },
            {
                'url': 'https://www.cdc.gov',
                'title': f'CDC: {query}',
                'source': 'cdc_fallback',
                'domain': 'cdc.gov'
            }
        ]
        
        return fallback_sources[:num_results]
