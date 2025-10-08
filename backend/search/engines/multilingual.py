import requests
from bs4 import BeautifulSoup
import logging
from typing import List, Dict, Optional
import time
import re
from urllib.parse import urlparse, quote

logger = logging.getLogger(__name__)

class MultilingualMedicalEngine:
    """Multilingual medical search engine supporting English, Vietnamese, and Chinese sources"""
    
    def __init__(self, timeout: int = 15):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5,vi;q=0.3,zh-CN;q=0.3',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })
        self.timeout = timeout
        
        # Comprehensive medical sources by language
        self.medical_sources = {
            'en': {
                'mayo_clinic': {
                    'base_url': 'https://www.mayoclinic.org',
                    'search_url': 'https://www.mayoclinic.org/search/search-results',
                    'domains': ['mayoclinic.org'],
                    'selectors': ['a[href*="/diseases-conditions/"]', 'a[href*="/symptoms/"]', '.search-result a']
                },
                'webmd': {
                    'base_url': 'https://www.webmd.com',
                    'search_url': 'https://www.webmd.com/search/search_results/default.aspx',
                    'domains': ['webmd.com'],
                    'selectors': ['a[href*="/default.htm"]', '.search-result a', 'a[href*="/content/"]']
                },
                'healthline': {
                    'base_url': 'https://www.healthline.com',
                    'search_url': 'https://www.healthline.com/search',
                    'domains': ['healthline.com'],
                    'selectors': ['a[href*="/health/"]', 'a[href*="/conditions/"]', '.search-result a']
                },
                'medlineplus': {
                    'base_url': 'https://medlineplus.gov',
                    'search_url': 'https://medlineplus.gov/search',
                    'domains': ['medlineplus.gov'],
                    'selectors': ['a[href*="/healthtopics/"]', '.search-result a']
                },
                'nih': {
                    'base_url': 'https://www.nih.gov',
                    'search_url': 'https://search.nih.gov/search',
                    'domains': ['nih.gov', 'nlm.nih.gov'],
                    'selectors': ['a[href*="/health/"]', '.search-result a']
                },
                'cdc': {
                    'base_url': 'https://www.cdc.gov',
                    'search_url': 'https://www.cdc.gov/search/index.html',
                    'domains': ['cdc.gov'],
                    'selectors': ['a[href*="/health/"]', '.search-result a']
                }
            },
            'vi': {
                'hello_bacsi': {
                    'base_url': 'https://hellobacsi.com',
                    'search_url': 'https://hellobacsi.com/tim-kiem',
                    'domains': ['hellobacsi.com'],
                    'selectors': ['a[href*="/suc-khoe/"]', 'a[href*="/benh/"]', '.search-result a', '.article-title a']
                },
                'alo_bacsi': {
                    'base_url': 'https://alobacsi.com',
                    'search_url': 'https://alobacsi.com/tim-kiem',
                    'domains': ['alobacsi.com'],
                    'selectors': ['a[href*="/suc-khoe/"]', 'a[href*="/benh/"]', '.search-result a']
                },
                'vinmec': {
                    'base_url': 'https://www.vinmec.com',
                    'search_url': 'https://www.vinmec.com/vi/tim-kiem',
                    'domains': ['vinmec.com'],
                    'selectors': ['a[href*="/suc-khoe/"]', 'a[href*="/benh/"]', '.search-result a']
                },
                'tam_anh': {
                    'base_url': 'https://tamanhhospital.vn',
                    'search_url': 'https://tamanhhospital.vn/tim-kiem',
                    'domains': ['tamanhhospital.vn'],
                    'selectors': ['a[href*="/suc-khoe/"]', 'a[href*="/benh/"]', '.search-result a']
                },
                'medlatec': {
                    'base_url': 'https://medlatec.vn',
                    'search_url': 'https://medlatec.vn/tim-kiem',
                    'domains': ['medlatec.vn'],
                    'selectors': ['a[href*="/suc-khoe/"]', 'a[href*="/benh/"]', '.search-result a']
                },
                'suckhoe_doisong': {
                    'base_url': 'https://suckhoedoisong.vn',
                    'search_url': 'https://suckhoedoisong.vn/tim-kiem',
                    'domains': ['suckhoedoisong.vn'],
                    'selectors': ['a[href*="/suc-khoe/"]', 'a[href*="/benh/"]', '.search-result a']
                },
                'vien_dinh_duong': {
                    'base_url': 'https://viendinhduong.vn',
                    'search_url': 'https://viendinhduong.vn/tim-kiem',
                    'domains': ['viendinhduong.vn'],
                    'selectors': ['a[href*="/dinh-duong/"]', 'a[href*="/suc-khoe/"]', '.search-result a']
                }
            },
            'zh': {
                'haodf': {
                    'base_url': 'https://www.haodf.com',
                    'search_url': 'https://www.haodf.com/search',
                    'domains': ['haodf.com'],
                    'selectors': ['a[href*="/jibing/"]', 'a[href*="/zixun/"]', '.search-result a']
                },
                'dxy': {
                    'base_url': 'https://www.dxy.cn',
                    'search_url': 'https://www.dxy.cn/search',
                    'domains': ['dxy.cn'],
                    'selectors': ['a[href*="/article/"]', 'a[href*="/jibing/"]', '.search-result a']
                },
                'chunyuyisheng': {
                    'base_url': 'https://www.chunyuyisheng.com',
                    'search_url': 'https://www.chunyuyisheng.com/search',
                    'domains': ['chunyuyisheng.com'],
                    'selectors': ['a[href*="/article/"]', 'a[href*="/jibing/"]', '.search-result a']
                },
                'xywy': {
                    'base_url': 'https://www.xywy.com',
                    'search_url': 'https://www.xywy.com/search',
                    'domains': ['xywy.com'],
                    'selectors': ['a[href*="/jibing/"]', 'a[href*="/article/"]', '.search-result a']
                },
                'jiankang': {
                    'base_url': 'https://www.jiankang.com',
                    'search_url': 'https://www.jiankang.com/search',
                    'domains': ['jiankang.com'],
                    'selectors': ['a[href*="/article/"]', 'a[href*="/jibing/"]', '.search-result a']
                },
                'familydoctor': {
                    'base_url': 'https://www.familydoctor.com.cn',
                    'search_url': 'https://www.familydoctor.com.cn/search',
                    'domains': ['familydoctor.com.cn'],
                    'selectors': ['a[href*="/article/"]', 'a[href*="/jibing/"]', '.search-result a']
                }
            }
        }
    
    def search(self, query: str, num_results: int = 10, languages: List[str] = None) -> List[Dict]:
        """Search across multiple languages and medical sources"""
        if languages is None:
            languages = ['en', 'vi', 'zh']
        
        all_results = []
        
        for lang in languages:
            if lang in self.medical_sources:
                lang_results = self._search_language_sources(query, lang, num_results // len(languages))
                all_results.extend(lang_results)
                time.sleep(0.5)  # Be respectful to servers
        
        # Remove duplicates and sort by relevance
        unique_results = self._remove_duplicates(all_results)
        return unique_results[:num_results]
    
    def _search_language_sources(self, query: str, language: str, num_results: int) -> List[Dict]:
        """Search sources for a specific language"""
        results = []
        sources = self.medical_sources.get(language, {})
        
        for source_name, source_config in sources.items():
            if len(results) >= num_results:
                break
            
            source_results = self._search_source(query, source_name, source_config, language)
            results.extend(source_results)
            time.sleep(0.3)  # Rate limiting
        
        return results
    
    def _search_source(self, query: str, source_name: str, source_config: Dict, language: str) -> List[Dict]:
        """Search a specific medical source"""
        try:
            search_url = source_config.get('search_url')
            if not search_url:
                return []
            
            # Prepare search parameters based on language
            params = self._prepare_search_params(query, language)
            
            response = self.session.get(search_url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            results = []
            
            # Try source-specific selectors
            selectors = source_config.get('selectors', ['a[href*="http"]'])
            
            for selector in selectors:
                links = soup.select(selector)
                if links:
                    logger.info(f"{source_name} ({language}) found {len(links)} results with selector: {selector}")
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
                            'language': language,
                            'domain': source_config['domains'][0]
                        })
                except Exception as e:
                    logger.debug(f"Error parsing {source_name} link: {e}")
                    continue
            
            return results
            
        except Exception as e:
            logger.warning(f"Medical source {source_name} ({language}) search failed: {e}")
            return []
    
    def _prepare_search_params(self, query: str, language: str) -> Dict[str, str]:
        """Prepare search parameters based on language"""
        # Common parameter names across different languages
        param_mappings = {
            'en': {'q': query, 'query': query, 'search': query},
            'vi': {'q': query, 'query': query, 'search': query, 'tu-khoa': query, 'tim-kiem': query},
            'zh': {'q': query, 'query': query, 'search': query, 'keyword': query, 'sousuo': query}
        }
        
        return param_mappings.get(language, {'q': query})
    
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
    
    def search_by_language(self, query: str, language: str, num_results: int = 10) -> List[Dict]:
        """Search sources for a specific language only"""
        if language not in self.medical_sources:
            logger.warning(f"Language {language} not supported")
            return []
        
        return self._search_language_sources(query, language, num_results)
    
    def get_fallback_sources(self, query: str, language: str, num_results: int) -> List[Dict]:
        """Get fallback sources when direct search fails"""
        fallback_sources = {
            'en': [
                {
                    'url': 'https://www.mayoclinic.org/diseases-conditions',
                    'title': f'Mayo Clinic: {query}',
                    'source': 'mayo_fallback',
                    'language': 'en',
                    'domain': 'mayoclinic.org'
                },
                {
                    'url': 'https://www.webmd.com/default.htm',
                    'title': f'WebMD: {query}',
                    'source': 'webmd_fallback',
                    'language': 'en',
                    'domain': 'webmd.com'
                }
            ],
            'vi': [
                {
                    'url': 'https://hellobacsi.com/suc-khoe',
                    'title': f'Hello Bacsi: {query}',
                    'source': 'hello_bacsi_fallback',
                    'language': 'vi',
                    'domain': 'hellobacsi.com'
                },
                {
                    'url': 'https://www.vinmec.com/vi/suc-khoe',
                    'title': f'Vinmec: {query}',
                    'source': 'vinmec_fallback',
                    'language': 'vi',
                    'domain': 'vinmec.com'
                }
            ],
            'zh': [
                {
                    'url': 'https://www.haodf.com/jibing',
                    'title': f'好大夫在线: {query}',
                    'source': 'haodf_fallback',
                    'language': 'zh',
                    'domain': 'haodf.com'
                },
                {
                    'url': 'https://www.dxy.cn/article',
                    'title': f'丁香园: {query}',
                    'source': 'dxy_fallback',
                    'language': 'zh',
                    'domain': 'dxy.cn'
                }
            ]
        }
        
        return fallback_sources.get(language, [])[:num_results]
