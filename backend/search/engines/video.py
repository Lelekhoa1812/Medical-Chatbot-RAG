import requests
from bs4 import BeautifulSoup
import logging
from typing import List, Dict
import time
import re
from urllib.parse import urlparse, quote

logger = logging.getLogger(__name__)

class VideoSearchEngine:
    """Search engine for medical videos across multiple platforms"""
    
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
        
        # Video platforms by language
        self.video_platforms = {
            'en': [
                {
                    'name': 'youtube',
                    'search_url': 'https://www.youtube.com/results',
                    'params': {'search_query': ''},
                    'selectors': ['a#video-title', 'a[href*="/watch?v="]'],
                    'base_url': 'https://www.youtube.com'
                },
                {
                    'name': 'medscape_videos',
                    'search_url': 'https://www.medscape.com/search',
                    'params': {'q': ''},
                    'selectors': ['a[href*="/video/"]', 'a[href*="/viewarticle/"]'],
                    'base_url': 'https://www.medscape.com'
                }
            ],
            'vi': [
                {
                    'name': 'youtube_vi',
                    'search_url': 'https://www.youtube.com/results',
                    'params': {'search_query': ''},
                    'selectors': ['a#video-title', 'a[href*="/watch?v="]'],
                    'base_url': 'https://www.youtube.com'
                },
                {
                    'name': 'vinmec_videos',
                    'search_url': 'https://www.vinmec.com/vi/tim-kiem',
                    'params': {'q': ''},
                    'selectors': ['a[href*="/video/"]', 'a[href*="/suc-khoe/"]'],
                    'base_url': 'https://www.vinmec.com'
                }
            ],
            'zh': [
                {
                    'name': 'youtube_zh',
                    'search_url': 'https://www.youtube.com/results',
                    'params': {'search_query': ''},
                    'selectors': ['a#video-title', 'a[href*="/watch?v="]'],
                    'base_url': 'https://www.youtube.com'
                },
                {
                    'name': 'haodf_videos',
                    'search_url': 'https://www.haodf.com/search',
                    'params': {'q': ''},
                    'selectors': ['a[href*="/video/"]', 'a[href*="/jibing/"]'],
                    'base_url': 'https://www.haodf.com'
                }
            ]
        }
    
    def search(self, query: str, num_results: int = 3, language: str = 'en') -> List[Dict]:
        """Search for medical videos across platforms"""
        logger.info(f"Searching for medical videos: {query} (language: {language})")
        
        results = []
        platforms = self.video_platforms.get(language, self.video_platforms['en'])
        
        for platform in platforms:
            if len(results) >= num_results:
                break
            
            try:
                platform_results = self._search_platform(query, platform, num_results - len(results))
                results.extend(platform_results)
                time.sleep(0.5)  # Rate limiting
            except Exception as e:
                logger.warning(f"Video search failed for {platform['name']}: {e}")
                continue
        
        # Add fallback video sources if needed
        if len(results) < num_results:
            fallback_results = self._get_fallback_videos(query, language, num_results - len(results))
            results.extend(fallback_results)
        
        logger.info(f"Found {len(results)} video results")
        return results[:num_results]
    
    def _search_platform(self, query: str, platform: Dict, num_results: int) -> List[Dict]:
        """Search a specific video platform"""
        try:
            search_url = platform['search_url']
            params = platform['params'].copy()
            
            # Set search query parameter
            for param_name in params.keys():
                params[param_name] = query
            
            response = self.session.get(search_url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            results = []
            
            # Try platform-specific selectors
            selectors = platform.get('selectors', ['a[href*="video"]', 'a[href*="watch"]'])
            
            for selector in selectors:
                links = soup.select(selector)
                if links:
                    logger.info(f"{platform['name']} found {len(links)} video links with selector: {selector}")
                    break
            
            for link in links[:num_results]:
                try:
                    href = link.get('href')
                    if not href:
                        continue
                    
                    # Make absolute URL
                    if href.startswith('/'):
                        href = platform['base_url'] + href
                    
                    title = link.get_text(strip=True)
                    if title and href:
                        results.append({
                            'url': href,
                            'title': title,
                            'platform': platform['name'],
                            'type': 'video',
                            'source': platform['name']
                        })
                except Exception as e:
                    logger.debug(f"Error parsing {platform['name']} link: {e}")
                    continue
            
            return results
            
        except Exception as e:
            logger.warning(f"Platform {platform['name']} search failed: {e}")
            return []
    
    def _get_fallback_videos(self, query: str, language: str, num_results: int) -> List[Dict]:
        """Get fallback video sources when direct search fails"""
        fallback_videos = {
            'en': [
                {
                    'url': 'https://www.youtube.com/results?search_query=medical+' + quote(query),
                    'title': f'Medical Videos: {query}',
                    'platform': 'youtube_fallback',
                    'type': 'video',
                    'source': 'youtube'
                },
                {
                    'url': 'https://www.medscape.com/search?q=' + quote(query),
                    'title': f'Medscape Videos: {query}',
                    'platform': 'medscape_fallback',
                    'type': 'video',
                    'source': 'medscape'
                }
            ],
            'vi': [
                {
                    'url': 'https://www.youtube.com/results?search_query=y+tế+' + quote(query),
                    'title': f'Video Y Tế: {query}',
                    'platform': 'youtube_vi_fallback',
                    'type': 'video',
                    'source': 'youtube'
                },
                {
                    'url': 'https://www.vinmec.com/vi/suc-khoe',
                    'title': f'Vinmec Videos: {query}',
                    'platform': 'vinmec_fallback',
                    'type': 'video',
                    'source': 'vinmec'
                }
            ],
            'zh': [
                {
                    'url': 'https://www.youtube.com/results?search_query=医疗+' + quote(query),
                    'title': f'医疗视频: {query}',
                    'platform': 'youtube_zh_fallback',
                    'type': 'video',
                    'source': 'youtube'
                },
                {
                    'url': 'https://www.haodf.com/jibing',
                    'title': f'好大夫视频: {query}',
                    'platform': 'haodf_fallback',
                    'type': 'video',
                    'source': 'haodf'
                }
            ]
        }
        
        return fallback_videos.get(language, fallback_videos['en'])[:num_results]