import requests
from bs4 import BeautifulSoup
import logging
from typing import List, Dict
import time
import re
from urllib.parse import urlparse, quote
from models.reranker import MedicalReranker

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
        self.reranker = MedicalReranker()
        
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
    
    def _normalize_query(self, q: str) -> str:
        if not q:
            return ""
        q = q.strip()
        q = re.sub(r"^(en|vi|zh)\s*:\s*", "", q, flags=re.IGNORECASE)
        # Remove bullet points and special characters
        q = re.sub(r'[•·▪▫‣⁃]', ' ', q)
        q = re.sub(r'[^\w\s\-\.]', ' ', q)
        q = re.sub(r"\s+", " ", q)
        return q.strip()

    def _is_valid_medical_video(self, result: Dict, query: str) -> bool:
        """Check if video is medically relevant and has valid URL"""
        url = result.get('url', '')
        title = result.get('title', '')
        
        # Skip generic YouTube search result pages
        if 'results?search_query=' in url:
            return False
        
        # Skip non-YouTube URLs that aren't medical platforms
        if 'youtube.com' not in url and not any(med in url for med in ['medscape.com', 'vinmec.com', 'haodf.com']):
            return False
        
        # Check if title contains medical keywords or query terms
        title_lower = title.lower()
        query_lower = query.lower()
        
        medical_keywords = [
            'medical', 'health', 'doctor', 'treatment', 'diagnosis',
            'symptoms', 'therapy', 'medicine', 'clinical', 'patient',
            'disease', 'condition', 'healthcare', 'physician'
        ]
        
        # Must contain medical keywords or query terms
        has_medical = any(keyword in title_lower for keyword in medical_keywords)
        has_query = any(word in title_lower for word in query_lower.split() if len(word) > 3)
        
        return has_medical or has_query

    def _search_platform_with_retry(self, query: str, platform: Dict, num_results: int, max_retries: int = 2) -> List[Dict]:
        """Search platform with retry logic and better error handling"""
        for attempt in range(max_retries):
            try:
                return self._search_platform(query, platform, num_results)
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {platform['name']}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)  # Wait before retry
                else:
                    logger.error(f"All attempts failed for {platform['name']}")
        return []

    def search(self, query: str, num_results: int = 3, language: str = 'en') -> List[Dict]:
        """Search for medical videos across platforms with deduplication and medical filtering"""
        query = self._normalize_query(query)
        logger.info(f"Searching for medical videos: {query} (language: {language})")
        
        results = []
        seen_urls = set()  # Track URLs to avoid duplicates
        seen_video_ids = set()  # Track video IDs to avoid duplicates
        platforms = self.video_platforms.get(language, self.video_platforms['en'])
        
        # Try platforms in order of reliability
        for platform in platforms:
            if len(results) >= num_results:
                break
            
            try:
                # Add timeout and retry logic
                platform_results = self._search_platform_with_retry(query, platform, num_results * 3)
                
                if not platform_results:
                    logger.warning(f"No results from {platform['name']}")
                    continue
                
                # Filter out duplicates and non-medical content
                for result in platform_results:
                    url = result.get('url', '')
                    video_id = self._extract_video_id(url)
                    
                    # Skip if URL or video ID already seen
                    if url in seen_urls or (video_id and video_id in seen_video_ids):
                        continue
                    
                    # Check if it's a valid medical video (less strict for more results)
                    if self._is_valid_medical_video(result, query):
                        seen_urls.add(url)
                        if video_id:
                            seen_video_ids.add(video_id)
                        
                        # Normalize YouTube URLs
                        if video_id and 'youtube.com' in url:
                            result['url'] = f"https://www.youtube.com/watch?v={video_id}"
                            result['video_id'] = video_id
                        
                        results.append(result)
                        if len(results) >= num_results:
                            break
                
                time.sleep(0.5)  # Rate limiting
            except Exception as e:
                logger.warning(f"Video search failed for {platform['name']}: {e}")
                continue
        
        # Add fallback video sources if needed
        if len(results) < num_results:
            # Try resilient YouTube via Invidious API
            try:
                resilient = self._search_youtube_invidious(query, language, num_results - len(results))
                for result in resilient:
                    url = result.get('url', '')
                    video_id = result.get('video_id', '')
                    
                    if (url not in seen_urls and 
                        video_id not in seen_video_ids and 
                        self._is_valid_medical_video(result, query)):
                        seen_urls.add(url)
                        if video_id:
                            seen_video_ids.add(video_id)
                        results.append(result)
                        if len(results) >= num_results:
                            break
            except Exception as e:
                logger.warning(f"Invidious fallback failed: {e}")
            
            # If still no results, try generic video search fallback
            if len(results) < num_results:
                try:
                    fallback_results = self._get_fallback_videos(query, language, num_results - len(results))
                    for result in fallback_results:
                        if result['url'] not in seen_urls:
                            seen_urls.add(result['url'])
                            results.append(result)
                            if len(results) >= num_results:
                                break
                    logger.info(f"Added {len(fallback_results)} fallback video results")
                except Exception as e:
                    logger.warning(f"Fallback video search failed: {e}")
        
        # Use reranker to improve quality and relevance
        if results:
            reranked_results = self.reranker.filter_youtube_results(results, query)
            logger.info(f"Reranked {len(results)} video results to {len(reranked_results)} high-quality results")
            return reranked_results[:num_results]
        
        logger.info(f"Found {len(results)} medical video results")
        return results[:num_results]
    
    def _search_platform(self, query: str, platform: Dict, num_results: int) -> List[Dict]:
        """Search a specific video platform with improved error handling"""
        try:
            search_url = platform['search_url']
            params = platform['params'].copy()
            
            # Set search query parameter
            for param_name in params.keys():
                params[param_name] = query
            
            # Add headers to avoid blocking
            headers = self.session.headers.copy()
            headers.update({
                'Referer': 'https://www.google.com/',
                'Cache-Control': 'no-cache',
            })
            
            # Try with shorter timeout first
            response = self.session.get(search_url, params=params, headers=headers, timeout=10)
            
            # Check for common error responses
            if response.status_code == 404:
                logger.warning(f"Platform {platform['name']} returned 404 - endpoint may have changed")
                return []
            elif response.status_code == 403:
                logger.warning(f"Platform {platform['name']} returned 403 - may be blocking requests")
                return []
            elif response.status_code >= 400:
                logger.warning(f"Platform {platform['name']} returned {response.status_code}")
                return []
            
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            results = []
            
            # Try platform-specific selectors
            selectors = platform.get('selectors', ['a[href*="video"]', 'a[href*="watch"]'])
            
            links = []
            for selector in selectors:
                links = soup.select(selector)
                if links:
                    logger.info(f"{platform['name']} found {len(links)} video links with selector: {selector}")
                    break
            
            # If no links found, try generic selectors
            if not links:
                generic_selectors = ['a[href*="http"]', 'a[href*="www"]']
                for selector in generic_selectors:
                    links = soup.select(selector)
                    if links:
                        logger.info(f"{platform['name']} found {len(links)} generic links with selector: {selector}")
                        break
            
            for link in links[:num_results]:
                try:
                    href = link.get('href')
                    if not href:
                        continue
                    
                    # Make absolute URL
                    if href.startswith('/'):
                        href = platform['base_url'] + href
                    
                    # Skip if not a valid URL
                    if not href.startswith('http'):
                        continue
                    
                    title = link.get_text(strip=True) or platform['name']
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
            
        except requests.exceptions.Timeout:
            logger.warning(f"Platform {platform['name']} search timed out")
            return []
        except requests.exceptions.ConnectionError:
            logger.warning(f"Platform {platform['name']} connection failed - network issue")
            return []
        except Exception as e:
            logger.warning(f"Platform {platform['name']} search failed: {e}")
            return []

    def _search_youtube_invidious(self, query: str, language: str, needed: int) -> List[Dict]:
        """Search YouTube via public Invidious instances (no API key)."""
        if needed <= 0:
            return []
        instances = [
            "https://yewtu.be",
            "https://invidious.flokinet.to",
            "https://vid.puffyan.us",
            "https://iv.ggtyler.dev"
        ]
        out: List[Dict] = []
        q = quote(query)
        for base in instances:
            if len(out) >= needed:
                break
            try:
                url = f"{base}/api/v1/search?q={q}&region={'VN' if language=='vi' else 'US'}&fields=title,videoId,author&type=video"
                r = self.session.get(url, timeout=6)
                r.raise_for_status()
                data = r.json()
                for item in data:
                    if len(out) >= needed:
                        break
                    vid = item.get("videoId")
                    title = (item.get("title") or "").strip()
                    if not vid or not title:
                        continue
                    out.append({
                        'url': f"https://www.youtube.com/watch?v={vid}",
                        'title': title,
                        'thumbnail': f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg",
                        'platform': 'youtube',
                        'source': 'youtube',
                        'type': 'video',
                        'language': language
                    })
            except Exception as e:
                logger.debug(f"Invidious {base} failed: {e}")
                continue
        return out
    
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
    
    def _extract_video_id(self, url: str) -> str:
        """Extract YouTube video ID from URL"""
        patterns = [
            r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
            r'(?:embed\/)([0-9A-Za-z_-]{11})',
            r'(?:watch\?v=)([0-9A-Za-z_-]{11})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        return None

