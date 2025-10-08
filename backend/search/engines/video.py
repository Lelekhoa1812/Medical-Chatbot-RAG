import requests
import logging
import re
from typing import List, Dict, Optional
from urllib.parse import quote, urlparse
import time

logger = logging.getLogger(__name__)

class VideoSearchEngine:
    """Video search engine for medical content from YouTube and other video sources"""
    
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
        
        # Video sources by language
        self.video_sources = {
            'en': {
                'youtube': {
                    'base_url': 'https://www.youtube.com',
                    'search_url': 'https://www.youtube.com/results',
                    'params': {'search_query': '', 'sp': 'EgIQAQ%253D%253D'},  # Videos only
                    'selectors': {
                        'title': 'h3 a#video-title',
                        'url': 'h3 a#video-title',
                        'thumbnail': 'img',
                        'duration': '.ytd-thumbnail-overlay-time-status-renderer',
                        'channel': 'a#channel-name'
                    }
                },
                'vimeo': {
                    'base_url': 'https://vimeo.com',
                    'search_url': 'https://vimeo.com/search',
                    'params': {'q': ''},
                    'selectors': {
                        'title': '.clip a',
                        'url': '.clip a',
                        'thumbnail': '.clip img',
                        'duration': '.clip .duration',
                        'channel': '.clip .byline a'
                    }
                },
                'ted': {
                    'base_url': 'https://www.ted.com',
                    'search_url': 'https://www.ted.com/search',
                    'params': {'q': '', 'cat': 'talks'},
                    'selectors': {
                        'title': '.search__result__title a',
                        'url': '.search__result__title a',
                        'thumbnail': '.search__result img',
                        'duration': '.search__result .duration',
                        'channel': '.search__result .speaker'
                    }
                }
            },
            'vi': {
                'youtube': {
                    'base_url': 'https://www.youtube.com',
                    'search_url': 'https://www.youtube.com/results',
                    'params': {'search_query': '', 'sp': 'EgIQAQ%253D%253D'},
                    'selectors': {
                        'title': 'h3 a#video-title',
                        'url': 'h3 a#video-title',
                        'thumbnail': 'img',
                        'duration': '.ytd-thumbnail-overlay-time-status-renderer',
                        'channel': 'a#channel-name'
                    }
                },
                'fpt_play': {
                    'base_url': 'https://fptplay.vn',
                    'search_url': 'https://fptplay.vn/tim-kiem',
                    'params': {'q': ''},
                    'selectors': {
                        'title': '.video-item .title a',
                        'url': '.video-item .title a',
                        'thumbnail': '.video-item img',
                        'duration': '.video-item .duration',
                        'channel': '.video-item .channel'
                    }
                }
            },
            'zh': {
                'youtube': {
                    'base_url': 'https://www.youtube.com',
                    'search_url': 'https://www.youtube.com/results',
                    'params': {'search_query': '', 'sp': 'EgIQAQ%253D%253D'},
                    'selectors': {
                        'title': 'h3 a#video-title',
                        'thumbnail': 'img',
                        'duration': '.ytd-thumbnail-overlay-time-status-renderer',
                        'channel': 'a#channel-name'
                    }
                },
                'bilibili': {
                    'base_url': 'https://www.bilibili.com',
                    'search_url': 'https://search.bilibili.com/video',
                    'params': {'keyword': ''},
                    'selectors': {
                        'title': '.video-item .title a',
                        'url': '.video-item .title a',
                        'thumbnail': '.video-item img',
                        'duration': '.video-item .duration',
                        'channel': '.video-item .up-name'
                    }
                },
                'youku': {
                    'base_url': 'https://www.youku.com',
                    'search_url': 'https://search.youku.com/search_video',
                    'params': {'keyword': ''},
                    'selectors': {
                        'title': '.video-item .title a',
                        'url': '.video-item .title a',
                        'thumbnail': '.video-item img',
                        'duration': '.video-item .duration',
                        'channel': '.video-item .channel'
                    }
                }
            }
        }
    
    def search(self, query: str, num_results: int = 3, language: str = 'en') -> List[Dict]:
        """Search for medical videos across multiple platforms"""
        logger.info(f"Searching for medical videos: {query} (language: {language})")
        
        # Enhance query for medical content
        enhanced_query = self._enhance_medical_query(query, language)
        
        all_results = []
        
        # Get sources for the language
        sources = self.video_sources.get(language, self.video_sources['en'])
        
        for source_name, source_config in sources.items():
            if len(all_results) >= num_results:
                break
                
            try:
                source_results = self._search_source(enhanced_query, source_name, source_config, language)
                all_results.extend(source_results)
                time.sleep(0.5)  # Rate limiting
            except Exception as e:
                logger.warning(f"Video search failed for {source_name}: {e}")
                continue
        
        # Remove duplicates and return top results
        unique_results = self._remove_duplicates(all_results)
        return unique_results[:num_results]
    
    def _enhance_medical_query(self, query: str, language: str) -> str:
        """Enhance query with medical-specific terms"""
        medical_terms = {
            'en': ['medical', 'health', 'doctor', 'treatment', 'symptoms', 'diagnosis'],
            'vi': ['y tế', 'sức khỏe', 'bác sĩ', 'điều trị', 'triệu chứng', 'chẩn đoán'],
            'zh': ['医疗', '健康', '医生', '治疗', '症状', '诊断']
        }
        
        terms = medical_terms.get(language, medical_terms['en'])
        
        # Add medical context if not already present
        query_lower = query.lower()
        has_medical_context = any(term in query_lower for term in terms)
        
        if not has_medical_context:
            # Add the most relevant medical term
            if language == 'vi':
                enhanced = f"{query} y tế sức khỏe"
            elif language == 'zh':
                enhanced = f"{query} 医疗健康"
            else:
                enhanced = f"{query} medical health"
        else:
            enhanced = query
        
        return enhanced
    
    def _search_source(self, query: str, source_name: str, source_config: Dict, language: str) -> List[Dict]:
        """Search a specific video source"""
        try:
            search_url = source_config['search_url']
            params = source_config['params'].copy()
            
            # Set search parameter based on source
            if 'search_query' in params:
                params['search_query'] = query
            elif 'q' in params:
                params['q'] = query
            elif 'keyword' in params:
                params['keyword'] = query
            
            response = self.session.get(search_url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            # Parse results (simplified - in production, you'd use proper HTML parsing)
            results = self._parse_video_results(response.text, source_config, source_name, language)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching {source_name}: {e}")
            return []
    
    def _parse_video_results(self, html_content: str, source_config: Dict, source_name: str, language: str) -> List[Dict]:
        """Parse video results from HTML content"""
        results = []
        
        # This is a simplified parser - in production, you'd use BeautifulSoup or similar
        # For now, we'll create mock results based on the query
        
        # Extract basic video information using regex patterns
        title_pattern = r'<title[^>]*>([^<]+)</title>'
        titles = re.findall(title_pattern, html_content, re.IGNORECASE)
        
        # Create mock results for demonstration
        # In production, you'd extract real video data
        mock_videos = self._get_mock_video_results(source_name, language)
        
        for i, video in enumerate(mock_videos[:3]):  # Limit to 3 results per source
            results.append({
                'title': video['title'],
                'url': video['url'],
                'thumbnail': video['thumbnail'],
                'duration': video['duration'],
                'channel': video['channel'],
                'source': source_name,
                'language': language,
                'type': 'video'
            })
        
        return results
    
    def _get_mock_video_results(self, source_name: str, language: str) -> List[Dict]:
        """Get mock video results for demonstration"""
        # In production, this would be replaced with actual video parsing
        mock_results = {
            'youtube': {
                'en': [
                    {
                        'title': 'Medical Diagnosis and Treatment Explained',
                        'url': 'https://www.youtube.com/watch?v=example1',
                        'thumbnail': 'https://img.youtube.com/vi/example1/mqdefault.jpg',
                        'duration': '12:34',
                        'channel': 'Medical Education Channel'
                    },
                    {
                        'title': 'Understanding Common Medical Symptoms',
                        'url': 'https://www.youtube.com/watch?v=example2',
                        'thumbnail': 'https://img.youtube.com/vi/example2/mqdefault.jpg',
                        'duration': '8:45',
                        'channel': 'Health & Wellness'
                    },
                    {
                        'title': 'Doctor Explains Treatment Options',
                        'url': 'https://www.youtube.com/watch?v=example3',
                        'thumbnail': 'https://img.youtube.com/vi/example3/mqdefault.jpg',
                        'duration': '15:20',
                        'channel': 'Medical Professionals'
                    }
                ],
                'vi': [
                    {
                        'title': 'Chẩn đoán và điều trị y tế',
                        'url': 'https://www.youtube.com/watch?v=example_vi1',
                        'thumbnail': 'https://img.youtube.com/vi/example_vi1/mqdefault.jpg',
                        'duration': '10:15',
                        'channel': 'Kênh Y Tế Việt Nam'
                    },
                    {
                        'title': 'Hiểu về các triệu chứng y tế thường gặp',
                        'url': 'https://www.youtube.com/watch?v=example_vi2',
                        'thumbnail': 'https://img.youtube.com/vi/example_vi2/mqdefault.jpg',
                        'duration': '7:30',
                        'channel': 'Sức Khỏe & Đời Sống'
                    }
                ],
                'zh': [
                    {
                        'title': '医疗诊断和治疗解释',
                        'url': 'https://www.youtube.com/watch?v=example_zh1',
                        'thumbnail': 'https://img.youtube.com/vi/example_zh1/mqdefault.jpg',
                        'duration': '11:25',
                        'channel': '医疗教育频道'
                    },
                    {
                        'title': '了解常见医疗症状',
                        'url': 'https://www.youtube.com/watch?v=example_zh2',
                        'thumbnail': 'https://img.youtube.com/vi/example_zh2/mqdefault.jpg',
                        'duration': '9:10',
                        'channel': '健康与医疗'
                    }
                ]
            },
            'ted': {
                'en': [
                    {
                        'title': 'The Future of Medical Technology',
                        'url': 'https://www.ted.com/talks/example_ted1',
                        'thumbnail': 'https://pi.tedcdn.com/r/example_ted1.jpg',
                        'duration': '18:45',
                        'channel': 'TED Talks'
                    }
                ]
            }
        }
        
        return mock_results.get(source_name, {}).get(language, [])
    
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
    
    def search_youtube_direct(self, query: str, num_results: int = 3) -> List[Dict]:
        """Direct YouTube search using their API (requires API key)"""
        # This would use YouTube Data API v3 in production
        # For now, return mock results
        logger.info(f"Direct YouTube search for: {query}")
        
        mock_results = [
            {
                'title': f'Medical Video: {query}',
                'url': f'https://www.youtube.com/watch?v=youtube_{i}',
                'thumbnail': f'https://img.youtube.com/vi/youtube_{i}/mqdefault.jpg',
                'duration': f'{10+i}:{30+i}',
                'channel': 'Medical Channel',
                'source': 'youtube',
                'type': 'video'
            }
            for i in range(1, num_results + 1)
        ]
        
        return mock_results
