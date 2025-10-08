import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin, urlparse
import time
import logging
from typing import List, Dict, Tuple
import os

logger = logging.getLogger(__name__)

class WebSearcher:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.max_results = 10
        self.timeout = 10
        
    def search_google(self, query: str, num_results: int = 10) -> List[Dict]:
        """Search Google and return results with URLs and titles"""
        try:
            # Use DuckDuckGo as it's more reliable for scraping
            return self.search_duckduckgo(query, num_results)
        except Exception as e:
            logger.error(f"Google search failed: {e}")
            return []
    
    def search_duckduckgo(self, query: str, num_results: int = 10) -> List[Dict]:
        """Search DuckDuckGo and return results"""
        try:
            url = "https://html.duckduckgo.com/html/"
            params = {
                'q': query,
                'kl': 'us-en'
            }
            
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            results = []
            
            # Find result links
            result_links = soup.find_all('a', class_='result__a')
            
            for link in result_links[:num_results]:
                try:
                    href = link.get('href')
                    if href and href.startswith('http'):
                        title = link.get_text(strip=True)
                        if title and href:
                            results.append({
                                'url': href,
                                'title': title,
                                'content': ''  # Will be filled later
                            })
                except Exception as e:
                    logger.warning(f"Error parsing result: {e}")
                    continue
                    
            return results
            
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
            return []
    
    def extract_content(self, url: str) -> str:
        """Extract text content from a webpage"""
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Limit content length
            if len(text) > 2000:
                text = text[:2000] + "..."
                
            return text
            
        except Exception as e:
            logger.warning(f"Failed to extract content from {url}: {e}")
            return ""
    
    def search_and_extract(self, query: str, num_results: int = 5) -> List[Dict]:
        """Search for query and extract content from top results"""
        logger.info(f"Searching for: {query}")
        
        # Get search results
        search_results = self.search_duckduckgo(query, num_results)
        
        # Extract content from each result
        enriched_results = []
        for i, result in enumerate(search_results):
            try:
                logger.info(f"Extracting content from {result['url']}")
                content = self.extract_content(result['url'])
                
                if content:
                    enriched_results.append({
                        'id': i + 1,
                        'url': result['url'],
                        'title': result['title'],
                        'content': content
                    })
                
                # Add delay to be respectful
                time.sleep(1)
                
            except Exception as e:
                logger.warning(f"Failed to process {result['url']}: {e}")
                continue
        
        logger.info(f"Successfully processed {len(enriched_results)} results")
        return enriched_results

def search_web(query: str, num_results: int = 5) -> List[Dict]:
    """Main function to search the web and return enriched results"""
    searcher = WebSearcher()
    return searcher.search_and_extract(query, num_results)
