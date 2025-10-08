import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin, urlparse
import time
import logging
from typing import List, Dict, Tuple, Set
from models import summarizer
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
            
            # Prefer main/article content if available
            main = soup.find('main') or soup.find('article') or soup.find('div', {'role': 'main'})
            text = (main.get_text(separator=' ') if main else soup.get_text(separator=' '))
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Keep larger text; we'll chunk later
            if len(text) > 20000:
                text = text[:20000]
                
            return text
            
        except Exception as e:
            logger.warning(f"Failed to extract content from {url}: {e}")
            return ""
    
    def _chunk_text(self, text: str, chunk_size: int = 1200, overlap: int = 150) -> List[str]:
        """Chunk large texts with overlap for LLM summarization."""
        chunks = []
        start = 0
        n = len(text)
        while start < n:
            end = min(start + chunk_size, n)
            chunk = text[start:end]
            chunks.append(chunk)
            if end == n:
                break
            start = end - overlap
            if start < 0:
                start = 0
        return chunks

    def _summarize_relevant(self, text: str, query: str) -> str:
        """Summarize only query-relevant facts from a text chunk using NVIDIA Llama."""
        return summarizer.summarize_for_query(text, query, max_length=260)

    def _expand_intrasite_links(self, base_url: str, soup: BeautifulSoup, limit: int = 3) -> List[str]:
        """Collect a few same-domain links for deeper crawl (e.g., subpages, sections)."""
        try:
            parsed_base = urlparse(base_url)
            base_domain = parsed_base.netloc
            links: List[str] = []
            seen: Set[str] = set()
            for a in soup.find_all('a', href=True):
                href = a['href']
                if href.startswith('#'):
                    continue
                abs_url = urljoin(base_url, href)
                parsed = urlparse(abs_url)
                if parsed.netloc != base_domain:
                    continue
                if abs_url in seen:
                    continue
                seen.add(abs_url)
                links.append(abs_url)
                if len(links) >= limit:
                    break
            return links
        except Exception:
            return []

    def search_and_extract(self, query: str, num_results: int = 10) -> List[Dict]:
        """Search for query and extract content from top results"""
        logger.info(f"Searching for: {query}")
        
        # Aggregate from multiple sources to hit more sites
        ddg_results = self.search_duckduckgo(query, min(num_results * 2, 20))
        # TODO: add additional engines/APIs if available in env (e.g., SerpAPI, Bing). For now, DDG only.
        search_results = ddg_results
        
        # Extract content from each result with parallel processing
        enriched_results = []
        failed_count = 0
        max_failures = 5  # Stop after 5 consecutive failures
        
        for i, result in enumerate(search_results):
            if len(enriched_results) >= num_results:
                break
                
            if failed_count >= max_failures:
                logger.warning(f"Too many failures ({failed_count}), stopping extraction")
                break
                
            try:
                logger.info(f"Extracting content from {result['url']}")
                # Fetch HTML once to support intrasite expansion
                resp = self.session.get(result['url'], timeout=self.timeout)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.content, 'html.parser')
                content = self.extract_content(result['url'])
                
                relevant_snippets: List[str] = []
                if content and len(content.strip()) > 50:
                    # Chunk and summarize only relevant parts
                    for chunk in self._chunk_text(content, chunk_size=1400, overlap=200):
                        rel = self._summarize_relevant(chunk, query)
                        if rel:
                            relevant_snippets.append(rel)

                # Try a few intrasite links (same domain) to gather more context
                for extra_url in self._expand_intrasite_links(result['url'], soup, limit=2):
                    try:
                        extra_text = self.extract_content(extra_url)
                        if not extra_text:
                            continue
                        for chunk in self._chunk_text(extra_text, chunk_size=1200, overlap=150):
                            rel = self._summarize_relevant(chunk, query)
                            if rel:
                                relevant_snippets.append(rel)
                        # Be polite between requests
                        time.sleep(0.3)
                    except Exception as e:
                        logger.debug(f"Failed extra link {extra_url}: {e}")

                # Only keep entries with relevant snippets
                if relevant_snippets:
                    enriched_results.append({
                        'id': len(enriched_results) + 1,
                        'url': result['url'],
                        'title': result['title'],
                        'content': " \n".join(relevant_snippets)[:2000]
                    })
                    failed_count = 0  # Reset failure counter
                else:
                    failed_count += 1
                    logger.warning(f"No query-relevant content from {result['url']}")
                
                # Add delay to be respectful
                time.sleep(0.4)
                
            except Exception as e:
                failed_count += 1
                logger.warning(f"Failed to process {result['url']}: {e}")
                continue
        
        logger.info(f"Successfully processed {len(enriched_results)} results out of {len(search_results)} attempted")
        return enriched_results

def search_web(query: str, num_results: int = 10) -> List[Dict]:
    """Main function to search the web and return enriched results"""
    searcher = WebSearcher()
    return searcher.search_and_extract(query, num_results)
