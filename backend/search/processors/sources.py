import logging
from typing import List, Dict, Tuple, Set
import re
from urllib.parse import urlparse
from collections import defaultdict

logger = logging.getLogger(__name__)

class SourceAggregator:
    """Aggregate and process sources for comprehensive information extraction"""
    
    def __init__(self):
        # Source credibility scoring
        self.source_credibility = {
            # English sources
            'mayoclinic.org': 0.95,
            'webmd.com': 0.90,
            'healthline.com': 0.88,
            'medlineplus.gov': 0.95,
            'nih.gov': 0.98,
            'cdc.gov': 0.98,
            'who.int': 0.97,
            'pubmed.ncbi.nlm.nih.gov': 0.96,
            'uptodate.com': 0.94,
            'merckmanuals.com': 0.92,
            'medscape.com': 0.89,
            
            # Vietnamese sources
            'hellobacsi.com': 0.85,
            'alobacsi.com': 0.82,
            'vinmec.com': 0.88,
            'tamanhhospital.vn': 0.85,
            'medlatec.vn': 0.83,
            'suckhoedoisong.vn': 0.90,
            'viendinhduong.vn': 0.87,
            
            # Chinese sources
            'haodf.com': 0.86,
            'dxy.cn': 0.89,
            'chunyuyisheng.com': 0.84,
            'xywy.com': 0.82,
            'jiankang.com': 0.80,
            'familydoctor.com.cn': 0.85,
            
            # Video platforms
            'youtube.com': 0.70,
            'medscape.com': 0.89
        }
        
        # Source type classification
        self.source_types = {
            'academic': ['nih.gov', 'pubmed.ncbi.nlm.nih.gov', 'who.int', 'cdc.gov'],
            'hospital': ['mayoclinic.org', 'vinmec.com', 'tamanhhospital.vn'],
            'commercial': ['webmd.com', 'healthline.com', 'hellobacsi.com'],
            'government': ['medlineplus.gov', 'suckhoedoisong.vn', 'viendinhduong.vn'],
            'professional': ['dxy.cn', 'medscape.com', 'uptodate.com'],
            'video': ['youtube.com', 'medscape.com']
        }
    
    def aggregate_sources(self, search_results: List[Dict], video_results: List[Dict] = None) -> Dict[str, any]:
        """Aggregate all sources and create comprehensive reference system"""
        all_sources = []
        
        # Process search results
        for result in search_results:
            source_info = self._process_source(result)
            if source_info:
                all_sources.append(source_info)
        
        # Process video results
        if video_results:
            for video in video_results:
                video_info = self._process_video_source(video)
                if video_info:
                    all_sources.append(video_info)
        
        # Remove duplicates and score sources
        unique_sources = self._deduplicate_sources(all_sources)
        scored_sources = self._score_sources(unique_sources)
        
        # Create comprehensive reference mapping
        reference_mapping = self._create_reference_mapping(scored_sources)
        
        # Generate source summary
        source_summary = self._generate_source_summary(scored_sources)
        
        return {
            'sources': scored_sources,
            'reference_mapping': reference_mapping,
            'source_summary': source_summary,
            'total_sources': len(scored_sources),
            'languages': self._get_language_distribution(scored_sources),
            'source_types': self._get_source_type_distribution(scored_sources)
        }
    
    def _process_source(self, result: Dict) -> Dict:
        """Process a single search result into standardized source format"""
        url = result.get('url', '')
        if not url:
            return None
        
        domain = self._extract_domain(url)
        source_type = self._classify_source_type(domain)
        credibility = self._get_source_credibility(domain)
        
        return {
            'url': url,
            'title': result.get('title', ''),
            'content': result.get('content', ''),
            'domain': domain,
            'source_type': source_type,
            'credibility_score': credibility,
            'language': result.get('language', 'en'),
            'source_name': result.get('source', ''),
            'platform': result.get('platform', ''),
            'type': 'text'
        }
    
    def _process_video_source(self, video: Dict) -> Dict:
        """Process a video result into standardized source format"""
        url = video.get('url', '')
        if not url:
            return None
        
        domain = self._extract_domain(url)
        source_type = 'video'
        credibility = self._get_source_credibility(domain)
        
        return {
            'url': url,
            'title': video.get('title', ''),
            'content': '',  # Videos don't have text content
            'domain': domain,
            'source_type': source_type,
            'credibility_score': credibility,
            'language': video.get('language', 'en'),
            'source_name': video.get('source', ''),
            'platform': video.get('platform', ''),
            'type': 'video'
        }
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            # Remove www. prefix
            if domain.startswith('www.'):
                domain = domain[4:]
            return domain
        except:
            return ''
    
    def _classify_source_type(self, domain: str) -> str:
        """Classify source type based on domain"""
        for source_type, domains in self.source_types.items():
            if domain in domains:
                return source_type
        return 'other'
    
    def _get_source_credibility(self, domain: str) -> float:
        """Get credibility score for domain"""
        return self.source_credibility.get(domain, 0.70)  # Default score
    
    def _deduplicate_sources(self, sources: List[Dict]) -> List[Dict]:
        """Remove duplicate sources based on URL and title similarity"""
        seen_urls = set()
        seen_titles = set()
        unique_sources = []
        
        for source in sources:
            url = source.get('url', '')
            title = source.get('title', '').lower().strip()
            
            # Check for URL duplicates
            if url in seen_urls:
                continue
            
            # Check for title similarity (fuzzy matching)
            title_similar = any(self._titles_similar(title, seen_title) for seen_title in seen_titles)
            if title_similar:
                continue
            
            seen_urls.add(url)
            seen_titles.add(title)
            unique_sources.append(source)
        
        return unique_sources
    
    def _titles_similar(self, title1: str, title2: str, threshold: float = 0.8) -> bool:
        """Check if two titles are similar (simple word overlap)"""
        if not title1 or not title2:
            return False
        
        words1 = set(title1.split())
        words2 = set(title2.split())
        
        if not words1 or not words2:
            return False
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        similarity = len(intersection) / len(union) if union else 0
        return similarity >= threshold
    
    def _score_sources(self, sources: List[Dict]) -> List[Dict]:
        """Score and rank sources by relevance and credibility"""
        for source in sources:
            # Calculate composite score
            credibility = source.get('credibility_score', 0.5)
            content_length = len(source.get('content', ''))
            title_length = len(source.get('title', ''))
            
            # Content quality score
            content_score = min(content_length / 1000, 1.0)  # Normalize to 0-1
            
            # Title quality score
            title_score = min(title_length / 100, 1.0)  # Normalize to 0-1
            
            # Composite score (weighted)
            composite_score = (
                credibility * 0.5 +      # 50% credibility
                content_score * 0.3 +    # 30% content quality
                title_score * 0.2        # 20% title quality
            )
            
            source['composite_score'] = composite_score
        
        # Sort by composite score
        sources.sort(key=lambda x: x.get('composite_score', 0), reverse=True)
        
        return sources
    
    def _create_reference_mapping(self, sources: List[Dict]) -> Dict[int, Dict]:
        """Create reference mapping for citations"""
        reference_mapping = {}
        
        for i, source in enumerate(sources, 1):
            reference_mapping[i] = {
                'url': source['url'],
                'title': source['title'],
                'domain': source['domain'],
                'source_type': source['source_type'],
                'credibility_score': source['credibility_score'],
                'language': source['language'],
                'type': source['type']
            }
        
        return reference_mapping
    
    def _generate_source_summary(self, sources: List[Dict]) -> str:
        """Generate summary of sources used"""
        if not sources:
            return "No sources available."
        
        # Group by source type
        type_counts = defaultdict(int)
        language_counts = defaultdict(int)
        high_credibility_count = 0
        
        for source in sources:
            source_type = source.get('source_type', 'other')
            language = source.get('language', 'en')
            credibility = source.get('credibility_score', 0)
            
            type_counts[source_type] += 1
            language_counts[language] += 1
            
            if credibility >= 0.8:
                high_credibility_count += 1
        
        # Generate summary
        summary_parts = []
        summary_parts.append(f"**Sources Used ({len(sources)} total):**")
        
        # Source types
        if type_counts:
            type_summary = ", ".join([f"{count} {type_name}" for type_name, count in type_counts.items()])
            summary_parts.append(f"• **Types**: {type_summary}")
        
        # Languages
        if language_counts:
            lang_summary = ", ".join([f"{count} {lang}" for lang, count in language_counts.items()])
            summary_parts.append(f"• **Languages**: {lang_summary}")
        
        # Credibility
        summary_parts.append(f"• **High Credibility Sources**: {high_credibility_count}/{len(sources)} (≥80% credibility)")
        
        return "\n".join(summary_parts)
    
    def _get_language_distribution(self, sources: List[Dict]) -> Dict[str, int]:
        """Get distribution of sources by language"""
        distribution = defaultdict(int)
        for source in sources:
            language = source.get('language', 'en')
            distribution[language] += 1
        return dict(distribution)
    
    def _get_source_type_distribution(self, sources: List[Dict]) -> Dict[str, int]:
        """Get distribution of sources by type"""
        distribution = defaultdict(int)
        for source in sources:
            source_type = source.get('source_type', 'other')
            distribution[source_type] += 1
        return dict(distribution)
    
    def create_comprehensive_references(self, sources: List[Dict], max_references: int = 15) -> str:
        """Create comprehensive reference list for the response"""
        if not sources:
            return ""
        
        # Take top sources
        top_sources = sources[:max_references]
        
        reference_parts = []
        reference_parts.append("**📚 References:**")
        
        for i, source in enumerate(top_sources, 1):
            url = source['url']
            title = source['title']
            domain = source['domain']
            source_type = source['source_type']
            credibility = source['credibility_score']
            language = source['language']
            source_type_icon = source['type']
            
            # Create credibility indicator
            if credibility >= 0.9:
                cred_indicator = "🟢"
            elif credibility >= 0.8:
                cred_indicator = "🟡"
            else:
                cred_indicator = "🔴"
            
            # Create type indicator
            type_icons = {
                'academic': '🎓',
                'hospital': '🏥',
                'government': '🏛️',
                'commercial': '💼',
                'professional': '👨‍⚕️',
                'video': '📹',
                'other': '📄'
            }
            type_icon = type_icons.get(source_type, '📄')
            
            # Create language indicator
            lang_icons = {
                'en': '🇺🇸',
                'vi': '🇻🇳',
                'zh': '🇨🇳'
            }
            lang_icon = lang_icons.get(language, '🌐')
            
            reference_line = f"{i}. {type_icon} {lang_icon} {cred_indicator} [{title}]({url}) - {domain}"
            reference_parts.append(reference_line)
        
        if len(sources) > max_references:
            reference_parts.append(f"... and {len(sources) - max_references} more sources")
        
        return "\n".join(reference_parts)

