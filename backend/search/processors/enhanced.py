import logging
from typing import List, Dict, Tuple, Set
import re
from collections import defaultdict
from models.summarizer import summarizer

logger = logging.getLogger(__name__)

class EnhancedContentProcessor:
    """Enhanced content processing for maximum information extraction"""
    
    def __init__(self):
        # Medical content patterns for extraction
        self.medical_patterns = {
            'symptoms': [
                r'symptoms?\s+(?:include|are|may include|can include)',
                r'common\s+symptoms?',
                r'signs?\s+(?:and\s+)?symptoms?',
                r'clinical\s+presentation',
                r'manifestations?'
            ],
            'causes': [
                r'causes?\s+(?:include|are|may include|can include)',
                r'risk\s+factors?',
                r'etiology',
                r'pathogenesis',
                r'underlying\s+causes?'
            ],
            'treatments': [
                r'treatment\s+(?:options?|include|are|may include)',
                r'therapy\s+(?:options?|include|are)',
                r'management\s+(?:options?|include|are)',
                r'interventions?',
                r'medications?',
                r'drugs?'
            ],
            'diagnosis': [
                r'diagnosis\s+(?:include|are|may include)',
                r'diagnostic\s+(?:tests?|procedures?|criteria)',
                r'testing\s+(?:include|are|may include)',
                r'evaluation\s+(?:include|are|may include)',
                r'assessment'
            ],
            'prevention': [
                r'prevention\s+(?:include|are|may include)',
                r'preventive\s+measures?',
                r'precautions?',
                r'risk\s+reduction',
                r'prophylaxis'
            ],
            'prognosis': [
                r'prognosis',
                r'outlook',
                r'course\s+of\s+disease',
                r'long-term\s+effects?',
                r'complications?'
            ]
        }
        
        # Content quality indicators
        self.quality_indicators = {
            'high': [
                'clinical trial', 'randomized', 'meta-analysis', 'systematic review',
                'evidence-based', 'peer-reviewed', 'published study', 'research shows',
                'clinical guidelines', 'medical consensus', 'expert opinion'
            ],
            'medium': [
                'studies show', 'research indicates', 'medical literature',
                'clinical experience', 'case studies', 'observational studies'
            ],
            'low': [
                'some people', 'may help', 'could be', 'might work',
                'anecdotal', 'personal experience', 'unverified'
            ]
        }
    
    def process_comprehensive_content(self, sources: List[Dict], user_query: str) -> Tuple[str, Dict[int, str]]:
        """Process all sources to extract maximum relevant information"""
        if not sources:
            return "", {}
        
        logger.info(f"Processing {len(sources)} sources for comprehensive information extraction")
        
        # Extract structured information from each source
        structured_info = self._extract_structured_information(sources, user_query)
        
        # Create comprehensive summary
        comprehensive_summary = self._create_comprehensive_summary(structured_info, user_query)
        
        # Create detailed reference mapping
        reference_mapping = self._create_detailed_reference_mapping(sources)
        
        return comprehensive_summary, reference_mapping
    
    def _extract_structured_information(self, sources: List[Dict], user_query: str) -> Dict[str, List[Dict]]:
        """Extract structured information by medical categories"""
        structured_info = defaultdict(list)
        
        for source in sources:
            content = source.get('content', '')
            if not content:
                continue
            
            # Extract information by medical categories
            for category, patterns in self.medical_patterns.items():
                extracted_info = self._extract_category_info(content, patterns, category, user_query)
                if extracted_info:
                    structured_info[category].append({
                        'content': extracted_info,
                        'source': source,
                        'relevance_score': self._calculate_relevance_score(extracted_info, user_query)
                    })
        
        # Sort by relevance within each category
        for category in structured_info:
            structured_info[category].sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return dict(structured_info)
    
    def _extract_category_info(self, content: str, patterns: List[str], category: str, user_query: str) -> str:
        """Extract information for a specific medical category"""
        extracted_sentences = []
        
        # Split content into sentences
        sentences = re.split(r'[.!?]+', content)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:  # Skip very short sentences
                continue
            
            # Check if sentence matches any pattern for this category
            for pattern in patterns:
                if re.search(pattern, sentence, re.IGNORECASE):
                    # Check relevance to user query
                    if self._is_relevant_to_query(sentence, user_query):
                        extracted_sentences.append(sentence)
                        break
        
        # Combine and summarize extracted sentences
        if extracted_sentences:
            combined_text = '. '.join(extracted_sentences[:5])  # Limit to top 5 sentences
            return summarizer.summarize_for_query(combined_text, user_query, max_length=300)
        
        return ""
    
    def _is_relevant_to_query(self, sentence: str, user_query: str) -> bool:
        """Check if sentence is relevant to user query"""
        query_words = set(user_query.lower().split())
        sentence_words = set(sentence.lower().split())
        
        # Calculate word overlap
        overlap = len(query_words.intersection(sentence_words))
        return overlap >= 2  # At least 2 words in common
    
    def _calculate_relevance_score(self, content: str, user_query: str) -> float:
        """Calculate relevance score for content"""
        if not content or not user_query:
            return 0.0
        
        query_words = set(user_query.lower().split())
        content_words = set(content.lower().split())
        
        # Word overlap score
        overlap = len(query_words.intersection(content_words))
        overlap_score = overlap / len(query_words) if query_words else 0
        
        # Content quality score
        quality_score = self._assess_content_quality(content)
        
        # Length score (prefer medium-length content)
        length_score = min(len(content) / 500, 1.0)  # Normalize to 0-1
        
        # Composite score
        composite_score = (
            overlap_score * 0.5 +    # 50% relevance to query
            quality_score * 0.3 +    # 30% content quality
            length_score * 0.2       # 20% appropriate length
        )
        
        return min(composite_score, 1.0)
    
    def _assess_content_quality(self, content: str) -> float:
        """Assess content quality based on medical indicators"""
        content_lower = content.lower()
        
        high_indicators = sum(1 for indicator in self.quality_indicators['high'] if indicator in content_lower)
        medium_indicators = sum(1 for indicator in self.quality_indicators['medium'] if indicator in content_lower)
        low_indicators = sum(1 for indicator in self.quality_indicators['low'] if indicator in content_lower)
        
        # Calculate quality score
        if high_indicators > 0:
            return 0.9
        elif medium_indicators > 0:
            return 0.7
        elif low_indicators > 0:
            return 0.5
        else:
            return 0.6  # Default score for neutral content
    
    def _create_comprehensive_summary(self, structured_info: Dict[str, List[Dict]], user_query: str) -> str:
        """Create comprehensive summary from structured information"""
        if not structured_info:
            return ""
        
        summary_parts = []
        
        # Process each category
        category_headers = {
            'symptoms': "**🔍 Symptoms & Signs:**",
            'causes': "**🔬 Causes & Risk Factors:**",
            'treatments': "**💊 Treatment Options:**",
            'diagnosis': "**🔬 Diagnosis & Testing:**",
            'prevention': "**🛡️ Prevention & Precautions:**",
            'prognosis': "**📈 Prognosis & Outlook:**"
        }
        
        for category, info_list in structured_info.items():
            if not info_list:
                continue
            
            # Take top 2 most relevant items for each category
            top_items = info_list[:2]
            
            category_content = []
            for item in top_items:
                content = item['content']
                if content:
                    category_content.append(content)
            
            if category_content:
                # Combine and summarize category content
                combined_content = ' '.join(category_content)
                category_summary = summarizer.summarize_for_query(combined_content, user_query, max_length=400)
                
                if category_summary:
                    header = category_headers.get(category, f"**{category.title()}:**")
                    summary_parts.append(f"{header}\n{category_summary}")
        
        # Combine all parts
        comprehensive_summary = "\n\n".join(summary_parts)
        
        # Final summarization to ensure conciseness
        if len(comprehensive_summary) > 2000:
            comprehensive_summary = summarizer.summarize_text(comprehensive_summary, max_length=2000)
        
        return comprehensive_summary
    
    def _create_detailed_reference_mapping(self, sources: List[Dict]) -> Dict[int, Dict]:
        """Create detailed reference mapping with source metadata"""
        reference_mapping = {}
        
        for i, source in enumerate(sources, 1):
            # Be defensive: some upstream sources may miss optional fields
            reference_mapping[i] = {
                'url': source.get('url', ''),
                'title': source.get('title', ''),
                'domain': source.get('domain', ''),
                'source_type': source.get('source_type', 'text'),
                'language': source.get('language', 'en'),
                'type': source.get('type', 'text'),
                'content_length': len(source.get('content', '')),
                'composite_score': source.get('composite_score', 0.7)
            }
        
        return reference_mapping
    
    def create_inline_citations(self, text: str, reference_mapping: Dict[int, Dict]) -> str:
        """Create inline citations within the text"""
        if not reference_mapping:
            return text
        
        # Find places where citations should be added
        # This is a simplified version - in practice, you'd use more sophisticated NLP
        
        # Add citations after key medical statements
        citation_patterns = [
            r'(symptoms?\s+(?:include|are)[^.]*\.)',
            r'(treatment\s+(?:options?|include|are)[^.]*\.)',
            r'(diagnosis\s+(?:include|are)[^.]*\.)',
            r'(causes?\s+(?:include|are)[^.]*\.)',
            r'(studies?\s+show[^.]*\.)',
            r'(research\s+(?:indicates|shows)[^.]*\.)'
        ]
        
        cited_text = text
        citation_count = 1
        
        for pattern in citation_patterns:
            matches = re.finditer(pattern, cited_text, re.IGNORECASE)
            for match in matches:
                if citation_count <= len(reference_mapping):
                    citation_tag = f" <#{citation_count}>"
                    cited_text = cited_text.replace(match.group(1), match.group(1) + citation_tag, 1)
                    citation_count += 1
        
        return cited_text
    
    def generate_source_statistics(self, sources: List[Dict]) -> str:
        """Generate statistics about sources used"""
        if not sources:
            return ""
        
        total_sources = len(sources)
        # credibility removed
        
        # Language distribution
        languages = defaultdict(int)
        for source in sources:
            lang = source.get('language', 'en')
            languages[lang] += 1
        
        # Source type distribution
        source_types = defaultdict(int)
        for source in sources:
            source_type = source.get('source_type', 'other')
            source_types[source_type] += 1
        
        # Content length statistics
        content_lengths = [len(s.get('content', '')) for s in sources]
        avg_content_length = sum(content_lengths) / len(content_lengths) if content_lengths else 0
        
        stats_parts = []
        stats_parts.append(f"**📊 Source Statistics:**")
        stats_parts.append(f"• **Total Sources**: {total_sources}")
        # removed credibility summary
        stats_parts.append(f"• **Languages**: {', '.join([f'{count} {lang}' for lang, count in languages.items()])}")
        stats_parts.append(f"• **Types**: {', '.join([f'{count} {type_name}' for type_name, count in source_types.items()])}")
        stats_parts.append(f"• **Avg Content Length**: {avg_content_length:.0f} characters")
        
        return "\n".join(stats_parts)

