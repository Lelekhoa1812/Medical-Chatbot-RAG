import logging
from typing import List, Dict, Tuple
from models.summarizer import summarizer
import re

logger = logging.getLogger(__name__)

class MedicalSearchProcessor:
    """Process and enhance medical search results"""
    
    def __init__(self):
        self.medical_keywords = [
            'symptom', 'symptoms', 'pain', 'headache', 'migraine', 'fever', 'cough',
            'treatment', 'treatments', 'medicine', 'medication', 'drug', 'therapy',
            'diagnosis', 'diagnose', 'condition', 'disease', 'disorder', 'syndrome',
            'doctor', 'physician', 'medical', 'health', 'clinical', 'patient',
            'blood pressure', 'heart', 'lung', 'stomach', 'back', 'neck', 'chest',
            'allergy', 'allergies', 'infection', 'inflammation', 'swelling', 'rash',
            'sleep', 'insomnia', 'anxiety', 'depression', 'stress', 'mental health',
            'pregnancy', 'baby', 'child', 'elderly', 'senior', 'age', 'covid',
            'vaccine', 'immunization', 'surgery', 'operation', 'hospital', 'clinic'
        ]
    
    def process_results(self, results: List[Dict], user_query: str) -> Tuple[str, Dict[int, str]]:
        """Process search results and create comprehensive medical summary"""
        if not results:
            return "", {}
        
        # Filter and rank results by medical relevance
        relevant_results = self._filter_medical_results(results, user_query)
        
        if not relevant_results:
            logger.warning("No medically relevant results found")
            return "", {}
        
        # Extract and summarize content
        summarized_results = self._summarize_results(relevant_results, user_query)
        
        # Create comprehensive summary
        combined_summary = self._create_combined_summary(summarized_results, user_query)
        
        # Create URL mapping for citations
        url_mapping = self._create_url_mapping(relevant_results)
        
        return combined_summary, url_mapping
    
    def _filter_medical_results(self, results: List[Dict], user_query: str) -> List[Dict]:
        """Filter results by medical relevance"""
        relevant_results = []
        
        for result in results:
            relevance_score = self._calculate_relevance_score(result, user_query)
            
            if relevance_score > 0.3:  # Threshold for medical relevance
                result['relevance_score'] = relevance_score
                relevant_results.append(result)
        
        # Sort by relevance score
        relevant_results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        # Limit to top results
        return relevant_results[:10]
    
    def _calculate_relevance_score(self, result: Dict, user_query: str) -> float:
        """Calculate medical relevance score for a result"""
        score = 0.0
        
        # Check title relevance
        title = result.get('title', '').lower()
        query_lower = user_query.lower()
        
        # Direct query match in title
        if any(word in title for word in query_lower.split()):
            score += 0.4
        
        # Medical keyword match in title
        medical_matches = sum(1 for keyword in self.medical_keywords if keyword in title)
        score += min(medical_matches * 0.1, 0.3)
        
        # Domain credibility
        url = result.get('url', '').lower()
        credible_domains = [
            'mayoclinic.org', 'webmd.com', 'healthline.com', 'medlineplus.gov',
            'nih.gov', 'cdc.gov', 'who.int', 'pubmed.ncbi.nlm.nih.gov',
            'uptodate.com', 'merckmanuals.com', 'medscape.com'
        ]
        
        if any(domain in url for domain in credible_domains):
            score += 0.3
        
        # Source type bonus
        source = result.get('source', '')
        if 'medical' in source or any(domain in source for domain in credible_domains):
            score += 0.2
        
        return min(score, 1.0)
    
    def _summarize_results(self, results: List[Dict], user_query: str) -> List[Dict]:
        """Summarize content from search results"""
        summarized_results = []
        
        for i, result in enumerate(results):
            try:
                content = result.get('content', '')
                if not content:
                    continue
                
                # Create focused summary
                summary = summarizer.summarize_for_query(content, user_query, max_length=300)
                
                if summary:
                    summarized_results.append({
                        'id': i + 1,
                        'url': result['url'],
                        'title': result['title'],
                        'summary': summary,
                        'relevance_score': result.get('relevance_score', 0)
                    })
                    
            except Exception as e:
                logger.warning(f"Failed to summarize result {i}: {e}")
                continue
        
        return summarized_results
    
    def _create_combined_summary(self, summarized_results: List[Dict], user_query: str) -> str:
        """Create a comprehensive summary from all results"""
        if not summarized_results:
            return ""
        
        # Group by topic/similarity
        topic_groups = self._group_by_topic(summarized_results)
        
        summary_parts = []
        
        for topic, results in topic_groups.items():
            if not results:
                continue
            
            # Create topic summary
            topic_summary = self._create_topic_summary(topic, results, user_query)
            if topic_summary:
                summary_parts.append(topic_summary)
        
        # Combine all parts
        combined_summary = "\n\n".join(summary_parts)
        
        # Final summarization to ensure conciseness
        if len(combined_summary) > 1500:
            combined_summary = summarizer.summarize_text(combined_summary, max_length=1500)
        
        return combined_summary
    
    def _group_by_topic(self, results: List[Dict]) -> Dict[str, List[Dict]]:
        """Group results by medical topic"""
        topics = {
            'symptoms': [],
            'treatments': [],
            'diagnosis': [],
            'general': []
        }
        
        for result in results:
            title_lower = result['title'].lower()
            summary_lower = result.get('summary', '').lower()
            content_lower = f"{title_lower} {summary_lower}"
            
            # Categorize by content
            if any(word in content_lower for word in ['symptom', 'sign', 'pain', 'ache']):
                topics['symptoms'].append(result)
            elif any(word in content_lower for word in ['treatment', 'therapy', 'medicine', 'medication']):
                topics['treatments'].append(result)
            elif any(word in content_lower for word in ['diagnosis', 'test', 'examination', 'evaluation']):
                topics['diagnosis'].append(result)
            else:
                topics['general'].append(result)
        
        return topics
    
    def _create_topic_summary(self, topic: str, results: List[Dict], user_query: str) -> str:
        """Create summary for a specific topic"""
        if not results:
            return ""
        
        # Combine summaries for this topic
        combined_text = " ".join([r.get('summary', '') for r in results])
        
        if not combined_text:
            return ""
        
        # Create focused summary for this topic
        topic_summary = summarizer.summarize_for_query(combined_text, user_query, max_length=400)
        
        if topic_summary:
            # Add topic header
            topic_headers = {
                'symptoms': "**Symptoms and Signs:**",
                'treatments': "**Treatment Options:**",
                'diagnosis': "**Diagnosis and Testing:**",
                'general': "**General Information:**"
            }
            
            header = topic_headers.get(topic, "**Information:**")
            return f"{header}\n{topic_summary}"
        
        return ""
    
    def _create_url_mapping(self, results: List[Dict]) -> Dict[int, str]:
        """Create URL mapping for citations"""
        url_mapping = {}
        
        for i, result in enumerate(results):
            url_mapping[i + 1] = result['url']
        
        return url_mapping
