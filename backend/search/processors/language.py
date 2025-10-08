import re
import logging
from typing import List, Dict, Tuple, Optional
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

logger = logging.getLogger(__name__)

# Set seed for consistent language detection
DetectorFactory.seed = 0

class LanguageProcessor:
    """Process and enhance queries for multilingual medical search"""
    
    def __init__(self):
        # Medical keywords in different languages
        self.medical_keywords = {
            'en': [
                'symptom', 'symptoms', 'pain', 'headache', 'migraine', 'fever', 'cough',
                'treatment', 'treatments', 'medicine', 'medication', 'drug', 'therapy',
                'diagnosis', 'diagnose', 'condition', 'disease', 'disorder', 'syndrome',
                'doctor', 'physician', 'medical', 'health', 'clinical', 'patient',
                'blood pressure', 'heart', 'lung', 'stomach', 'back', 'neck', 'chest',
                'allergy', 'allergies', 'infection', 'inflammation', 'swelling', 'rash',
                'sleep', 'insomnia', 'anxiety', 'depression', 'stress', 'mental health',
                'pregnancy', 'baby', 'child', 'elderly', 'senior', 'age', 'covid',
                'vaccine', 'immunization', 'surgery', 'operation', 'hospital', 'clinic'
            ],
            'vi': [
                'triệu chứng', 'đau', 'đau đầu', 'đau nửa đầu', 'sốt', 'ho',
                'điều trị', 'thuốc', 'dược phẩm', 'liệu pháp', 'chẩn đoán',
                'bệnh', 'tình trạng', 'rối loạn', 'hội chứng', 'bác sĩ', 'y tế',
                'sức khỏe', 'lâm sàng', 'bệnh nhân', 'huyết áp', 'tim', 'phổi',
                'dạ dày', 'lưng', 'cổ', 'ngực', 'dị ứng', 'nhiễm trùng',
                'viêm', 'sưng', 'phát ban', 'ngủ', 'mất ngủ', 'lo âu',
                'trầm cảm', 'căng thẳng', 'sức khỏe tâm thần', 'mang thai',
                'em bé', 'trẻ em', 'người già', 'tuổi tác', 'covid', 'vaccine',
                'tiêm chủng', 'phẫu thuật', 'bệnh viện', 'phòng khám'
            ],
            'zh': [
                '症状', '疼痛', '头痛', '偏头痛', '发烧', '咳嗽', '治疗', '药物',
                '药品', '疗法', '诊断', '疾病', '状况', '紊乱', '综合征', '医生',
                '医疗', '健康', '临床', '患者', '血压', '心脏', '肺', '胃',
                '背部', '颈部', '胸部', '过敏', '感染', '炎症', '肿胀', '皮疹',
                '睡眠', '失眠', '焦虑', '抑郁', '压力', '心理健康', '怀孕',
                '婴儿', '儿童', '老年人', '年龄', '新冠', '疫苗', '免疫',
                '手术', '医院', '诊所'
            ]
        }
        
        # Language-specific search enhancements
        self.language_enhancements = {
            'vi': {
                'common_terms': ['là gì', 'nguyên nhân', 'cách điều trị', 'triệu chứng'],
                'medical_context': ['y tế', 'sức khỏe', 'bệnh viện', 'bác sĩ']
            },
            'zh': {
                'common_terms': ['是什么', '原因', '治疗方法', '症状'],
                'medical_context': ['医疗', '健康', '医院', '医生']
            },
            'en': {
                'common_terms': ['what is', 'causes', 'treatment', 'symptoms'],
                'medical_context': ['medical', 'health', 'hospital', 'doctor']
            }
        }
    
    def detect_language(self, text: str) -> str:
        """Detect the language of the input text"""
        if not text or not text.strip():
            return 'en'  # Default to English
        
        try:
            # Clean text for better detection
            cleaned_text = re.sub(r'[^\w\s]', ' ', text)
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
            
            if len(cleaned_text) < 3:
                return 'en'
            
            detected = detect(cleaned_text)
            
            # Map detected language to our supported languages
            language_mapping = {
                'vi': 'vi',  # Vietnamese
                'zh-cn': 'zh',  # Chinese Simplified
                'zh-tw': 'zh',  # Chinese Traditional
                'zh': 'zh',     # Chinese
                'en': 'en'      # English
            }
            
            return language_mapping.get(detected, 'en')
            
        except LangDetectException as e:
            logger.warning(f"Language detection failed: {e}")
            return 'en'
    
    def enhance_query(self, query: str, target_language: str = None) -> Dict[str, str]:
        """Enhance query for better search results in multiple languages"""
        if not query or not query.strip():
            return {}
        
        # Detect source language
        source_language = self.detect_language(query)
        
        # If target language not specified, use source language
        if target_language is None:
            target_language = source_language
        
        enhanced_queries = {}
        
        # Original query
        enhanced_queries[source_language] = query
        
        # Enhance for source language
        if source_language in self.language_enhancements:
            enhanced_queries[source_language] = self._enhance_for_language(
                query, source_language
            )
        
        # Create translations for other languages if needed
        if target_language != source_language:
            enhanced_queries[target_language] = self._translate_query(
                query, source_language, target_language
            )
        
        # Add English version for comprehensive search
        if 'en' not in enhanced_queries:
            if source_language != 'en':
                enhanced_queries['en'] = self._translate_query(query, source_language, 'en')
            else:
                enhanced_queries['en'] = query
        
        return enhanced_queries
    
    def _enhance_for_language(self, query: str, language: str) -> str:
        """Enhance query for a specific language"""
        enhancements = self.language_enhancements.get(language, {})
        common_terms = enhancements.get('common_terms', [])
        medical_context = enhancements.get('medical_context', [])
        
        # Check if query already contains medical context
        query_lower = query.lower()
        has_medical_context = any(term in query_lower for term in medical_context)
        
        # If no medical context, add it
        if not has_medical_context and medical_context:
            # Add the most relevant medical context term
            query += f" {medical_context[0]}"
        
        # Check if query is a question and add relevant terms
        if any(term in query_lower for term in ['là gì', '是什么', 'what is', 'how', 'tại sao', '为什么', 'why']):
            if common_terms:
                query += f" {common_terms[0]}"  # Add "causes" or equivalent
        
        return query.strip()
    
    def _translate_query(self, query: str, source_lang: str, target_lang: str) -> str:
        """Simple keyword-based translation for medical terms"""
        # This is a basic implementation - in production, you'd use a proper translation service
        
        # Medical term translations
        translations = {
            ('vi', 'en'): {
                'triệu chứng': 'symptoms',
                'đau': 'pain',
                'đau đầu': 'headache',
                'sốt': 'fever',
                'ho': 'cough',
                'điều trị': 'treatment',
                'thuốc': 'medicine',
                'bệnh': 'disease',
                'bác sĩ': 'doctor',
                'sức khỏe': 'health',
                'bệnh viện': 'hospital'
            },
            ('zh', 'en'): {
                '症状': 'symptoms',
                '疼痛': 'pain',
                '头痛': 'headache',
                '发烧': 'fever',
                '咳嗽': 'cough',
                '治疗': 'treatment',
                '药物': 'medicine',
                '疾病': 'disease',
                '医生': 'doctor',
                '健康': 'health',
                '医院': 'hospital'
            },
            ('en', 'vi'): {
                'symptoms': 'triệu chứng',
                'pain': 'đau',
                'headache': 'đau đầu',
                'fever': 'sốt',
                'cough': 'ho',
                'treatment': 'điều trị',
                'medicine': 'thuốc',
                'disease': 'bệnh',
                'doctor': 'bác sĩ',
                'health': 'sức khỏe',
                'hospital': 'bệnh viện'
            },
            ('en', 'zh'): {
                'symptoms': '症状',
                'pain': '疼痛',
                'headache': '头痛',
                'fever': '发烧',
                'cough': '咳嗽',
                'treatment': '治疗',
                'medicine': '药物',
                'disease': '疾病',
                'doctor': '医生',
                'health': '健康',
                'hospital': '医院'
            }
        }
        
        translation_map = translations.get((source_lang, target_lang), {})
        
        # Simple word-by-word translation
        translated_query = query
        for source_term, target_term in translation_map.items():
            translated_query = translated_query.replace(source_term, target_term)
        
        return translated_query
    
    def get_medical_relevance_score(self, text: str, language: str) -> float:
        """Calculate medical relevance score for text in a specific language"""
        if not text:
            return 0.0
        
        keywords = self.medical_keywords.get(language, [])
        if not keywords:
            return 0.0
        
        text_lower = text.lower()
        matches = sum(1 for keyword in keywords if keyword in text_lower)
        
        # Normalize by text length and keyword count
        score = matches / max(len(keywords), 1)
        
        # Boost score for longer matches
        if matches > 0:
            score *= (1 + matches * 0.1)
        
        return min(score, 1.0)
    
    def filter_by_language(self, results: List[Dict], target_language: str) -> List[Dict]:
        """Filter results by language preference"""
        if not results:
            return results
        
        # Score results by language match
        scored_results = []
        for result in results:
            result_language = result.get('language', 'en')
            language_score = 1.0 if result_language == target_language else 0.5
            
            # Add language score to result
            result_copy = result.copy()
            result_copy['language_score'] = language_score
            scored_results.append(result_copy)
        
        # Sort by language score (prefer target language)
        scored_results.sort(key=lambda x: x.get('language_score', 0), reverse=True)
        
        return scored_results
