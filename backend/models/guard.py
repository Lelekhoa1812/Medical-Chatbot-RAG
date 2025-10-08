import os
import requests
import logging
from typing import Tuple, List, Dict


logger = logging.getLogger(__name__)


class SafetyGuard:
    """
    Wrapper around NVIDIA Llama Guard (meta/llama-guard-4-12b) hosted at
    https://integrate.api.nvidia.com/v1/chat/completions

    Exposes helpers to validate:
      - user input safety
      - model output safety (in context of the user question)
    """

    def __init__(self):
        self.api_key = os.getenv("NVIDIA_URI")
        if not self.api_key:
            raise ValueError("NVIDIA_URI environment variable not set for SafetyGuard")
        self.base_url = "https://integrate.api.nvidia.com/v1/chat/completions"
        self.model = "meta/llama-guard-4-12b"
        self.timeout_s = 30

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 2800, overlap: int = 200) -> List[str]:
        """Chunk long text to keep request payloads small enough for the guard.
        Uses character-based approximation with small overlap.
        """
        if not text:
            return [""]
        n = len(text)
        if n <= chunk_size:
            return [text]
        chunks: List[str] = []
        start = 0
        while start < n:
            end = min(start + chunk_size, n)
            chunks.append(text[start:end])
            if end == n:
                break
            start = max(0, end - overlap)
        return chunks

    def _call_guard(self, messages: List[Dict], max_tokens: int = 512) -> str:
        # Enhance messages with medical context if detected
        enhanced_messages = self._enhance_messages_with_context(messages)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        # Try OpenAI-compatible schema first
        payload_chat = {
            "model": self.model,
            "messages": enhanced_messages,
            "temperature": 0.2,
            "top_p": 0.7,
            "max_tokens": max_tokens,
            "stream": False,
        }
        # Alternative schema (some NVIDIA deployments require message content objects)
        alt_messages = []
        for m in enhanced_messages:
            content = m.get("content", "")
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            alt_messages.append({"role": m.get("role", "user"), "content": content})
        payload_alt = {
            "model": self.model,
            "messages": alt_messages,
            "temperature": 0.2,
            "top_p": 0.7,
            "max_tokens": max_tokens,
            "stream": False,
        }
        # Attempt primary, then fallback
        for payload in (payload_chat, payload_alt):
            try:
                resp = requests.post(self.base_url, headers=headers, json=payload, timeout=self.timeout_s)
                if resp.status_code >= 400:
                    # Log server message for debugging payload issues
                    try:
                        logger.error(f"[SafetyGuard] HTTP {resp.status_code}: {resp.text[:400]}")
                    except Exception:
                        pass
                    resp.raise_for_status()
                data = resp.json()
                content = (
                    data.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                        .strip()
                )
                if content:
                    return content
            except Exception as e:
                # Try next payload shape
                logger.error(f"[SafetyGuard] Guard API call failed: {e}")
                continue
        # All attempts failed
        return ""

    @staticmethod
    def _parse_guard_reply(text: str) -> Tuple[bool, str]:
        """Parse guard reply; expect 'SAFE' or 'UNSAFE: <reason>' (case-insensitive)."""
        if not text:
            # Fail-open: treat as SAFE if guard unavailable to avoid false blocks
            return True, "guard_unavailable"
        t = text.strip()
        upper = t.upper()
        if upper.startswith("SAFE") and not upper.startswith("SAFEGUARD"):
            return True, ""
        if upper.startswith("UNSAFE"):
            # Extract reason after the first colon if present
            parts = t.split(":", 1)
            reason = parts[1].strip() if len(parts) > 1 else "policy violation"
            return False, reason
        # Fallback: treat unknown response as unsafe
        return False, t[:180]

    def _is_medical_query(self, query: str) -> bool:
        """Check if query is clearly medical in nature using comprehensive patterns."""
        if not query:
            return False
        
        query_lower = query.lower()
        
        # Medical keyword categories
        medical_categories = {
            'symptoms': [
                'symptom', 'pain', 'ache', 'hurt', 'sore', 'tender', 'stiff', 'numb',
                'headache', 'migraine', 'fever', 'cough', 'cold', 'flu', 'sneeze',
                'nausea', 'vomit', 'diarrhea', 'constipation', 'bloating', 'gas',
                'dizziness', 'vertigo', 'fatigue', 'weakness', 'tired', 'exhausted',
                'shortness of breath', 'wheezing', 'chest pain', 'heart palpitations',
                'joint pain', 'muscle pain', 'back pain', 'neck pain', 'stomach pain',
                'abdominal pain', 'pelvic pain', 'menstrual pain', 'cramps'
            ],
            'conditions': [
                'disease', 'condition', 'disorder', 'syndrome', 'illness', 'sickness',
                'infection', 'inflammation', 'allergy', 'asthma', 'diabetes', 'hypertension',
                'depression', 'anxiety', 'stress', 'panic', 'phobia', 'ocd', 'ptsd',
                'adhd', 'autism', 'dementia', 'alzheimer', 'parkinson', 'epilepsy',
                'cancer', 'tumor', 'cancerous', 'malignant', 'benign', 'metastasis',
                'heart disease', 'stroke', 'heart attack', 'coronary', 'arrhythmia',
                'pneumonia', 'bronchitis', 'copd', 'emphysema', 'tuberculosis'
            ],
            'treatments': [
                'treatment', 'therapy', 'medication', 'medicine', 'drug', 'pill', 'tablet',
                'injection', 'vaccine', 'immunization', 'surgery', 'operation', 'procedure',
                'chemotherapy', 'radiation', 'physical therapy', 'occupational therapy',
                'psychotherapy', 'counseling', 'rehabilitation', 'recovery', 'healing',
                'prescription', 'dosage', 'side effects', 'contraindications'
            ],
            'body_parts': [
                'head', 'brain', 'eye', 'ear', 'nose', 'mouth', 'throat', 'neck',
                'chest', 'heart', 'lung', 'liver', 'kidney', 'stomach', 'intestine',
                'back', 'spine', 'joint', 'muscle', 'bone', 'skin', 'hair', 'nail',
                'arm', 'leg', 'hand', 'foot', 'finger', 'toe', 'pelvis', 'genital'
            ],
            'medical_context': [
                'doctor', 'physician', 'nurse', 'specialist', 'surgeon', 'dentist',
                'medical', 'health', 'healthcare', 'hospital', 'clinic', 'emergency',
                'ambulance', 'paramedic', 'pharmacy', 'pharmacist', 'lab', 'test',
                'diagnosis', 'prognosis', 'examination', 'checkup', 'screening',
                'patient', 'case', 'history', 'medical history', 'family history'
            ],
            'life_stages': [
                'pregnancy', 'pregnant', 'baby', 'infant', 'newborn', 'child', 'pediatric',
                'teenager', 'adolescent', 'adult', 'elderly', 'senior', 'geriatric',
                'menopause', 'puberty', 'aging', 'birth', 'delivery', 'miscarriage'
            ],
            'vital_signs': [
                'blood pressure', 'heart rate', 'pulse', 'temperature', 'fever',
                'respiratory rate', 'oxygen saturation', 'weight', 'height', 'bmi',
                'blood sugar', 'glucose', 'cholesterol', 'hemoglobin', 'white blood cell'
            ]
        }
        
        # Check for medical keywords
        for category, keywords in medical_categories.items():
            if any(keyword in query_lower for keyword in keywords):
                return True
        
        # Check for medical question patterns
        medical_patterns = [
            r'\b(what|how|why|when|where)\s+(causes?|treats?|prevents?|symptoms?|signs?)\b',
            r'\b(is|are)\s+(.*?)\s+(dangerous|serious|harmful|safe|normal)\b',
            r'\b(should|can|may|might)\s+(i|you|we)\s+(take|use|do|avoid)\b',
            r'\b(diagnosis|diagnosed|symptoms|treatment|medicine|drug)\b',
            r'\b(medical|health|doctor|physician|hospital|clinic)\b',
            r'\b(pain|hurt|ache|sore|fever|cough|headache)\b'
        ]
        
        import re
        for pattern in medical_patterns:
            if re.search(pattern, query_lower):
                return True
        
        return False

    def check_user_query(self, user_query: str) -> Tuple[bool, str]:
        """Validate the user query is safe to process with medical context awareness."""
        text = user_query or ""
        
        # For medical queries, be more permissive
        if self._is_medical_query(text):
            logger.info("[SafetyGuard] Medical query detected, skipping strict validation")
            return True, "medical_query"
        
        # If too long, validate each chunk; any UNSAFE makes overall UNSAFE
        for part in self._chunk_text(text):
            messages = [{"role": "user", "content": part}]
            reply = self._call_guard(messages, max_tokens=64)
            ok, reason = self._parse_guard_reply(reply)
            if not ok:
                return False, reason
        return True, ""

    def _detect_harmful_content(self, text: str) -> Tuple[bool, str]:
        """Detect harmful content using sophisticated pattern matching."""
        if not text:
            return True, ""
        
        text_lower = text.lower()
        
        # Critical harmful patterns (immediate block)
        critical_patterns = {
            'suicide_self_harm': [
                r'\b(kill\s+yourself|suicide|end\s+your\s+life|take\s+your\s+life)\b',
                r'\b(self\s*harm|self\s*injury|cut\s+yourself|hurt\s+yourself)\b',
                r'\b(overdose|poison\s+yourself|hang\s+yourself)\b'
            ],
            'violence': [
                r'\b(kill\s+someone|murder|assassinate|violence|harm\s+others)\b',
                r'\b(bomb|explosive|weapon|gun|knife|attack)\b'
            ],
            'illegal_drugs': [
                r'\b(cocaine|heroin|methamphetamine|ecstasy|lsd|marijuana\s+abuse)\b',
                r'\b(prescription\s+abuse|drug\s+dealing|illegal\s+substances)\b'
            ],
            'dangerous_medical': [
                r'\b(overdose\s+on|take\s+too\s+much|excessive\s+dosage)\b',
                r'\b(mix\s+drugs|combine\s+medications|contraindicated)\b',
                r'\b(stop\s+taking\s+prescribed|discontinue\s+medication)\b'
            ]
        }
        
        # Check critical patterns
        for category, patterns in critical_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return False, f"contains {category}: {pattern}"
        
        # Context-aware medical safety checks
        medical_safety_patterns = {
            'dosage_warnings': [
                r'\b(take\s+more\s+than\s+prescribed|exceed\s+recommended\s+dose)\b',
                r'\b(double\s+up\s+on\s+medication|take\s+extra\s+pills)\b'
            ],
            'emergency_advice': [
                r'\b(don\'t\s+call\s+911|avoid\s+emergency\s+room|ignore\s+severe\s+symptoms)\b',
                r'\b(self\s*treat\s+emergency|handle\s+at\s+home\s+when\s+critical)\b'
            ]
        }
        
        # Check medical safety patterns
        for category, patterns in medical_safety_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return False, f"contains dangerous medical advice: {pattern}"
        
        return True, ""

    def _enhance_messages_with_context(self, messages: List[Dict]) -> List[Dict]:
        """Enhance messages with medical context awareness for better guard performance."""
        if not messages:
            return messages
        
        # Check if any message contains medical content
        has_medical_content = any(
            self._is_medical_query(msg.get("content", "")) for msg in messages
        )
        
        if has_medical_content:
            # Add medical context to the system prompt
            enhanced_messages = []
            for i, msg in enumerate(messages):
                if i == 0 and msg.get("role") == "user":
                    # Add medical context to first user message
                    original_content = msg.get("content", "")
                    enhanced_content = f"""MEDICAL CONTEXT: This is a medical consultation. The user is asking about health-related topics.

ORIGINAL MESSAGE: {original_content}

Please evaluate this message for safety, keeping in mind that:
- Medical questions and health concerns are legitimate
- General medical information sharing is acceptable
- Only block content that is clearly harmful, dangerous, or illegal
- Medical advice should be educational, not prescriptive"""
                    
                    enhanced_messages.append({
                        "role": msg["role"],
                        "content": enhanced_content
                    })
                else:
                    enhanced_messages.append(msg)
            return enhanced_messages
        
        return messages

    def _assess_risk_level(self, text: str) -> Tuple[str, float]:
        """Assess the risk level of content using multiple indicators."""
        if not text:
            return "low", 0.0
        
        text_lower = text.lower()
        risk_indicators = {
            'high': [
                'suicide', 'kill yourself', 'end your life', 'self harm',
                'overdose', 'poison', 'illegal drugs', 'violence', 'harm others'
            ],
            'medium': [
                'prescription abuse', 'excessive dosage', 'mix drugs',
                'stop medication', 'ignore symptoms', 'avoid doctor'
            ],
            'low': [
                'pain', 'headache', 'fever', 'cough', 'treatment',
                'medicine', 'doctor', 'hospital', 'symptoms'
            ]
        }
        
        risk_score = 0.0
        for level, indicators in risk_indicators.items():
            for indicator in indicators:
                if indicator in text_lower:
                    if level == 'high':
                        risk_score += 3.0
                    elif level == 'medium':
                        risk_score += 1.5
                    else:
                        risk_score += 0.5
        
        # Normalize score
        risk_score = min(risk_score / 10.0, 1.0)
        
        if risk_score >= 0.7:
            return "high", risk_score
        elif risk_score >= 0.3:
            return "medium", risk_score
        else:
            return "low", risk_score

    def check_model_answer(self, user_query: str, model_answer: str) -> Tuple[bool, str]:
        """Validate the model's answer is safe with medical context awareness."""
        uq = user_query or ""
        ans = model_answer or ""
        
        # Assess risk level first
        risk_level, risk_score = self._assess_risk_level(ans)
        logger.info(f"[SafetyGuard] Risk assessment: {risk_level} (score: {risk_score:.2f})")
        
        # Always check for harmful content first
        is_safe, reason = self._detect_harmful_content(ans)
        if not is_safe:
            return False, reason
        
        # For high-risk content, always use strict validation
        if risk_level == "high":
            logger.warning("[SafetyGuard] High-risk content detected, using strict validation")
            user_parts = self._chunk_text(uq, chunk_size=2000)
            user_context = user_parts[0] if user_parts else ""
            for ans_part in self._chunk_text(ans):
                messages = [
                    {"role": "user", "content": user_context},
                    {"role": "assistant", "content": ans_part},
                ]
                reply = self._call_guard(messages, max_tokens=96)
                ok, reason = self._parse_guard_reply(reply)
                if not ok:
                    return False, reason
            return True, "high_risk_validated"
        
        # For medical queries and answers, use relaxed validation
        if self._is_medical_query(uq) or self._is_medical_query(ans):
            logger.info("[SafetyGuard] Medical content detected, using relaxed validation")
            return True, "medical_content"
        
        # For medium-risk non-medical content, use guard validation
        if risk_level == "medium":
            logger.info("[SafetyGuard] Medium-risk content detected, using guard validation")
            user_parts = self._chunk_text(uq, chunk_size=2000)
            user_context = user_parts[0] if user_parts else ""
            for ans_part in self._chunk_text(ans):
                messages = [
                    {"role": "user", "content": user_context},
                    {"role": "assistant", "content": ans_part},
                ]
                reply = self._call_guard(messages, max_tokens=96)
                ok, reason = self._parse_guard_reply(reply)
                if not ok:
                    return False, reason
            return True, "medium_risk_validated"
        
        # For low-risk content, allow through
        logger.info("[SafetyGuard] Low-risk content detected, allowing through")
        return True, "low_risk"


# Global instance (optional convenience)
safety_guard = SafetyGuard()