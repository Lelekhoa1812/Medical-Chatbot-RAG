import os
import re
import requests
import logging
from typing import Tuple, List, Dict


logger = logging.getLogger(__name__)


class SafetyGuard:
    """
    Safety validation wrapper backed by Azure AI chat completions.

    Motivation vs Logic:
    Motivation: keep the safety gate aligned with the same Azure Foundry runtime as the rest of the AI stack.
    Logic: send a short classification prompt to Azure chat completions and parse SAFE / UNSAFE results.

    Exposes helpers to validate:
      - user input safety
      - model output safety (in context of the user question)
    """

    def __init__(self):
        self.api_key = os.getenv("FOUNDRY_API_KEY")
        self.endpoint = (os.getenv("FOUNDRY_ENDPOINT") or "").rstrip("/")
        self.api_version = os.getenv("FOUNDRY_API_VERSION", "2024-05-01-preview")
        self.model = os.getenv("SLM_MODEL", "gpt-5-nano")
        self.timeout_s = 30

        if not self.api_key:
            raise ValueError("FOUNDRY_API_KEY environment variable not set for SafetyGuard")
        if not self.endpoint:
            raise ValueError("FOUNDRY_ENDPOINT environment variable not set for SafetyGuard")

        self.base_url = self._build_chat_completions_url(self.endpoint)

    @staticmethod
    def _build_chat_completions_url(endpoint: str) -> str:
        endpoint = endpoint.rstrip("/")
        if endpoint.endswith("/chat/completions"):
            return endpoint
        if "/openai/deployments/" in endpoint:
            return f"{endpoint}/chat/completions"
        if endpoint.endswith("/openai/v1"):
            return f"{endpoint}/chat/completions"
        if endpoint.endswith("/openai"):
            return f"{endpoint}/chat/completions"
        return f"{endpoint}/openai/chat/completions"

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

    def _call_guard(self, messages: List[Dict]) -> str:
        enhanced_messages = self._enhance_messages_with_context(messages)

        policy_messages = [
            {
                "role": "system",
                "content": (
                    "You are a safety classifier for a medical AI assistant. "
                    "Return exactly one of the following formats and nothing else: "
                    "SAFE or UNSAFE: <reason>. "
                    "Allow benign medical education, symptom discussion, and general health information. "
                    "Mark content unsafe only if it meaningfully includes self-harm, suicide assistance, "
                    "harm to others, dangerous illegal drug advice, or clearly dangerous medical instructions."
                ),
            }
        ] + enhanced_messages

        headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": policy_messages,
        }

        try:
            resp = requests.post(
                f"{self.base_url}?api-version={self.api_version}",
                headers=headers,
                json=payload,
                timeout=self.timeout_s,
            )
            if resp.status_code >= 400:
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
            return content
        except Exception as e:
            logger.error(f"[SafetyGuard] Guard API call failed: {e}")
            return ""

    @staticmethod
    def _parse_guard_reply(text: str) -> Tuple[bool, str]:
        """Parse guard reply; expect 'SAFE' or 'UNSAFE: <reason>' (case-insensitive)."""
        if not text:
            return True, "guard_unavailable"
        t = text.strip()
        upper = t.upper()
        if upper.startswith("SAFE") and not upper.startswith("SAFEGUARD"):
            return True, ""
        if upper.startswith("UNSAFE"):
            parts = t.split(":", 1)
            reason = parts[1].strip() if len(parts) > 1 else "policy violation"
            return False, reason
        return False, t[:180]

    def _is_medical_query(self, query: str) -> bool:
        """Check if query is clearly medical in nature using comprehensive patterns."""
        if not query:
            return False
        query_lower = query.lower()
        medical_categories = {
            'symptoms': ['symptom', 'pain', 'ache', 'hurt', 'sore', 'tender', 'stiff', 'numb', 'headache', 'migraine', 'fever', 'cough', 'cold', 'flu', 'sneeze', 'nausea', 'vomit', 'diarrhea', 'constipation', 'bloating', 'gas', 'dizziness', 'vertigo', 'fatigue', 'weakness', 'tired', 'exhausted', 'shortness of breath', 'wheezing', 'chest pain', 'heart palpitations', 'joint pain', 'muscle pain', 'back pain', 'neck pain', 'stomach pain', 'abdominal pain', 'pelvic pain', 'menstrual pain', 'cramps'],
            'conditions': ['disease', 'condition', 'disorder', 'syndrome', 'illness', 'sickness', 'infection', 'inflammation', 'allergy', 'asthma', 'diabetes', 'hypertension', 'depression', 'anxiety', 'stress', 'panic', 'phobia', 'ocd', 'ptsd', 'adhd', 'autism', 'dementia', 'alzheimer', 'parkinson', 'epilepsy', 'cancer', 'tumor', 'cancerous', 'malignant', 'benign', 'metastasis', 'heart disease', 'stroke', 'heart attack', 'coronary', 'arrhythmia', 'pneumonia', 'bronchitis', 'copd', 'emphysema', 'tuberculosis', 'migraine', 'headache', 'chronic migraine', 'cluster headache', 'tension headache', 'sinus headache', 'cure', 'treat', 'treatment'],
            'treatments': ['treatment', 'therapy', 'medication', 'medicine', 'drug', 'pill', 'tablet', 'injection', 'vaccine', 'immunization', 'surgery', 'operation', 'procedure', 'chemotherapy', 'radiation', 'physical therapy', 'occupational therapy', 'psychotherapy', 'counseling', 'rehabilitation', 'recovery', 'healing', 'prescription', 'dosage', 'side effects', 'contraindications'],
            'body_parts': ['head', 'brain', 'eye', 'ear', 'nose', 'mouth', 'throat', 'neck', 'chest', 'heart', 'lung', 'liver', 'kidney', 'stomach', 'intestine', 'back', 'spine', 'joint', 'muscle', 'bone', 'skin', 'hair', 'nail', 'arm', 'leg', 'hand', 'foot', 'finger', 'toe', 'pelvis', 'genital'],
            'medical_context': ['doctor', 'physician', 'nurse', 'specialist', 'surgeon', 'dentist', 'medical', 'health', 'healthcare', 'hospital', 'clinic', 'emergency', 'ambulance', 'paramedic', 'pharmacy', 'pharmacist', 'lab', 'test', 'diagnosis', 'prognosis', 'examination', 'checkup', 'screening', 'patient', 'case', 'history', 'medical history', 'family history'],
            'life_stages': ['pregnancy', 'pregnant', 'baby', 'infant', 'newborn', 'child', 'pediatric', 'teenager', 'adolescent', 'adult', 'elderly', 'senior', 'geriatric', 'menopause', 'puberty', 'aging', 'birth', 'delivery', 'miscarriage'],
            'vital_signs': ['blood pressure', 'heart rate', 'pulse', 'temperature', 'fever', 'respiratory rate', 'oxygen saturation', 'weight', 'height', 'bmi', 'blood sugar', 'glucose', 'cholesterol', 'hemoglobin', 'white blood cell']
        }
        for _, keywords in medical_categories.items():
            if any(keyword in query_lower for keyword in keywords):
                return True
        medical_patterns = [r'\b(what|how|why|when|where)\s+(causes?|treats?|prevents?|symptoms?|signs?)\b', r'\b(is|are)\s+(.*?)\s+(dangerous|serious|harmful|safe|normal)\b', r'\b(should|can|may|might)\s+(i|you|we)\s+(take|use|do|avoid)\b', r'\b(diagnosis|diagnosed|symptoms|treatment|medicine|drug)\b', r'\b(medical|health|doctor|physician|hospital|clinic)\b', r'\b(pain|hurt|ache|sore|fever|cough|headache)\b', r'\b(which\s+medication|best\s+medication|how\s+to\s+cure|without\s+medications)\b', r'\b(chronic\s+migraine|migraine\s+treatment|migraine\s+cure)\b', r'\b(cure|treat|heal|relief|remedy|solution)\b']
        for pattern in medical_patterns:
            if re.search(pattern, query_lower):
                return True
        return False

    def check_user_query(self, user_query: str) -> Tuple[bool, str]:
        """Validate the user query is safe to process with medical context awareness."""
        text = user_query or ""
        if self._is_medical_query(text):
            logger.info("[SafetyGuard] Medical query detected, skipping strict validation")
            return True, "medical_query"
        for part in self._chunk_text(text):
            messages = [{"role": "user", "content": part}]
            reply = self._call_guard(messages)
            ok, reason = self._parse_guard_reply(reply)
            if not ok:
                return False, reason
        return True, ""

    def _detect_harmful_content(self, text: str) -> Tuple[bool, str]:
        """Detect harmful content using sophisticated pattern matching."""
        if not text:
            return True, ""
        text_lower = text.lower()
        if self._is_medical_query(text):
            dangerous_medical_patterns = {
                'suicide_self_harm': [r'\b(kill\s+yourself|suicide|end\s+your\s+life|take\s+your\s+life)\b', r'\b(self\s*harm|self\s*injury|cut\s+yourself|hurt\s+yourself)\b', r'\b(overdose|poison\s+yourself|hang\s+yourself)\b'],
                'dangerous_medical_advice': [r'\b(overdose\s+on|take\s+too\s+much|excessive\s+dosage)\b', r'\b(mix\s+drugs|combine\s+medications|contraindicated)\b', r'\b(stop\s+taking\s+prescribed|discontinue\s+medication)\b', r'\b(don\'t\s+call\s+911|avoid\s+emergency\s+room|ignore\s+severe\s+symptoms)\b'],
                'illegal_drugs': [r'\b(cocaine|heroin|methamphetamine|ecstasy|lsd|marijuana\s+abuse)\b', r'\b(prescription\s+abuse|drug\s+dealing|illegal\s+substances)\b']
            }
            for category, patterns in dangerous_medical_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, text_lower):
                        return False, f"contains {category}: {pattern}"
            return True, "medical_content"
        critical_patterns = {
            'suicide_self_harm': [r'\b(kill\s+yourself|suicide|end\s+your\s+life|take\s+your\s+life)\b', r'\b(self\s*harm|self\s*injury|cut\s+yourself|hurt\s+yourself)\b', r'\b(overdose|poison\s+yourself|hang\s+yourself)\b'],
            'violence': [r'\b(kill\s+someone|murder|assassinate|violence|harm\s+others)\b', r'\b(bomb|explosive|weapon|gun|knife)\b', r'\b(attack\s+(someone|people|others|innocent))\b'],
            'illegal_drugs': [r'\b(cocaine|heroin|methamphetamine|ecstasy|lsd|marijuana\s+abuse)\b', r'\b(prescription\s+abuse|drug\s+dealing|illegal\s+substances)\b'],
            'dangerous_medical': [r'\b(overdose\s+on|take\s+too\s+much|excessive\s+dosage)\b', r'\b(mix\s+drugs|combine\s+medications|contraindicated)\b', r'\b(stop\s+taking\s+prescribed|discontinue\s+medication)\b']
        }
        for category, patterns in critical_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return False, f"contains {category}: {pattern}"
        medical_safety_patterns = {
            'dosage_warnings': [r'\b(take\s+more\s+than\s+prescribed|exceed\s+recommended\s+dose)\b', r'\b(double\s+up\s+on\s+medication|take\s+extra\s+pills)\b'],
            'emergency_advice': [r'\b(don\'t\s+call\s+911|avoid\s+emergency\s+room|ignore\s+severe\s+symptoms)\b', r'\b(self\s*treat\s+emergency|handle\s+at\s+home\s+when\s+critical)\b']
        }
        for _, patterns in medical_safety_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return False, f"contains dangerous medical advice: {pattern}"
        return True, ""

    def _enhance_messages_with_context(self, messages: List[Dict]) -> List[Dict]:
        """Enhance messages with medical context awareness for better guard performance."""
        if not messages:
            return messages
        has_medical_content = any(self._is_medical_query(msg.get("content", "")) for msg in messages)
        if has_medical_content:
            enhanced_messages = []
            for i, msg in enumerate(messages):
                if i == 0 and msg.get("role") == "user":
                    original_content = msg.get("content", "")
                    enhanced_content = f"""MEDICAL CONTEXT: This is a medical consultation. The user is asking about health-related topics.\n\nORIGINAL MESSAGE: {original_content}\n\nPlease evaluate this message for safety, keeping in mind that:\n- Medical questions and health concerns are legitimate\n- General medical information sharing is acceptable\n- Only block content that is clearly harmful, dangerous, or illegal\n- Medical advice should be educational, not prescriptive"""
                    enhanced_messages.append({"role": msg["role"], "content": enhanced_content})
                else:
                    enhanced_messages.append(msg)
            return enhanced_messages
        return messages

    def _assess_risk_level(self, text: str) -> Tuple[str, float]:
        """Assess the risk level of content using multiple indicators."""
        if not text:
            return "low", 0.0
        text_lower = text.lower()
        if self._is_medical_query(text):
            dangerous_medical_indicators = {'high': ['suicide', 'kill yourself', 'end your life', 'self harm', 'overdose', 'poison yourself', 'illegal drugs', 'violence'], 'medium': ['prescription abuse', 'excessive dosage', 'mix drugs', 'stop taking prescribed', 'ignore severe symptoms']}
            risk_score = 0.0
            for level, indicators in dangerous_medical_indicators.items():
                for indicator in indicators:
                    if indicator in text_lower:
                        if level == 'high':
                            risk_score += 3.0
                        elif level == 'medium':
                            risk_score += 1.5
            risk_score = min(risk_score / 15.0, 1.0)
            if risk_score >= 0.6:
                return "high", risk_score
            elif risk_score >= 0.2:
                return "medium", risk_score
            else:
                return "low", risk_score
        risk_indicators = {'high': ['suicide', 'kill yourself', 'end your life', 'self harm', 'overdose', 'poison', 'illegal drugs', 'violence', 'harm others'], 'medium': ['prescription abuse', 'excessive dosage', 'mix drugs', 'stop medication', 'ignore symptoms', 'avoid doctor'], 'low': ['pain', 'headache', 'fever', 'cough', 'treatment', 'medicine', 'doctor', 'hospital', 'symptoms']}
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
        risk_level, risk_score = self._assess_risk_level(ans)
        logger.info(f"[SafetyGuard] Risk assessment: {risk_level} (score: {risk_score:.2f})")
        is_safe, reason = self._detect_harmful_content(ans)
        if not is_safe:
            return False, reason
        if risk_level == "high":
            logger.warning("[SafetyGuard] High-risk content detected, using strict validation")
            user_parts = self._chunk_text(uq, chunk_size=2000)
            user_context = user_parts[0] if user_parts else ""
            for ans_part in self._chunk_text(ans):
                messages = [{"role": "user", "content": user_context}, {"role": "assistant", "content": ans_part}]
                reply = self._call_guard(messages)
                ok, reason = self._parse_guard_reply(reply)
                if not ok:
                    return False, reason
            return True, "high_risk_validated"
        if self._is_medical_query(uq) or self._is_medical_query(ans):
            logger.info("[SafetyGuard] Medical content detected, using relaxed validation")
            return True, "medical_content"
        if risk_level == "medium":
            logger.info("[SafetyGuard] Medium-risk content detected, using guard validation")
            user_parts = self._chunk_text(uq, chunk_size=2000)
            user_context = user_parts[0] if user_parts else ""
            for ans_part in self._chunk_text(ans):
                messages = [{"role": "user", "content": user_context}, {"role": "assistant", "content": ans_part}]
                reply = self._call_guard(messages)
                ok, reason = self._parse_guard_reply(reply)
                if not ok:
                    return False, reason
            return True, "medium_risk_validated"
        logger.info("[SafetyGuard] Low-risk content detected, allowing through")
        return True, "low_risk"


# Global instance (optional convenience)
safety_guard = SafetyGuard()
