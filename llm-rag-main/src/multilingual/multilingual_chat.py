"""
Multilingual Legal Chat Interface for Indian Languages.

This module provides multilingual support for legal queries in Hindi, Tamil, 
Malayalam, and other Indian languages with translation capabilities.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class SupportedLanguage(Enum):
    """Supported languages for multilingual chat."""
    ENGLISH = "en"
    HINDI = "hi"
    TAMIL = "ta"
    MALAYALAM = "ml"
    BENGALI = "bn"
    GUJARATI = "gu"
    MARATHI = "mr"
    TELUGU = "te"
    KANNADA = "kn"
    PUNJABI = "pa"

@dataclass
class TranslationResult:
    """Result of translation operation."""
    original_text: str
    translated_text: str
    source_language: SupportedLanguage
    target_language: SupportedLanguage
    confidence: float
    translation_method: str

@dataclass
class MultilingualResponse:
    """Response in multiple languages."""
    english_response: str
    native_response: str
    detected_language: SupportedLanguage
    translation_confidence: float
    code_switching_detected: bool

class LanguageDetector:
    """Simple language detection for Indian languages."""
    
    def __init__(self):
        """Initialize language detector."""
        # Language patterns based on Unicode ranges and common words
        self.language_patterns = {
            SupportedLanguage.HINDI: {
                'unicode_range': r'[\u0900-\u097F]',
                'common_words': ['क्या', 'है', 'में', 'का', 'के', 'को', 'से', 'पर', 'और', 'या'],
                'script_name': 'Devanagari'
            },
            SupportedLanguage.TAMIL: {
                'unicode_range': r'[\u0B80-\u0BFF]',
                'common_words': ['என்ன', 'இது', 'அது', 'இல்', 'உள்ள', 'மற்றும்', 'அல்லது'],
                'script_name': 'Tamil'
            },
            SupportedLanguage.MALAYALAM: {
                'unicode_range': r'[\u0D00-\u0D7F]',
                'common_words': ['എന്ത്', 'ഇത്', 'അത്', 'ഉള്ള', 'കൂടാതെ', 'അല്ലെങ്കിൽ'],
                'script_name': 'Malayalam'
            },
            SupportedLanguage.BENGALI: {
                'unicode_range': r'[\u0980-\u09FF]',
                'common_words': ['কি', 'এই', 'সেই', 'আছে', 'এবং', 'অথবা'],
                'script_name': 'Bengali'
            },
            SupportedLanguage.GUJARATI: {
                'unicode_range': r'[\u0A80-\u0AFF]',
                'common_words': ['શું', 'આ', 'તે', 'છે', 'અને', 'અથવા'],
                'script_name': 'Gujarati'
            },
            SupportedLanguage.MARATHI: {
                'unicode_range': r'[\u0900-\u097F]',  # Same as Hindi but different words
                'common_words': ['काय', 'हे', 'ते', 'आहे', 'आणि', 'किंवा'],
                'script_name': 'Devanagari'
            },
            SupportedLanguage.TELUGU: {
                'unicode_range': r'[\u0C00-\u0C7F]',
                'common_words': ['ఏమి', 'ఇది', 'అది', 'ఉంది', 'మరియు', 'లేదా'],
                'script_name': 'Telugu'
            },
            SupportedLanguage.KANNADA: {
                'unicode_range': r'[\u0C80-\u0CFF]',
                'common_words': ['ಏನು', 'ಇದು', 'ಅದು', 'ಇದೆ', 'ಮತ್ತು', 'ಅಥವಾ'],
                'script_name': 'Kannada'
            },
            SupportedLanguage.PUNJABI: {
                'unicode_range': r'[\u0A00-\u0A7F]',
                'common_words': ['ਕੀ', 'ਇਹ', 'ਉਹ', 'ਹੈ', 'ਅਤੇ', 'ਜਾਂ'],
                'script_name': 'Gurmukhi'
            }
        }
    
    def detect_language(self, text: str) -> Tuple[SupportedLanguage, float]:
        """
        Detect language of input text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Tuple of (detected_language, confidence_score)
        """
        if not text.strip():
            return SupportedLanguage.ENGLISH, 0.0
        
        # Check for English first (if only ASCII)
        if text.isascii():
            return SupportedLanguage.ENGLISH, 0.9
        
        language_scores = {}
        
        # Check Unicode ranges and common words
        for lang, patterns in self.language_patterns.items():
            score = 0.0
            
            # Unicode range matching
            unicode_matches = len(re.findall(patterns['unicode_range'], text))
            if unicode_matches > 0:
                score += (unicode_matches / len(text)) * 0.7
            
            # Common words matching
            text_lower = text.lower()
            word_matches = sum(1 for word in patterns['common_words'] if word in text_lower)
            if word_matches > 0:
                score += (word_matches / len(patterns['common_words'])) * 0.3
            
            if score > 0:
                language_scores[lang] = score
        
        if not language_scores:
            return SupportedLanguage.ENGLISH, 0.5
        
        # Return language with highest score
        best_lang = max(language_scores.items(), key=lambda x: x[1])
        return best_lang[0], min(best_lang[1], 1.0)
    
    def detect_code_switching(self, text: str) -> bool:
        """Detect if text contains code-switching (multiple languages)."""
        detected_scripts = set()
        
        for lang, patterns in self.language_patterns.items():
            if re.search(patterns['unicode_range'], text):
                detected_scripts.add(patterns['script_name'])
        
        # If ASCII + non-ASCII or multiple scripts detected
        has_ascii = bool(re.search(r'[a-zA-Z]', text))
        has_non_ascii = bool(re.search(r'[^\x00-\x7F]', text))
        
        return (has_ascii and has_non_ascii) or len(detected_scripts) > 1

class SimpleTranslator:
    """Simple translation system for legal terms."""
    
    def __init__(self):
        """Initialize translator with legal term dictionaries."""
        self.legal_translations = self._load_legal_translations()
        self.common_translations = self._load_common_translations()
    
    def _load_legal_translations(self) -> Dict:
        """Load legal term translations."""
        return {
            SupportedLanguage.HINDI: {
                'court': 'न्यायालय',
                'judge': 'न्यायाधीश',
                'lawyer': 'वकील',
                'case': 'मामला',
                'appeal': 'अपील',
                'judgment': 'फैसला',
                'accused': 'आरोपी',
                'plaintiff': 'वादी',
                'defendant': 'प्रतिवादी',
                'evidence': 'सबूत',
                'witness': 'गवाह',
                'bail': 'जमानत',
                'fine': 'जुर्माना',
                'sentence': 'सजा',
                'law': 'कानून',
                'legal': 'कानूनी',
                'rights': 'अधिकार',
                'justice': 'न्याय'
            },
            SupportedLanguage.TAMIL: {
                'court': 'நீதிமன்றம்',
                'judge': 'நீதிபதி',
                'lawyer': 'வழக்கறிஞர்',
                'case': 'வழக்கு',
                'appeal': 'மேல்முறையீடு',
                'judgment': 'தீர்ப்பு',
                'accused': 'குற்றம் சாட்டப்பட்டவர்',
                'plaintiff': 'வாதி',
                'defendant': 'பிரதிவாதி',
                'evidence': 'சாక்ஷ்யம்',
                'witness': 'சாட்சி',
                'bail': 'பிணை',
                'fine': 'அபராதம்',
                'sentence': 'தண்டனை',
                'law': 'சட்டம்',
                'legal': 'சட்டப்பூர்வ',
                'rights': 'உரிமைகள்',
                'justice': 'நீதி'
            },
            SupportedLanguage.MALAYALAM: {
                'court': 'കോടതി',
                'judge': 'ജഡ്ജി',
                'lawyer': 'വക്കീൽ',
                'case': 'കേസ്',
                'appeal': 'അപ്പീൽ',
                'judgment': 'വിധി',
                'accused': 'പ്രതി',
                'plaintiff': 'വാദി',
                'defendant': 'പ്രതിവാദി',
                'evidence': 'തെളിവ്',
                'witness': 'സാക്ഷി',
                'bail': 'ജാമ്യം',
                'fine': 'പിഴ',
                'sentence': 'ശിക്ഷ',
                'law': 'നിയമം',
                'legal': 'നിയമപരമായ',
                'rights': 'അവകാശങ്ങൾ',
                'justice': 'നീതി'
            }
        }
    
    def _load_common_translations(self) -> Dict:
        """Load common word translations."""
        return {
            SupportedLanguage.HINDI: {
                'what': 'क्या',
                'how': 'कैसे',
                'when': 'कब',
                'where': 'कहाँ',
                'why': 'क्यों',
                'who': 'कौन',
                'help': 'मदद',
                'please': 'कृपया',
                'thank you': 'धन्यवाद',
                'yes': 'हाँ',
                'no': 'नहीं',
                'can': 'सकता',
                'will': 'होगा',
                'need': 'चाहिए'
            },
            SupportedLanguage.TAMIL: {
                'what': 'என்ன',
                'how': 'எப்படி',
                'when': 'எப்போது',
                'where': 'எங்கே',
                'why': 'ஏன்',
                'who': 'யார்',
                'help': 'உதவி',
                'please': 'தயவுசெய்து',
                'thank you': 'நன்றி',
                'yes': 'ஆம்',
                'no': 'இல்லை',
                'can': 'முடியும்',
                'will': 'வேண்டும்',
                'need': 'தேவை'
            },
            SupportedLanguage.MALAYALAM: {
                'what': 'എന്ത്',
                'how': 'എങ്ങനെ',
                'when': 'എപ്പോൾ',
                'where': 'എവിടെ',
                'why': 'എന്തുകൊണ്ട്',
                'who': 'ആര്',
                'help': 'സഹായം',
                'please': 'ദയവായി',
                'thank you': 'നന്ദി',
                'yes': 'അതെ',
                'no': 'ഇല്ല',
                'can': 'കഴിയും',
                'will': 'ചെയ്യും',
                'need': 'വേണം'
            }
        }
    
    def translate_to_english(self, text: str, source_language: SupportedLanguage) -> TranslationResult:
        """Translate text from source language to English."""
        if source_language == SupportedLanguage.ENGLISH:
            return TranslationResult(
                original_text=text,
                translated_text=text,
                source_language=source_language,
                target_language=SupportedLanguage.ENGLISH,
                confidence=1.0,
                translation_method="no_translation_needed"
            )
        
        # Simple word-by-word translation for legal terms
        translated_text = text
        confidence = 0.5  # Base confidence
        
        if source_language in self.legal_translations:
            legal_dict = self.legal_translations[source_language]
            common_dict = self.common_translations.get(source_language, {})
            
            # Reverse dictionary for translation
            reverse_legal = {v: k for k, v in legal_dict.items()}
            reverse_common = {v: k for k, v in common_dict.items()}
            
            # Replace legal terms
            for native_term, english_term in reverse_legal.items():
                if native_term in text:
                    translated_text = translated_text.replace(native_term, english_term)
                    confidence += 0.1
            
            # Replace common terms
            for native_term, english_term in reverse_common.items():
                if native_term in text:
                    translated_text = translated_text.replace(native_term, english_term)
                    confidence += 0.05
        
        return TranslationResult(
            original_text=text,
            translated_text=translated_text,
            source_language=source_language,
            target_language=SupportedLanguage.ENGLISH,
            confidence=min(confidence, 1.0),
            translation_method="dictionary_based"
        )
    
    def translate_from_english(self, text: str, target_language: SupportedLanguage) -> TranslationResult:
        """Translate text from English to target language."""
        if target_language == SupportedLanguage.ENGLISH:
            return TranslationResult(
                original_text=text,
                translated_text=text,
                source_language=SupportedLanguage.ENGLISH,
                target_language=target_language,
                confidence=1.0,
                translation_method="no_translation_needed"
            )
        
        translated_text = text
        confidence = 0.5
        
        if target_language in self.legal_translations:
            legal_dict = self.legal_translations[target_language]
            common_dict = self.common_translations.get(target_language, {})
            
            # Replace legal terms
            for english_term, native_term in legal_dict.items():
                if english_term.lower() in text.lower():
                    translated_text = re.sub(
                        r'\b' + re.escape(english_term) + r'\b',
                        native_term,
                        translated_text,
                        flags=re.IGNORECASE
                    )
                    confidence += 0.1
            
            # Replace common terms
            for english_term, native_term in common_dict.items():
                if english_term.lower() in text.lower():
                    translated_text = re.sub(
                        r'\b' + re.escape(english_term) + r'\b',
                        native_term,
                        translated_text,
                        flags=re.IGNORECASE
                    )
                    confidence += 0.05
        
        return TranslationResult(
            original_text=text,
            translated_text=translated_text,
            source_language=SupportedLanguage.ENGLISH,
            target_language=target_language,
            confidence=min(confidence, 1.0),
            translation_method="dictionary_based"
        )

class MultilingualLegalChat:
    """
    Multilingual legal chat interface supporting Indian languages.
    """
    
    def __init__(self, base_chatbot=None):
        """Initialize multilingual chat interface."""
        self.language_detector = LanguageDetector()
        self.translator = SimpleTranslator()
        self.base_chatbot = base_chatbot  # Existing legal chatbot
        
        # Conversation history with language tracking
        self.conversation_history = []
        self.user_preferred_language = SupportedLanguage.ENGLISH
        
        logger.info("Multilingual Legal Chat initialized")
    
    def process_query(self, query: str, preferred_language: Optional[SupportedLanguage] = None) -> MultilingualResponse:
        """
        Process a multilingual legal query.
        
        Args:
            query: User query in any supported language
            preferred_language: User's preferred response language
            
        Returns:
            MultilingualResponse: Response in both English and native language
        """
        logger.info(f"Processing multilingual query: {query[:50]}...")
        
        # Detect input language
        detected_language, detection_confidence = self.language_detector.detect_language(query)
        
        # Detect code-switching
        code_switching = self.language_detector.detect_code_switching(query)
        
        # Update user preferred language if not specified
        if preferred_language is None:
            preferred_language = detected_language if detected_language != SupportedLanguage.ENGLISH else self.user_preferred_language
        
        # Translate query to English if needed
        if detected_language != SupportedLanguage.ENGLISH:
            translation_result = self.translator.translate_to_english(query, detected_language)
            english_query = translation_result.translated_text
            translation_confidence = translation_result.confidence
        else:
            english_query = query
            translation_confidence = 1.0
        
        # Process query with base chatbot (if available)
        if self.base_chatbot:
            try:
                english_response = self.base_chatbot.chat(english_query)
                if isinstance(english_response, dict):
                    english_response = english_response.get('answer', str(english_response))
            except Exception as e:
                logger.error(f"Error with base chatbot: {e}")
                english_response = self._generate_fallback_response(english_query)
        else:
            english_response = self._generate_fallback_response(english_query)
        
        # Translate response to preferred language
        if preferred_language != SupportedLanguage.ENGLISH:
            response_translation = self.translator.translate_from_english(english_response, preferred_language)
            native_response = response_translation.translated_text
        else:
            native_response = english_response
        
        # Store conversation
        self.conversation_history.append({
            'query': query,
            'detected_language': detected_language,
            'english_query': english_query,
            'english_response': english_response,
            'native_response': native_response,
            'preferred_language': preferred_language
        })
        
        return MultilingualResponse(
            english_response=english_response,
            native_response=native_response,
            detected_language=detected_language,
            translation_confidence=translation_confidence,
            code_switching_detected=code_switching
        )
    
    def _generate_fallback_response(self, query: str) -> str:
        """Generate fallback response when base chatbot is not available."""
        query_lower = query.lower()
        
        # Simple pattern matching for common legal queries
        if any(word in query_lower for word in ['bail', 'जमानत', 'பிணை', 'ജാമ്യം']):
            return "Bail is a legal provision that allows an accused person to be released from custody while awaiting trial. The court considers factors like the nature of the offense, flight risk, and likelihood of tampering with evidence."
        
        elif any(word in query_lower for word in ['divorce', 'तलाक', 'விவாகரத்து', 'വിവാഹമോചനം']):
            return "Divorce laws in India are governed by personal laws based on religion. The process involves filing a petition, attempting reconciliation, and obtaining a decree from the family court."
        
        elif any(word in query_lower for word in ['property', 'संपत्ति', 'சொத்து', 'സ്വത്ത്']):
            return "Property law in India covers ownership, transfer, and inheritance of real estate and personal property. It includes provisions for registration, stamp duty, and legal documentation."
        
        elif any(word in query_lower for word in ['rights', 'अधिकार', 'உரிமைகள்', 'അവകാശങ്ങൾ']):
            return "Fundamental rights in India are guaranteed by the Constitution and include right to equality, freedom, life and liberty, religious freedom, cultural and educational rights, and constitutional remedies."
        
        else:
            return "I understand you have a legal query. For specific legal advice, please consult with a qualified lawyer. I can provide general information about Indian laws and legal procedures."
    
    def get_supported_languages(self) -> List[Dict[str, str]]:
        """Get list of supported languages."""
        return [
            {'code': lang.value, 'name': lang.name.title(), 'native_name': self._get_native_name(lang)}
            for lang in SupportedLanguage
        ]
    
    def _get_native_name(self, language: SupportedLanguage) -> str:
        """Get native name of language."""
        native_names = {
            SupportedLanguage.ENGLISH: "English",
            SupportedLanguage.HINDI: "हिन्दी",
            SupportedLanguage.TAMIL: "தமிழ்",
            SupportedLanguage.MALAYALAM: "മലയാളം",
            SupportedLanguage.BENGALI: "বাংলা",
            SupportedLanguage.GUJARATI: "ગુજરાતી",
            SupportedLanguage.MARATHI: "मराठी",
            SupportedLanguage.TELUGU: "తెలుగు",
            SupportedLanguage.KANNADA: "ಕನ್ನಡ",
            SupportedLanguage.PUNJABI: "ਪੰਜਾਬੀ"
        }
        return native_names.get(language, language.name.title())
    
    def set_preferred_language(self, language: SupportedLanguage):
        """Set user's preferred language."""
        self.user_preferred_language = language
        logger.info(f"User preferred language set to: {language.value}")
    
    def get_conversation_history(self) -> List[Dict]:
        """Get conversation history."""
        return self.conversation_history.copy()
    
    def clear_conversation_history(self):
        """Clear conversation history."""
        self.conversation_history.clear()
        logger.info("Conversation history cleared")

def main():
    """Example usage of Multilingual Legal Chat."""
    # Initialize multilingual chat
    multilingual_chat = MultilingualLegalChat()
    
    # Test queries in different languages
    test_queries = [
        "What is bail?",  # English
        "जमानत क्या है?",  # Hindi
        "பிணை என்றால் என்ன?",  # Tamil
        "ജാമ്യം എന്താണ്?",  # Malayalam
        "What are my rights अधिकार?"  # Code-switching
    ]
    
    print("=" * 60)
    print("MULTILINGUAL LEGAL CHAT DEMO")
    print("=" * 60)
    
    # Show supported languages
    print("\nSupported Languages:")
    for lang_info in multilingual_chat.get_supported_languages():
        print(f"- {lang_info['name']} ({lang_info['code']}): {lang_info['native_name']}")
    
    print("\nProcessing Test Queries:")
    print("-" * 40)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: {query}")
        
        response = multilingual_chat.process_query(query)
        
        print(f"   Detected Language: {response.detected_language.value}")
        print(f"   Translation Confidence: {response.translation_confidence:.2f}")
        print(f"   Code-switching: {response.code_switching_detected}")
        print(f"   English Response: {response.english_response[:100]}...")
        if response.native_response != response.english_response:
            print(f"   Native Response: {response.native_response[:100]}...")

if __name__ == "__main__":
    main()
