"""
Multi-Language Support for DISHA
Handles automatic language detection, translation, and voice selection
Supports all Indian regional languages + international languages
"""

from langdetect import detect, DetectorFactory
from googletrans import Translator
import re

# Set seed for consistent language detection
DetectorFactory.seed = 0

class MultiLanguageHandler:
    """Handles automatic language detection and translation for DISHA"""
    
    def __init__(self):
        self.translator = Translator()
        self.current_language = 'en'  # Default to English
        
        # Map language codes to Microsoft Edge TTS voices
        # Covers Indian regional languages + international languages
        # Using warmer, more mature female voices for caring tone
        self.voice_map = {
            # English variants - Aria has warm, empathetic tone
            'en': 'en-US-AriaNeural',
            
            # Indian Languages - Selected warmest female voices
            'hi': 'hi-IN-SwaraNeural',      # Hindi (Female, warm)
            'bn': 'bn-IN-TanishaaNeural',   # Bengali (Female, caring)
            'te': 'te-IN-ShrutiNeural',     # Telugu (Female, gentle)
            'mr': 'mr-IN-AarohiNeural',     # Marathi (Female, warm)
            'ta': 'ta-IN-PallaviNeural',    # Tamil (Female, soothing)
            'gu': 'gu-IN-DhwaniNeural',     # Gujarati (Female, soft)
            'kn': 'kn-IN-SapnaNeural',      # Kannada (Female, caring)
            'ml': 'ml-IN-SobhanaNeural',    # Malayalam (Female, gentle)
            'pa': 'pa-IN-GurpreetNeural',   # Punjabi (Female, warm)
            'or': 'en-IN-NeerjaNeural',     # Odia (using warm Indian English)
            'as': 'en-IN-NeerjaNeural',     # Assamese (using warm Indian English)
            
            # Major International Languages - Warmest female voices
            'es': 'es-ES-ElviraNeural',     # Spanish (Female, warm)
            'fr': 'fr-FR-DeniseNeural',     # French (Female, caring)
            'de': 'de-DE-KatjaNeural',      # German (Female, gentle)
            'it': 'it-IT-ElsaNeural',       # Italian (Female, warm)
            'pt': 'pt-BR-FranciscaNeural',  # Portuguese (Female, soothing)
            'ru': 'ru-RU-DariyaNeural',     # Russian (Female, soft)
            'ja': 'ja-JP-NanamiNeural',     # Japanese (Female, caring)
            'ko': 'ko-KR-SunHiNeural',      # Korean (Female, warm)
            'zh-cn': 'zh-CN-XiaoxiaoNeural',# Chinese (Female, gentle)
            'zh-tw': 'zh-TW-HsiaoChenNeural',# Chinese Traditional (Female, warm)
            'ar': 'ar-SA-ZariyahNeural',    # Arabic (Female, caring)
            'tr': 'tr-TR-EmelNeural',       # Turkish (Female, soothing)
            'nl': 'nl-NL-ColetteNeural',    # Dutch (Female, warm)
            'pl': 'pl-PL-ZofiaNeural',      # Polish (Female, gentle)
            'sv': 'sv-SE-SofieNeural',      # Swedish (Female, caring)
            'vi': 'vi-VN-HoaiMyNeural',     # Vietnamese (Female, soft)
            'th': 'th-TH-PremwadeeNeural',  # Thai (Female, warm)
            'id': 'id-ID-GadisNeural',      # Indonesian (Female, caring)
        }
        
        # Language names for display
        self.language_names = {
            'en': 'English',
            'hi': 'हिंदी (Hindi)',
            'bn': 'বাংলা (Bengali)',
            'te': 'తెలుగు (Telugu)',
            'mr': 'मराठी (Marathi)',
            'ta': 'தமிழ் (Tamil)',
            'gu': 'ગુજરાતી (Gujarati)',
            'kn': 'ಕನ್ನಡ (Kannada)',
            'ml': 'മലയാളം (Malayalam)',
            'pa': 'ਪੰਜਾਬੀ (Punjabi)',
            'es': 'Español (Spanish)',
            'fr': 'Français (French)',
            'de': 'Deutsch (German)',
            'ja': '日本語 (Japanese)',
            'zh-cn': '中文 (Chinese)',
            'ar': 'العربية (Arabic)',
        }
    
    def detect_language(self, text):
        """Detect the language of the input text"""
        try:
            # Clean text for better detection
            clean_text = re.sub(r'[^\w\s]', '', text)
            if len(clean_text.strip()) < 3:
                return self.current_language  # Too short, use current
            
            detected = detect(clean_text)
            self.current_language = detected
            return detected
        except:
            return self.current_language  # Fallback to current language
    
    def translate_to_english(self, text, source_lang=None):
        """Translate text to English for AI processing"""
        try:
            if source_lang is None:
                source_lang = self.detect_language(text)
            
            # If already English, return as-is
            if source_lang == 'en':
                return text, 'en'
            
            # Translate to English
            result = self.translator.translate(text, src=source_lang, dest='en')
            return result.text, source_lang
        except Exception as e:
            print(f"⚠️ Translation error: {e}")
            return text, 'en'  # Fallback to original
    
    def translate_from_english(self, text, target_lang):
        """Translate English response back to user's language"""
        try:
            # If target is English, return as-is
            if target_lang == 'en':
                return text
            
            # Translate from English to target language
            result = self.translator.translate(text, src='en', dest=target_lang)
            return result.text
        except Exception as e:
            print(f"⚠️ Translation error: {e}")
            return text  # Fallback to original
    
    def get_voice_for_language(self, lang_code):
        """Get the appropriate neural voice for the language"""
        # Handle Chinese variants
        if lang_code.startswith('zh'):
            return self.voice_map.get('zh-cn', self.voice_map['en'])
        
        return self.voice_map.get(lang_code, self.voice_map['en'])
    
    def get_language_name(self, lang_code):
        """Get the display name for a language"""
        return self.language_names.get(lang_code, lang_code.upper())
    
    def process_input(self, user_input):
        """Complete pipeline: detect language, translate to English for AI"""
        text_for_ai, detected_lang = self.translate_to_english(user_input)
        
        # Show language detection (only if not English)
        if detected_lang != 'en':
            lang_name = self.get_language_name(detected_lang)
            print(f"🌍 Language detected: {lang_name}")
        
        return text_for_ai, detected_lang
    
    def process_output(self, ai_response, target_lang):
        """Complete pipeline: translate AI response to user's language"""
        if target_lang == 'en':
            return ai_response
        
        translated = self.translate_from_english(ai_response, target_lang)
        return translated


# Global instance
language_handler = MultiLanguageHandler()


# Test function
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  🌍 DISHA Multi-Language System - Test Mode 🌍")
    print("="*60 + "\n")
    
    handler = MultiLanguageHandler()
    
    # Test with different languages
    test_inputs = [
        "Hello, how are you?",
        "मैं बहुत परेशान हूं",  # Hindi: I'm very worried
        "আমি দুঃখিত বোধ করছি",  # Bengali: I'm feeling sad
        "నాకు సహాయం కావాలి",  # Telugu: I need help
        "Je me sens anxieux",  # French: I feel anxious
        "私は悲しいです",  # Japanese: I am sad
    ]
    
    print("Testing language detection and translation:\n")
    for text in test_inputs:
        print(f"Input: {text}")
        translated, lang = handler.translate_to_english(text)
        print(f"  → Detected: {handler.get_language_name(lang)}")
        print(f"  → English: {translated}")
        voice = handler.get_voice_for_language(lang)
        print(f"  → Voice: {voice}")
        print()
    
    print("="*60)
    print("✅ Multi-language system ready!")
    print("="*60 + "\n")
