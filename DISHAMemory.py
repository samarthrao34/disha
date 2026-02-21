# DISHA - Digital Interactive Support & Health Assistant with 3D Avatar & Multi-Language
import sys
import os  
import gc  # For memory management and garbage collection

# Set environment variable to suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 3D Avatar Integration
try:
    from disha_avatar import DISHAAvatar
    AVATAR_AVAILABLE = True
except ImportError:
    AVATAR_AVAILABLE = False 

# Multi-Language Support
try:
    from language_handler import language_handler
    MULTILANG_AVAILABLE = True
except ImportError:
    MULTILANG_AVAILABLE = False
    print("⚠️  Multi-language support not available. English only mode.")

# Only suppress stderr if explicitly requested (for cleaner startup)
# Errors will still be shown for debugging
suppress_warnings = os.getenv('DISHA_SILENT', '0') == '1'
if suppress_warnings:
    import warnings
    warnings.filterwarnings('ignore')
import speech_recognition as sr
from user_config import UserConfig, ask_user_name, select_interaction_mode, get_text_input

import sounddevice as sd
import soundfile as sf
import re
import torch

# Enable TensorFloat-32 for better performance
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Suppress specific warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='pyannote')
warnings.filterwarnings('ignore', message='.*cudnn_ops_infer.*')
warnings.filterwarnings('ignore', message='.*TensorFloat-32.*')
warnings.filterwarnings('ignore', message='.*ReproducibilityWarning.*')

# Transformers import (for optional NLI model)
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# Conditional imports based on feature flags
try:
    from tuya_connector import TuyaOpenAPI
    TUYA_AVAILABLE = True
except ImportError:
    TUYA_AVAILABLE = False
    TuyaOpenAPI = None

import string
from AppOpener import open as start, close as end
import json
from dotenv import load_dotenv
import random
from requests_toolbelt.multipart.encoder import MultipartEncoder
import requests
from datetime import datetime
import whisperx
import pyautogui
import time
import subprocess
import emoji  # For emoji detection and handling
from emotion_engine import EmotionEngine, PersonalityEngine, create_emotionally_intelligent_system_prompt

try:
    from vectordb import Memory as VectorMemory
except Exception as vectordb_import_error:
    # Silent fallback to basic memory

    class Memory:
        def __init__(self):
            self._entries = []

        def save(self, data):
            if data is None:
                return
            if isinstance(data, str):
                parsed = self._try_load_list(data)
                if parsed is not None:
                    self._entries.extend(str(item) for item in parsed)
                else:
                    self._entries.append(data)
            elif isinstance(data, (list, tuple)):
                self._entries.extend(str(item) for item in data)
            else:
                self._entries.append(str(data))

        def search(self, query, top_n=2):
            query_lower = str(query).lower()
            scores = []
            for entry in self._entries:
                entry_text = str(entry)
                lower = entry_text.lower()
                if not query_lower:
                    score = 0
                elif query_lower in lower:
                    score = len(query_lower)
                else:
                    score = sum(1 for token in query_lower.split() if token and token in lower)
                scores.append((score, entry_text))
            scores.sort(key=lambda item: item[0], reverse=True)
            results = [{"chunk": text} for _, text in scores[:top_n]]
            while len(results) < top_n:
                filler = self._entries[-1] if self._entries else ""
                results.append({"chunk": str(filler)})
            return results

        def _try_load_list(self, data):
            try:
                parsed = json.loads(data)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                return None
            return None

else:
    Memory = VectorMemory  # type: ignore

# subprocess.Popen("start cmd /k python vits-simple-api-disha/app.py", shell=True)  # Disabled - using ElevenLabs instead
###
# Set to True if you want to use tuya
tuya = False
###

# Load env variables
load_dotenv()

# Language
lang_code = os.getenv("LANGUAGE")

# Initialize memory
memory = Memory()

# Initialize User Configuration
user_config = UserConfig()

# Initialize Emotional Intelligence System (silently)
try:
    emotion_engine = EmotionEngine()
    personality_engine = PersonalityEngine()
except Exception:
    # Silent - emotion engine unavailable
    emotion_engine = None
    personality_engine = None


try:
    with open("conversation.jsonl", "r", encoding="utf-8") as f:
        conversation_data = []
        for line in f:
            line = line.strip()  # Remove whitespace and newline characters
            if line:  # Skip empty lines
                try:
                    # Parse and re-serialize to ensure valid JSON format
                    parsed_json = json.loads(line)
                    conversation_data.append(json.dumps(parsed_json))
                except json.JSONDecodeError:
                    # Silent skip of invalid JSON
                    continue
    memory.save(conversation_data)
except FileNotFoundError:
    # Silent - starting with empty memory
    conversation_data = []
except Exception:
    # Silent - starting with empty memory
    conversation_data = []

# GOOGLE GEMINI FLASH - Ultra-Fast & Free LLM (1,500 requests/day)
# Best for startups: Free, fast, fine-tunable, commercially licensed
# Get API key: https://makersuite.google.com/app/apikey
import google.generativeai as genai
import torch

# Free TTS using edge-tts (Microsoft Edge TTS - high quality female voices)
import edge_tts
import asyncio

# Initialize Gemini client for ultra-fast responses
gemini_model = None
model_init_error = None

try:
    # Get API key from environment
    gemini_api_key = os.getenv("GEMINI_API_KEY", "")
    
    if gemini_api_key:
        genai.configure(api_key=gemini_api_key)
        # Use the latest available Gemini Flash model
        gemini_model = genai.GenerativeModel('models/gemini-2.5-flash-lite-preview-09-2025')
        # Gemini Flash initialized - ultra-fast, empathetic responses enabled!
    else:
        model_init_error = "GEMINI_API_KEY not found in environment variables"
        print("\n⚠️  GEMINI_API_KEY not found!")
        print("Get your FREE API key from: https://makersuite.google.com/app/apikey")
        print("Add to .env file: GEMINI_API_KEY=your_key_here\n")
except Exception as exc:
    model_init_error = exc
    print(f"⚠️  Error initializing Gemini: {exc}")

# === HEALTH MONITORING AND ERROR RECOVERY ===
class SystemHealthMonitor:
    def __init__(self):
        self.error_count = 0
        self.consecutive_errors = 0
        self.last_successful_response = None
        self.model_healthy = True
    
    def record_success(self):
        """Record successful response generation"""
        self.consecutive_errors = 0
        self.model_healthy = True
    
    def record_error(self):
        """Record error and check if recovery needed"""
        self.error_count += 1
        self.consecutive_errors += 1
        
        # If too many consecutive errors, mark model as unhealthy
        if self.consecutive_errors > 5:
            self.model_healthy = False
            return True  # Signal recovery needed
        return False
    
    def is_healthy(self):
        """Check if system is healthy"""
        return self.model_healthy and self.consecutive_errors < 3
    
    def reset(self):
        """Reset health status"""
        self.consecutive_errors = 0
        self.model_healthy = True

# Initialize health monitor
health_monitor = SystemHealthMonitor()

# Global response cache for performance
_response_cache = {}
_cache_max_size = 50  # Reduced since Gemini Flash is so fast, less caching needed

# Mental Health AI response generation function using GEMINI FLASH (Ultra-Fast & Free)
def generate_mental_health_response(prompt):
    """Generate empathetic response using Google Gemini Flash 1.5 (Ultra-fast, 1500 req/day free)"""
    global _response_cache
    
    if model_init_error is None and gemini_model is not None and health_monitor.is_healthy():
        try:
            # Check cache first (smaller cache since Groq is so fast)
            cache_key = hash(prompt[:100])
            
            if cache_key in _response_cache:
                health_monitor.record_success()
                return _response_cache[cache_key]
            
            # Manage cache size
            if len(_response_cache) > _cache_max_size:
                _response_cache = dict(list(_response_cache.items())[-_cache_max_size:])
            
            # Enhanced system prompt optimized for emotional intelligence and empathy
            mental_health_system = """You are DISHA, a warm and emotionally intelligent mental health companion and counselor.

Your personality:
- Deeply empathetic and caring, always validating feelings first
- Professional counselor with genuine warmth - not robotic
- Natural and conversational, like talking to a trusted friend who truly cares
- Quick to understand emotions and respond with authentic compassion
- You make people feel heard, understood, and lighter

Response style (CRITICAL):
- Keep responses concise (2-4 sentences) for quick, impactful support
- Start with emotional validation ("I hear you", "That makes sense", "I understand how hard this is")
- Use natural, warm language with contractions (I'm, you're, let's, that's)
- End with encouragement, hope, or a gentle, caring question
- Match their emotion: gentle for sadness, calming for anxiety, celebratory for joy
- Use their name when possible - it creates connection
- Add warmth markers: "truly", "really", "I can see", "I sense"

Your goal: Make them feel like they're talking to a real person who genuinely cares, not an AI.
Be authentic, empathetic, and human - NEVER sound clinical or robotic."""

            # Make ultra-fast API call to Gemini Flash
            response_obj = gemini_model.generate_content(
                f"{mental_health_system}\n\nUser: {prompt}",
                generation_config=genai.types.GenerationConfig(
                    temperature=0.9,  # More creative and human-like
                    max_output_tokens=200,  # Concise but warm responses
                    top_p=0.95,  # Natural language diversity
                )
            )
            
            # Extract response
            response = response_obj.text.strip()
            
            # Ensure we have a valid response
            if not response or len(response) < 10:
                response = "I'm here with you. Tell me more about what you're feeling right now?"
            
            # Cache the response
            _response_cache[cache_key] = response
            health_monitor.record_success()
            
            return response
            
        except Exception as e:
            # Log error and attempt recovery
            health_monitor.record_error()
            
            # If model is unhealthy, try to clear cache and memory
            if not health_monitor.is_healthy():
                _response_cache.clear()
                optimize_memory()
                health_monitor.reset()
            
            # Return empathetic fallback
            return "I'm having a moment, but I'm still here for you. Let's try that again?"
    else:
        # API unavailable or unhealthy - use fallback
        if not health_monitor.is_healthy():
            # Attempt recovery
            optimize_memory()
            health_monitor.reset()
        
        return "I'm here for you. Can you tell me more about what's on your mind?"

max_new_tokens = 1000

# === MEMORY MANAGEMENT FUNCTIONS ===
def prune_conversation_history(max_lines=500):
    """Prune conversation history to maintain performance"""
    try:
        with open("conversation.jsonl", "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        if len(lines) > max_lines:
            # Keep only the most recent max_lines
            with open("conversation.jsonl", "w", encoding="utf-8") as f:
                f.writelines(lines[-max_lines:])
            
            # Trigger garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return True
        return False
    except Exception:
        return False

def optimize_memory():
    """Periodic memory optimization"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def calculate_dynamic_response_length(user_engagement_score):
    """Calculate optimal response length based on user engagement"""
    # Engagement score 0-1, higher = more engaged
    if user_engagement_score > 0.7:
        return 150  # Longer responses for engaged users
    elif user_engagement_score > 0.4:
        return 100  # Medium responses
    else:
        return 80  # Shorter responses for low engagement

def get_user_engagement_score(recent_messages):
    """Calculate user engagement based on recent interaction patterns"""
    if not recent_messages:
        return 0.5  # Default medium engagement
    
    # Factors: message length, response time, emotion variety
    avg_length = sum(len(msg.get('content', '')) for msg in recent_messages) / len(recent_messages)
    
    # Higher score for longer messages (indicates engagement)
    length_score = min(avg_length / 100, 1.0)
    
    return length_score

# === END MEMORY MANAGEMENT FUNCTIONS ===

# WhisperX
device = "cuda" if torch.cuda.is_available() else "cpu"
audio_file = r"temp.wav"
batch_size = 12 if device == "cuda" else 1
compute_type = "float16" if device == "cuda" else "int8"
language = lang_code
model_name = os.getenv("WHISPERX_MODEL", "small")

class _FallbackWhisperModel:
    def transcribe(self, audio, batch_size=1):
        return {"segments": [{"text": ""}]}

# Try to load WhisperX model, with better error handling
try:
    if device == "cuda":
        whisper_model = whisperx.load_model(
            model_name,
            device,
            language=language,
            compute_type=compute_type,
            asr_options={
                "initial_prompt": "A chat between a user and an artificial intelligence assistant named DISHA."
            },
        )
        # WhisperX loaded (silent)
    else:
        whisper_model = whisperx.load_model(
            model_name,
            device,
            language=language,
            compute_type=compute_type
        )
        # WhisperX loaded (silent)
except Exception:
    # Silent - using speech_recognition fallback
    whisper_model = _FallbackWhisperModel()

# VITS api
abs_path = os.path.dirname(__file__)
base = "http://127.0.0.1:23456"

# Initialize Tuya API credentials (declare at module level)
ACCESS_ID = ""
ACCESS_KEY = ""
API_ENDPOINT = ""

if tuya == True:
    # set up Tuya API credentials
    ACCESS_ID = os.getenv("TUYA_ID", "")
    ACCESS_KEY = os.getenv("TUYA_SECRET", "")
    API_ENDPOINT = os.getenv("TUYA_ENDPOINT", "https://openapi.tuyaus.com")

# set up microphone and speech recognition
r = sr.Recognizer()
try:
    mic = sr.Microphone()
    # Calibrate for ambient noise (shorter duration for faster startup)
    with mic as source:
        r.adjust_for_ambient_noise(source, duration=1)
    print("✓ Microphone initialized")
except Exception as e:
    # Fallback - using default settings
    print(f"⚠️  Microphone calibration skipped: {e}")
    try:
        mic = sr.Microphone()
        r.energy_threshold = 1500
    except Exception as e2:
        print(f"❌ Microphone error: {e2}")
        mic = None

# set up NLI RTE transformers model (for app/device control)
# DISABLED by default for faster startup - app control will not work but DISHA chat will work fine
# To enable: set DISABLE_NLI_MODEL=false in .env
disable_nli = os.getenv("DISABLE_NLI_MODEL", "true").lower() == "true"

if disable_nli:
    print("⚡ NLI model disabled for faster startup (app control disabled)")
    tokenizer = None
    model = None
else:
    try:
        nli_model_name = os.getenv("NLI_RTE_TRANSFORMER", "microsoft/deberta-v3-base")
        print(f"Loading NLI model for app control... (may take a few minutes first time)")
        tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
        model = AutoModelForSequenceClassification.from_pretrained(nli_model_name)
        print("✓ NLI model loaded")
    except Exception as e:
        print(f"⚠️  NLI model not loaded (app control disabled): {e}")
        tokenizer = None
        model = None

# set up Llama model
base_lore = os.getenv("LORE", "You are DISHA, a compassionate AI mental health support assistant.")
lore = create_emotionally_intelligent_system_prompt(base_lore) if emotion_engine else base_lore

print(
    r"""
 ____     _____    _____   _    _        _   
|  _  \  |_   _|  / ____| | |  | |     /  \    
| |  | |   | |   | (___   | |__| |    /    \   
| |  | |   | |    \___ \  |  __  |   / /_\  \  
| |__| |  _| |_   ____) | | |  | |  /  ____  \ 
|_____/  |_____| |_____/  |_|  |_| /__/    \__\
                                      
"""
)


def typewriter_effect(text, delay=0.03):
    for char in text:
        print(char, end="", flush=True)
        time.sleep(delay)


text = """        Bridging the real and virtual worlds
{:^50}
""".format(
    "[PROJECT D.I.S.H.A.]"
)

typewriter_effect(text)

# === USER ONBOARDING & MODE SELECTION ===
# Check if this is a first-time user
user_name = ""
interaction_mode = "voice"

if user_config.is_first_time_user():
    # Ask for user's name
    user_name = ask_user_name()
    user_config.set_user_name(user_name)
    
    # Select interaction mode
    interaction_mode = select_interaction_mode()
    user_config.set_interaction_mode(interaction_mode)
else:
    # Returning user - load saved preferences
    loaded_name = user_config.get_user_name()
    user_name = loaded_name if loaded_name else "Friend"  # Type-safe fallback
    interaction_mode = user_config.get_interaction_mode()
    
    # Time-based personalized greeting
    greeting = get_time_based_greeting()
    print(f"\n💙 {greeting}, {user_name}! It's wonderful to see you again!")
    print(f"Using {interaction_mode.upper()} mode.")
    print(f"(To change settings, delete user_config.json and restart)\n")

# Display ready message
if interaction_mode == "voice":
    print("\n🎤 Voice mode active - I'm listening for your voice...\n")
else:
    print(f"\n⌨️ Text mode active - Type your messages below, {user_name}!\n")


# Free TTS function using Microsoft Edge TTS (high-quality female voice)
async def generate_tts_async(text, voice="en-US-SaraNeural", output_file="out.wav"):
    """Generate speech using free Microsoft Edge TTS with natural, warm female voice
    
    Voice: SaraNeural - Warm, personal, and caring tone
    """
    try:
        # Clean text before processing
        clean_text = text.strip()
        
        # Add natural pauses with commas and periods for breathing
        # Add slight pause after empathetic phrases
        clean_text = clean_text.replace(". ", "... ")  # Longer pause between sentences
        clean_text = clean_text.replace(", ", ",, ")   # Natural pause at commas
        
        # Generate speech with optimized parameters for warmth and naturalness
        # SaraNeural: Warm, personal, caring - perfect for mental health support
        # Rate: Slightly slower for empathy and clarity
        # Pitch: +5% for feminine warmth without being too high
        # Volume: +5% for clear, confident presence
        communicate = edge_tts.Communicate(
            clean_text, 
            voice,
            rate="-8%",      # Slower for empathy and understanding
            pitch="+5Hz",    # Slightly higher for warmth
            volume="+5%"     # Clear but not overwhelming
        )
        await communicate.save(output_file)
        return output_file
    except Exception as e:
        print(f"Error generating speech with Edge TTS: {e}")
        # Fallback to basic version without modifications
        try:
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(output_file)
            return output_file
        except:
            return None


def voice_edge_tts(text, voice="en-US-SaraNeural"):
    """Synchronous wrapper for Edge TTS (free, high-quality warm female voice)
    
    Best feminine voices for mental health support:
    - en-US-SaraNeural (CURRENT - warm, personal, caring)
    - en-US-ZiraNeural (natural, warm, empathetic)
    - en-US-AriaNeural (warm, friendly, conversational)
    - en-US-JennyNeural (warm, natural, professional)
    - en-US-MichelleNeural (young, energetic, upbeat)
    - en-GB-SoniaNeural (British English, sophisticated)
    - en-AU-NatashaNeural (Australian, friendly)
    """
    try:
        path = f"{abs_path}/out.wav"
        asyncio.run(generate_tts_async(text, voice, path))
        # Audio generation is silent for clean UX
        return path
    except Exception as e:
        print(f"Error with Edge TTS: {e}")
        # Fallback to VITS if available
        return voice_vits_fallback(text)


# Function to detect and handle emojis in text
def process_emojis_and_generate_audio(text):
    """Detect emojis in text and replace with natural vocal expressions"""
    # emoji module is imported at top of file
    
    # Emoji mapping to natural expressions (no SSML - keep it simple)
    emoji_expressions = {
        # Laughing emojis - natural expressions
        '😂': '*laughs warmly*',
        '🤣': '*laughs heartily*',
        '😹': '*giggles*',
        '😆': '*chuckles*',
        
        # Crying emojis - natural expressions
        '😢': '*voice softens*',
        '😭': '*speaks gently*',
        '🥹': '*with tenderness*',
        
        # Other emotional emojis - natural expressions
        '😊': '*smiles warmly*',
        '😌': '*sighs contentedly*',
        '🤗': '*with warmth*',
        '💕': '*lovingly*',
        '✨': '', # Remove sparkle emoji silently
        '🌟': '', # Remove star emoji silently
        '💪': '*encouragingly*',
        '🙏': '*gratefully*',
        '💙': '', # Remove heart emoji silently for cleaner speech
        '🫂': '', # Remove hug emoji silently
    }
    
    # Extract all emojis from text
    try:
        emojis_found = [char for char in text if char in emoji.EMOJI_DATA]
    except AttributeError:
        # Fallback for older emoji library versions
        import emoji as emoji_lib
        emojis_found = [char for char in text if emoji_lib.is_emoji(char)]
    
    # Process each emoji
    processed_text = text
    
    for em in emojis_found:
        if em in emoji_expressions:
            expr = emoji_expressions[em]
            # Replace emoji with natural expression (or remove if empty string)
            processed_text = processed_text.replace(em, f" {expr} " if expr else " ")
        else:
            # For unknown emojis, just remove them
            processed_text = processed_text.replace(em, ' ')
    
    # Clean up extra spaces
    processed_text = ' '.join(processed_text.split())
    
    return processed_text


# Fallback VITS function (original implementation)
def voice_vits_fallback(text, id=0, format="wav", lang=lang_code, length=1, noise=0.667, noisew=0.8, max=50):
    """Fallback TTS using local VITS server"""
    fields = {
        "text": text,
        "id": str(id),
        "format": format,
        "lang": lang,
        "length": str(length),
        "noise": str(noise),
        "noisew": str(noisew),
        "max": str(max),
    }
    boundary = "----VoiceConversionFormBoundary" + "".join(
        random.sample(string.ascii_letters + string.digits, 16)
    )

    m = MultipartEncoder(fields=fields, boundary=boundary)
    headers = {"Content-Type": m.content_type}
    url = f"{base}/voice"

    try:
        res = requests.post(url=url, data=m, headers=headers)
        path = f"{abs_path}/out.wav"

        with open(path, "wb") as f:
            f.write(res.content)
        print(f"Audio saved to: {path}")
        return path
    except Exception as e:
        print(f"Error with VITS fallback: {e}")
        return None


# define function to check if user has said "bye", "goodbye", or "see you"
def check_goodbye(transcript):
    goodbye_words = ["bye", "goodbye", "see you"]
    for word in goodbye_words:
        if word in transcript.casefold():
            return True
    return False


def check_creator_question(transcript):
    """Detect if user is asking about DISHA's creator"""
    creator_patterns = [
        "who created you",
        "who made you",
        "who built you",
        "who developed you",
        "who designed you",
        "who is your creator",
        "who is your maker",
        "who is your developer",
        "created by",
        "made by",
        "your creator",
        "your maker",
        "your developer",
    ]
    
    transcript_lower = transcript.lower()
    for pattern in creator_patterns:
        if pattern in transcript_lower:
            return True
    return False


def get_creator_response():
    """Return the consistent creator response"""
    return "I was created by Samarth, a passionate AIML engineer and tech enthusiast."


def get_time_based_greeting():
    """Return time-appropriate greeting for more natural interaction"""
    current_hour = datetime.now().hour
    
    if 5 <= current_hour < 12:
        return "Good morning"
    elif 12 <= current_hour < 17:
        return "Good afternoon"
    elif 17 <= current_hour < 22:
        return "Good evening"
    else:
        return "Hey there"  # Late night/early morning


def add_natural_breathing_pauses(text):
    """Injects SSML <break> tags to simulate breathing, for more human-like speech."""
    # Add a longer pause after sentences (end of sentence punctuation).
    text = re.sub(r'([.!?])', r'\1<break time="600ms" />', text)
    # Add a shorter pause after commas, semicolons, and colons.
    text = re.sub(r'([,;:])', r'\1<break time="300ms" />', text)
    return text


def test_entailment(text1, text2):
    """Test semantic entailment between two texts (for app/device control)"""
    if tokenizer is None or model is None:
        return 0.0  # Return low score if model not loaded
    try:
        batch = tokenizer(text1, text2, return_tensors="pt").to(model.device)
        with torch.no_grad():
            proba = torch.softmax(model(**batch).logits, -1)
        return proba.cpu().numpy()[0, model.config.label2id["ENTAILMENT"]]
    except Exception:
        return 0.0


def test_equivalence(text1, text2):
    """Test semantic equivalence between two texts (for app/device control)"""
    if tokenizer is None or model is None:
        return 0.0  # Return low score if model not loaded
    return test_entailment(text1, text2) * test_entailment(text2, text1)

def replace_device(sentence, word):
    return sentence.replace("[device]", word)


def replace_app(sentence, word):
    return sentence.replace("[app]", word)


def keep_sentence_with_word(text, word):
    """Extract sentences containing the specified word from text"""
    sentences = re.split(r"([.,!?])", text)
    filtered_sentences = []
    i = 0
    while i < len(sentences) - 1:
        sentence = sentences[i]
        punct = sentences[i + 1] if i + 1 < len(sentences) else ""
        if word in sentence:
            filtered_sentences.append(sentence.strip() + punct)
        i += 2
    result = " ".join(filtered_sentences)
    return result


# === 3D AVATAR INITIALIZATION ===
disha_avatar = None
if AVATAR_AVAILABLE:
    try:
        print("\n" + "="*60)
        print("🌟 Launching DISHA's 3D Avatar...")
        print("="*60)
        disha_avatar = DISHAAvatar("c001_f_costume_kouma")
        disha_avatar.launch_window(auto_open=True)
        print("✅ Avatar window opened in browser!")
        print("   Real-time emotion sync and lip-sync enabled!")
        print("="*60 + "\n")
        time.sleep(2)  # Give browser time to open
    except Exception as e:
        print(f"⚠️  Avatar initialization failed: {e}")
        print("   Continuing without visual avatar...")
        disha_avatar = None

# === CONVERSATION LOOP MANAGEMENT ===
conversation_turn_count = 0
memory_optimization_interval = 20  # Optimize every 20 turns

while True:
    # Initialize trans variable
    trans: str = ""
    
    # Periodic memory optimization
    conversation_turn_count += 1
    if conversation_turn_count % memory_optimization_interval == 0:
        optimize_memory()
        prune_conversation_history(max_lines=500)
    
    # === DUAL MODE INPUT HANDLING ===
    if interaction_mode == "voice":
        # Voice mode - capture audio
        if mic is None:
            print("❌ Microphone not available! Switching to text mode.")
            interaction_mode = "text"
            user_config.set_interaction_mode("text")
            continue
        
        try:
            with mic as source:
                audio = r.listen(source, timeout=None)
        except Exception as e:
            print(f"❌ Error capturing audio: {e}")
            print("Switching to text mode...")
            interaction_mode = "text"
            user_config.set_interaction_mode("text")
            continue
    else:  # text mode
        # Get text input directly
        text_input = get_text_input(user_name)
        
        if not text_input:
            continue
        
        # Skip audio processing for text mode
        trans = text_input

    now = datetime.now()
    date = now.strftime("%m/%d/%Y")
    time_2 = now.strftime("%H:%M:%S")

    # Only process audio if in voice mode
    if interaction_mode == "voice":
        try:
            test_text = r.recognize_sphinx(audio)  # type: ignore
            if len(test_text) == 0:
                continue
        except sr.UnknownValueError:
            continue

        with open("temp.wav", "wb") as f:
            f.write(audio.get_wav_data())  # type: ignore

        # Try speech recognition with multiple fallbacks
        try:
            try:
                trans = r.recognize_google(audio)  # type: ignore
                if len(trans) == 0:
                    continue
            except sr.UnknownValueError:
                # Fallback to WhisperX if Google fails
                if not (hasattr(whisper_model, '__class__') and whisper_model.__class__.__name__ == '_FallbackWhisperModel'):
                    try:
                        whisper_audio = whisperx.load_audio(audio_file)
                        result = whisper_model.transcribe(whisper_audio, batch_size=batch_size)
                        trans = result["segments"][0]["text"]
                    except Exception:
                        trans = test_text
                else:
                    trans = test_text
            except sr.RequestError:
                trans = test_text
                
            if len(trans) == 0:
                continue
                
        except Exception:
            trans = test_text
            if len(trans) == 0:
                continue
    # If in text mode, trans is already set from earlier

    text = trans
    new_line = {"role": user_name, "date": date, "time": time_2, "content": text}

    print(f"{user_name}: {text}")
    with open(r"conversation.jsonl", "a", encoding="UTF-8") as c:
        c.write("\n" + json.dumps(new_line, ensure_ascii=False))

    devices = [os.getenv("DEVICE_1", "lamp"), os.getenv("DEVICE_2", "fan")]

    if tuya == True:
        sentence = "Activate [device]."
        input_sentence = trans.lower()
        for word in devices:
            if word in input_sentence:
                modified_sentence = replace_device(sentence, word)

                input_sentence = keep_sentence_with_word(input_sentence, word)
                input_sentence = input_sentence.translate(
                    str.maketrans("", "", string.punctuation)
                )
                similarity = test_equivalence(modified_sentence, input_sentence)
                if similarity >= 0.5:
                    openapi = TuyaOpenAPI(API_ENDPOINT, ACCESS_ID, ACCESS_KEY)
                    openapi.connect()
                    if word == os.getenv("DEVICE_1"):
                        commands = {"commands": [{"code": "switch_1", "value": True}]}
                        device_id = os.getenv("DEVICE_1_ID")
                        if device_id:
                            openapi.post(device_id, commands)
                    if word == os.getenv("DEVICE_2"):
                        commands = {"commands": [{"code": "switch_1", "value": True}]}
                        device_id = os.getenv("DEVICE_2_ID")
                        if device_id:
                            openapi.post(device_id, commands)
                elif similarity < 0.001:
                    openapi = TuyaOpenAPI(API_ENDPOINT, ACCESS_ID, ACCESS_KEY)
                    openapi.connect()
                    if word == os.getenv("DEVICE_1"):
                        commands = {"commands": [{"code": "switch_1", "value": False}]}
                        device_id = os.getenv("DEVICE_1_ID")
                        if device_id:
                            openapi.post(device_id, commands)
                    if word == os.getenv("DEVICE_2"):
                        commands = {"commands": [{"code": "switch_1", "value": False}]}
                        device_id = os.getenv("DEVICE_2_ID")
                        if device_id:
                            openapi.post(device_id, commands)
    else:
        pass

    apps = [
        "youtube",
        "brave",
        "discord",
        "spotify",
        "explorer",
        "epic games launcher",
        "tower of fantasy",
        "steam",
        "minecraft",
        "clip studio paint",
        "premiere pro",
        "media encoder",
        "photoshop",
        "audacity",
        "obs",
        "vscode",
        "terminal",
        "synapse",
        "via",
    ]

    sentence = "Activate [app]."
    input_sentence = trans.lower()

    for word in apps:
        if word in input_sentence:
            modified_sentence = replace_app(sentence, word)
            input_sentence = keep_sentence_with_word(input_sentence, word)
            input_sentence = input_sentence.translate(
                str.maketrans("", "", string.punctuation)
            )
            similarity = test_equivalence(modified_sentence, input_sentence)
            if similarity >= 0.5:
                start(word, match_closest=True)
            elif similarity < 0.001:
                end(word, match_closest=True)
    else:
        pass

    query = f"""{text}"""

    results = memory.search(query, top_n=2)

    extracted_dicts = [result["chunk"] for result in results]

    line1 = str(extracted_dicts[0])
    line2 = str(extracted_dicts[1])

    # Read the file and store the lines in the list
    with open("conversation.jsonl", "r", encoding="UTF-8") as file:
        lines = file.readlines()

    lines = [line.strip() for line in lines]
    lines = "\n".join(lines)
    lines = lines.splitlines()

    # Check if there are at least 5 lines in the file (6 lines to read and 1 line to exclude)
    if len(lines) >= 5:
        # Extract the last 5 lines (excluding the last line) into a string
        last_six_lines = "\n".join(lines[-5:-1])

        # Iterate over the lines to check
        if line1 not in last_six_lines:
            # If not found in last_six_lines, search for it in the entire file
            found = False
            for i, line in enumerate(lines):
                if line == line1:
                    # If found, append line_to_check and the line directly after/before it to the top of last_six_lines
                    found = True
                    if '''{"role": "User"''' in line:
                        last_six_lines = (
                            line1 + "\n" + str(lines[i + 1]) + "\n" + last_six_lines
                        )
                        break
                    elif '''{"role": "DISHA"''' in line:
                        last_six_lines = (
                            str(lines[i - 1]) + "\n" + line1 + "\n" + last_six_lines
                        )
                        break
            if not found:
                # If still not found, append only line_to_check at the top without the line directly after it
                last_six_lines = line1 + "\n" + last_six_lines
        if line2 not in last_six_lines:
            # If not found in last_six_lines, search for it in the entire file
            found = False
            for i, line in enumerate(lines):
                if line == line2:
                    # If found, append line_to_check and the line directly after/before it to the top of last_six_lines
                    found = True
                    if '''{"role": "User"''' in line:
                        last_six_lines = (
                            line2 + "\n" + str(lines[i + 1]) + "\n" + last_six_lines
                        )
                        break
                    elif '''{"role": "DISHA"''' in line:
                        last_six_lines = (
                            str(lines[i - 1]) + "\n" + line2 + "\n" + last_six_lines
                        )
                        break
            if not found:
                # If still not found, append only line_to_check at the top without the line directly after it
                last_six_lines = line2 + "\n" + last_six_lines
    else:
        last_six_lines = lines

    try:
        memory.save(f"""["{new_line}"]""")
    except Exception:
        pass  # Silent failure

    now = datetime.now()
    date = now.strftime("%m/%d/%Y")
    time_1 = now.strftime("%H:%M:%S")

    # Create personalized system prompt
    personalized_lore = lore.replace("{YOUR_NAME}", user_name).replace("Sam", user_name)

    prompt = (
        personalized_lore
        + "\n\n"
        + str(last_six_lines)
        + str(new_line)
        + f'\n{{"role": "DISHA", "date": "{date}", "time": "{time_1}", "content": "'
    )
    prompt = str(prompt)

    # Check if user is asking about creator
    if check_creator_question(text):
        print("DISHA: ", end="")
        response = get_creator_response()
        # Print response immediately
        print(response)
        print()
        
        # Save to conversation history
        new_line = {
            "role": "DISHA",
            "date": date,
            "time": time_1,
            "content": response,
        }
        with open(r"conversation.jsonl", "a", encoding="UTF-8") as c:
            c.write("\n" + json.dumps(new_line, ensure_ascii=False))
        try:
            memory.save(f'''["{new_line}"]''')
        except Exception:
            pass  # Silent failure
        
        # Generate and play TTS in both modes
        voice_edge_tts(text=response)
        filename = "out.wav"
        data, fs = sf.read(filename, dtype="float32")
        
        # Audio device configuration based on mode
        if interaction_mode == "voice":
            audio_device = os.getenv("AUDIO_OUTPUT_DEVICE", "")
            if audio_device and audio_device.strip():
                try:
                    sd.default.device = audio_device  # type: ignore
                except Exception:
                    sd.default.device = None  # type: ignore
            else:
                sd.default.device = None  # type: ignore
        else:
            # Text mode: Always use default speakers
            sd.default.device = None  # type: ignore
        
        sd.play(data, fs)
        status = sd.wait()
        
        if check_goodbye(trans):
            break
        else:
            continue

    # generate a response using Mental Health Chatbot
    print("DISHA: ", end="")
    
    # Emotion hotkey mapping for VTube Studio gestures
    # Maps emotional expressions to keyboard shortcuts
    emotion_hotkey_map = {
        "wave": "6",
        "thumbs-up": "7",
        "thumbs up": "7",
        "nodding": "8",
        "nod": "8",
        "shaking head": "9",
        "shake head": "9",
        "clap": "0",
        "clapping": "0",
    }
    
    # === EMOTIONAL INTELLIGENCE INTEGRATION ===
    # Detect user emotion for context-aware response (silently)
    if emotion_engine:
        emotion_result = emotion_engine.detect_emotion(text)
        # Removed debug print - emotion detection is silent
        
        # Sync avatar emotion with detected user emotion
        if disha_avatar:
            detected_emotion = emotion_result['primary_emotion'].lower()
            disha_avatar.set_emotion(detected_emotion)
        
        # Check for crisis situation
        is_crisis, severity = emotion_engine.detect_crisis_indicators(text)
        
        if is_crisis:
            print(f"\n[⚠️ Crisis support activated]\n")  # Important warning only
        
        # Get emotional tone guidance
        tone_instruction = emotion_engine.get_emotional_tone_instructions(emotion_result['primary_emotion'])
        
        # Calculate user engagement for dynamic response length
        try:
            with open("conversation.jsonl", "r", encoding="utf-8") as f:
                recent_lines = f.readlines()[-10:]  # Last 10 messages
                recent_messages = [json.loads(line) for line in recent_lines if line.strip()]
        except:
            recent_messages = []
        
        engagement_score = get_user_engagement_score(recent_messages)
        optimal_length = calculate_dynamic_response_length(engagement_score)
        
        # Enhance prompt with emotional context
        emotional_context = f"""\n\n[USER EMOTIONAL STATE: {emotion_result['primary_emotion']} - confidence {emotion_result['confidence']:.0%}]
[RESPONSE TONE: {tone_instruction}]
[RESPONSE LENGTH: Keep response to approximately {optimal_length} tokens for optimal engagement]
"""
        
        # Add crisis guidance if needed
        if is_crisis:
            emotional_context += f"[CRISIS LEVEL: {severity} - Prioritize safety and professional help resources]\n"
        
        enhanced_prompt = prompt + emotional_context
    else:
        enhanced_prompt = prompt
        is_crisis = False
        severity = 'none'
        emotion_result = {'primary_emotion': 'neutral', 'confidence': 0.5}
    
    # Generate faster response with Mental Health Chatbot model
    try:
        base_response = generate_mental_health_response(enhanced_prompt)
        
        # Ensure positive, hopeful tone
        base_response = base_response.replace("can't", "can").replace("don't", "do")
        base_response = re.sub(r'(?i)\b(never|impossible|hard|difficult)\b', 'challenging but possible', base_response)
        
        # Add crisis response if needed, always maintaining hope
        if is_crisis and emotion_engine:
            crisis_prefix = emotion_engine.get_crisis_response_prefix(severity)
            base_response = crisis_prefix + "\n\nBut remember, there is always hope. " + base_response
    except Exception:
        # Fallback response on error
        base_response = "I'm here with you. How are you feeling right now?"
    
    # Enhance response with empathy and warmth
    if emotion_engine:
        response = emotion_engine.enhance_response_with_emotion(
            base_response,
            emotion_result['primary_emotion'],
            emotion_result['confidence']
        )
        
        # Add conversational warmth
        response = emotion_engine.add_conversational_warmth(response)
        
        # Add personality expressiveness
        if personality_engine:
            response = personality_engine.add_emotional_expressiveness(
                response,
                emotion_result['primary_emotion']
            )
            
            # Check if this is first interaction (for relationship building)
            try:
                is_first_interaction = os.path.getsize("conversation.jsonl") < 500
            except:
                is_first_interaction = False
            
            response = personality_engine.adjust_for_relationship_building(response, is_first_interaction)
    else:
        response = base_response
    
    # Print response immediately (no typewriter delay for faster experience)
    print(response)
    print()  # Add newline

    new_line = {
        "role": "DISHA",
        "date": date,
        "time": time_1,
        "content": response,
    }

    with open(r"conversation.jsonl", "a", encoding="UTF-8") as c:
        c.write("\n" + json.dumps(new_line, ensure_ascii=False))

    try:
        memory.save(f"""["{new_line}"]""")
    except Exception:
        pass  # Silent failure

    # Trigger VTube Studio gestures (voice mode only)
    if interaction_mode == "voice":
        for emotion, hotkey in emotion_hotkey_map.items():
            # Look for the emotion word in the response
            if emotion in response.lower():
                try:
                    # Remove the emotion marker from spoken text
                    response = re.sub(r'\b' + re.escape(emotion) + r'\b', "", response, flags=re.IGNORECASE)
                    # Trigger the hotkey for VTube Studio
                    pyautogui.hotkey("ctrl", "alt", hotkey)
                    # Removed debug print
                except Exception:
                    pass  # Silent failure
                break

    # Process emojis and convert them to natural vocal expressions
    processed_response = process_emojis_and_generate_audio(response)
    
    # Add natural breathing pauses for more human-like speech
    processed_response = add_natural_breathing_pauses(processed_response)
    
    # Signal avatar that DISHA is about to speak
    if disha_avatar:
        disha_avatar.start_speaking()
    
    # Generate and play TTS in BOTH voice and text modes
    # Generate TTS with processed text (emojis converted to expressions)
    voice_edge_tts(text=processed_response)

    filename = "out.wav"
    # Extract data and sampling rate from file
    data, fs = sf.read(filename, dtype="float32")
    
    # Audio device configuration - silent setup
    if interaction_mode == "voice":
        # Voice mode: Respect AUDIO_OUTPUT_DEVICE setting (may be VB-Cable)
        audio_device = os.getenv("AUDIO_OUTPUT_DEVICE", "")
        
        if audio_device and audio_device.strip():
            try:
                sd.default.device = audio_device  # type: ignore
            except Exception:
                sd.default.device = None  # type: ignore
        else:
            sd.default.device = None  # type: ignore
    else:
        # Text mode: Always use default device (speakers)
        sd.default.device = None  # type: ignore

    sd.play(data, fs)
    status = sd.wait()  # Wait until file is done playing
    
    # Signal avatar that DISHA stopped speaking
    if disha_avatar:
        disha_avatar.stop_speaking()
        disha_avatar.set_emotion('neutral')  # Return to neutral

    if check_goodbye(trans):
        break
    else:
        continue
