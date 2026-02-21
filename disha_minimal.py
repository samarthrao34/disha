"""
Minimal DISHA - Multi-Language Text Mode with Google Gemini Flash + 3D Avatar
Ultra-fast, free (1,500 requests/day), supports all Indian + international languages
"""
import os
from dotenv import load_dotenv
import google.generativeai as genai
from datetime import datetime
import edge_tts
import asyncio
import sounddevice as sd
import soundfile as sf
import re
from language_handler import language_handler
import time

# Avatar support
try:
    from disha_avatar import DISHAAvatar
    AVATAR_AVAILABLE = True
except:
    AVATAR_AVAILABLE = False

# Load environment
load_dotenv()

# Initialize Gemini
api_key = os.getenv("GEMINI_API_KEY", "")
if not api_key:
    print("❌ GEMINI_API_KEY not found in .env file!")
    print("Get your FREE API key from: https://makersuite.google.com/app/apikey")
    exit(1)

genai.configure(api_key=api_key)
model = genai.GenerativeModel('models/gemini-2.5-flash-lite-preview-09-2025')

# Initialize Avatar
disha_avatar = None
if AVATAR_AVAILABLE:
    try:
        print("\n🌟 Launching DISHA's 3D Avatar...")
        disha_avatar = DISHAAvatar("c001_f_costume_kouma")
        disha_avatar.launch_window(auto_open=True)
        print("✅ Avatar launched! Check your browser.")
        time.sleep(2)  # Give browser time to open
    except Exception as e:
        print(f"⚠️ Avatar error: {e}")
        disha_avatar = None

print("\n" + "="*60)
print(" "*5 + "🌟 DISHA - Multi-Language AI Companion 🌟")
print("="*60)
print("\n✨ Features:")
print("  • 3D Live2D Avatar with real-time emotions")
print("  • Multi-language support (30+ languages)")
print("  • Emotion-synced animations")
print("  • Natural voice synthesis")
print("\nSupported: Hindi, Bengali, Tamil, Telugu, Marathi, Gujarati,")
print("          Kannada, Malayalam, Punjabi + 20+ international languages")
print("\nJust type in your language - DISHA will understand!")
print("Type 'bye' to exit.\n")

# Simple TTS function with multi-language support
async def speak_async(text, voice="en-US-AriaNeural"):
    try:
        # Clean the text - remove any SSML tags or special characters that shouldn't be spoken
        clean_text = text.strip()
        
        communicate = edge_tts.Communicate(
            clean_text,
            voice,
            rate="-15%",  # Slower for warmth and care
            pitch="-3Hz"   # Lower pitch for mature, warm tone
        )
        await communicate.save("out.wav")
        data, fs = sf.read("out.wav", dtype="float32")
        sd.play(data, fs)
        sd.wait()
    except:
        pass  # Silent errors

def speak(text, lang_code='en'):
    try:
        # Get appropriate voice for the language
        voice = language_handler.get_voice_for_language(lang_code)
        asyncio.run(speak_async(text, voice))
    except:
        pass  # Silent audio errors

def add_natural_breathing_pauses(text):
    """Injects natural pauses for more human-like speech."""
    # For edge-tts, we'll use natural sentence breaks without SSML
    # The TTS engine handles pauses naturally at punctuation
    return text.strip()

# Simple emotion detection from text
def detect_emotion_simple(text):
    """Simple keyword-based emotion detection"""
    text_lower = text.lower()
    
    # Happy keywords
    if any(word in text_lower for word in ['happy', 'joy', 'great', 'wonderful', 'excited', 'खुश', 'आनंद']):
        return 'happy'
    
    # Sad keywords
    if any(word in text_lower for word in ['sad', 'depressed', 'upset', 'crying', 'hurt', 'उदास', 'दुखी']):
        return 'sad'
    
    # Angry keywords
    if any(word in text_lower for word in ['angry', 'mad', 'furious', 'annoyed', 'frustrated', 'गुस्सा', 'क्रोध']):
        return 'angry'
    
    # Surprised keywords
    if any(word in text_lower for word in ['surprised', 'shocked', 'wow', 'amazing', 'unbelievable', 'आश्चर्य']):
        return 'surprise'
    
    # Fear/Anxiety keywords
    if any(word in text_lower for word in ['afraid', 'scared', 'anxious', 'worried', 'nervous', 'डर', 'चिंता']):
        return 'fear'
    
    return 'neutral'

# Conversation loop
user_name = "Friend"
conversation_history = []

while True:
    try:
        # Get user input
        user_input = input(f"\n{user_name}: ").strip()
        
        if not user_input:
            continue
            
        if user_input.lower() in ['bye', 'goodbye', 'quit', 'exit']:
            farewell = "Take care! I'm always here if you need me. 💙"
            print(f"DISHA: {farewell}")
            speak(farewell, language_handler.current_language)
            break
        
        # STEP 1: Detect language and translate to English for AI
        text_for_ai, detected_lang = language_handler.process_input(user_input)
        
        # Detect emotion and update avatar (silently)
        if disha_avatar:
            try:
                emotion = detect_emotion_simple(user_input)
                disha_avatar.set_emotion(emotion)
            except:
                pass
        
        # Add to history (in English for AI)
        conversation_history.append({
            "role": "user",
            "content": text_for_ai  # Use translated English text
        })
        
        # Keep only last 10 messages for context
        if len(conversation_history) > 10:
            conversation_history = conversation_history[-10:]
        
        # Create conversation context
        context = "\n".join([f"{msg['role'].title()}: {msg['content']}" for msg in conversation_history])
        
        # System prompt
        system_prompt = """You are DISHA, a warm and caring mental health companion and counselor.

Keep responses brief (2-4 sentences) and deeply empathetic.
Always validate emotions first, then offer hope and understanding.
Be conversational, warm, and human - never robotic or clinical.
Make people feel heard, understood, and lighter after talking to you."""
        
        print("DISHA: ", end="", flush=True)
        
        # STEP 2: Generate response with Gemini Flash (in English)
        response = model.generate_content(
            f"{system_prompt}\n\n{context}",
            generation_config=genai.types.GenerationConfig(
                temperature=0.9,
                max_output_tokens=200,
                top_p=0.95
            )
        )
        
        reply_english = response.text.strip()
        
        # STEP 3: Translate response back to user's language
        reply = language_handler.process_output(reply_english, detected_lang)
        
        print(reply)
        
        # Add to history
        conversation_history.append({
            "role": "assistant",
            "content": reply_english  # Store English version for AI context
        })
        
        # STEP 4: Signal avatar speaking and speak the response in user's language
        if disha_avatar:
            try:
                disha_avatar.start_speaking()
            except:
                pass
        
        # Speak only the reply text, nothing else
        speak(add_natural_breathing_pauses(reply), detected_lang)
        
        if disha_avatar:
            try:
                disha_avatar.stop_speaking()
                disha_avatar.set_emotion('neutral')
            except:
                pass
        
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye! Take care!")
        break
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Let's try again...")

print("\n" + "="*60)
print("Thanks for talking with me! 💙")
print("="*60 + "\n")
