# 🌟 DISHA - Digital Interactive Support & Health Assistant

## Overview
DISHA is an empathetic AI mental health companion designed to provide emotional support through natural conversation with a warm, human-like female voice.

## ✨ Recent Enhancements (Human-Like Experience)

### 🎤 Voice Improvements
- **Changed to AriaNeural**: Warmer, more conversational female voice
- **Speech Parameters**: 
  - Rate: -8% (slower for empathy)
  - Pitch: +5Hz (feminine warmth)
  - Volume: +5% (clear presence)
- **Natural Pauses**: Automatic breathing pauses at commas and periods
- **Conversational Flow**: Added natural fillers ("hmm", "well", "you know")

### 💬 Response Quality
- **More Natural**: Increased max_new_tokens to 150 for fuller responses
- **Better Variation**: Higher temperature (0.82) for human-like diversity
- **Reduced Repetition**: Stronger penalties for more natural conversation
- **Conversational Style**: Uses contractions, speaks like a real friend

### 🕐 Personalization
- **Time-Based Greetings**: "Good morning", "Good afternoon", "Good evening"
- **Warm Welcome Back**: Personalized returning user messages
- **Natural Expressions**: Emoji processing for natural vocal cues

## 📦 Installation

### Prerequisites
```bash
# Install Python 3.8 or higher
python --version

# Install required packages
pip install -r requirements.txt

# Install Groq for ultra-fast responses
pip install groq
```

### Get FREE Groq API Key (Required for Fast Responses)
1. Visit: https://console.groq.com/keys
2. Sign up (free, 2 minutes)
3. Create API key
4. Add to `.env` file:
```env
GROQ_API_KEY=your_key_here
```

**See `GROQ_SETUP.md` for detailed setup instructions!**

### Optional Dependencies
```bash
# For smart home control (Tuya devices)
pip install tuya-connector-python

# For advanced memory features
pip install vectordb
```

## 🚀 Usage

### First Time Setup
1. Run the application:
   ```bash
   python DISHAMemory.py
   ```
2. Enter your name when prompted
3. Choose interaction mode (Voice or Text)

### Voice Mode
- Speak naturally to DISHA
- She'll listen and respond with empathy
- Supports natural conversation flow

### Text Mode
- Type your messages
- Perfect for quiet environments
- Full TTS support with emotional voice

## 🎯 Key Features

### ⚡ Ultra-Fast Responses (NEW!)
- **Groq API Integration**: 10-20x faster than local models
- **Response Time**: 0.3-0.8 seconds (feels instant!)
- **Llama-3.1-8B**: Advanced emotional intelligence
- **Free Tier**: 30 requests/minute (perfect for personal use)

### Emotional Intelligence
- Detects 7 emotions: happy, sad, angry, fear, surprise, disgust, neutral
- Crisis detection with appropriate support resources
- Tone adaptation based on emotional state

### Memory System
- Remembers conversation context
- Retrieves relevant past interactions
- Smart pruning for performance

### Natural Voice
- Microsoft Edge TTS (AriaNeural)
- Warm, empathetic female voice
- Natural breathing and pauses
- Emotional expressiveness

## 🔧 Configuration

### Environment Variables (.env)
```env
# Groq API (REQUIRED for fast responses)
GROQ_API_KEY=your_groq_api_key_here

# Language
LANGUAGE=en

# AI Model
WHISPERX_MODEL=small

# System Prompt
LORE=You are DISHA, a compassionate AI mental health support assistant.

# Audio Output (optional - for voice mode routing)
AUDIO_OUTPUT_DEVICE=

# Smart Home (optional)
TUYA_ID=
TUYA_SECRET=
TUYA_ENDPOINT=https://openapi.tuyaus.com
DEVICE_1=lamp
DEVICE_1_ID=
DEVICE_2=fan
DEVICE_2_ID=
```

### Changing Voice
Edit `DISHAMemory.py` line ~576:
```python
async def generate_tts_async(text, voice="en-US-AriaNeural", output_file="out.wav"):
```

**Recommended Female Voices:**
- `en-US-AriaNeural` ⭐ (Default - warm, conversational)
- `en-US-JennyNeural` (natural, professional)
- `en-US-SaraNeural` (warm, personal)
- `en-US-MichelleNeural` (young, energetic)

## 🐛 Known Issues & Fixes

### Issue 1: Missing Dependencies
**Problem**: `tuya_connector` or `vectordb` not found
**Fix**: These are optional - code has fallbacks built-in

### Issue 2: CUDA Warnings
**Fix**: Already suppressed in code - safe to ignore

### Issue 3: Audio Device Not Found
**Fix**: Remove or clear `AUDIO_OUTPUT_DEVICE` in .env file

## 🎨 Customization

### Make Voice More Expressive
Adjust in `generate_tts_async()`:
```python
rate="-10%",    # Slower (more expressive)
pitch="+8Hz",   # Higher (more feminine)
```

### Adjust Response Length
Edit in `generate_mental_health_response()`:
```python
max_new_tokens=200,  # Longer responses
```

### Change Personality
Edit `mental_health_system` prompt in DISHAMemory.py (line ~227)

## 📊 Performance

- **Response Time**: 0.3-0.8 seconds (10-20x faster with Groq!)
- **Memory Usage**: ~500MB-1GB (much lighter than before)
- **Audio Quality**: 24kHz, 16-bit (high quality)
- **Internet Required**: Yes (for Groq API - but worth it for speed!)

## 🤝 Contributing

Created by **Samarth** - AIML Engineer

## 📝 License

Private project - All rights reserved

## 🆘 Crisis Resources

DISHA provides crisis support resources when needed:
- **988**: Suicide & Crisis Lifeline (US)
- **741741**: Crisis Text Line (text HOME)

---

**Note**: DISHA is a supportive companion but not a replacement for professional mental health care. Always seek professional help for serious concerns.
