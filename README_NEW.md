# 🌟 DISHA - Digital Interactive Support & Health Assistant

**AI-Powered Mental Health Companion with 3D Live2D Avatar**

DISHA is an emotionally intelligent AI assistant designed to provide mental health support, counseling, and genuine companionship. She features a beautiful **Live2D 3D avatar** with emotion-synced animations and the warm voice of a caring friend.

---

## ✨ Key Features

### 🎭 **3D Live2D Avatar** 
- **Visual presence** - Beautiful anime-style character
- **Real-time emotion animations** - 7+ emotional expressions (happy, sad, angry, surprised, etc.)
- **Speaking animations** - Lip-sync and natural gestures
- **Web-based viewer** - Opens automatically in your browser
- **Desktop support** - Also works as a standalone window
- 📖 See `AVATAR_GUIDE.md` for full details

### 🧠 **Emotional Intelligence**
- Detects user emotions from text/voice input
- Provides empathetic, context-aware responses
- Crisis detection and specialized support
- Natural conversational warmth
- Avoids robotic or clinical language

### 🚀 **Powered by Google Gemini Flash**
- **Ultra-fast responses** - Sub-second reply times
- **1,500 free requests/day** - Perfect for startups
- **Fine-tunable** - Train on counseling data
- **Commercial-ready** - Build and sell your product
- **Scalable** - Grows with your user base
- 📖 See `GEMINI_SETUP.md` for setup

### 🎤 **Dual Interaction Modes**
- **Voice Mode** - Natural speech recognition + TTS
- **Text Mode** - Fast keyboard input
- **Sara Voice** - Warm, personal, caring female voice
- **Emotion-matched tone** - Adapts to user's emotional state

### 🎯 **Mental Health Features**
- Emotional support and validation
- Active listening and reflection
- Crisis intervention capabilities
- Conversation memory and context
- Non-judgmental responses

---

## 🚀 Quick Start

### 1. Get Your FREE Gemini API Key

```
1. Go to: https://makersuite.google.com/app/apikey
2. Click "Create API Key"
3. Copy your key
```

### 2. Setup

```powershell
# Clone/download DISHA
cd d:\DISHA

# Install dependencies
pip install -r requirements.txt

# Add API key to .env file
# GEMINI_API_KEY=your_key_here
```

### 3. Run DISHA

**Option A: Minimal Text Version (Fastest)**
```powershell
python disha_minimal.py
```

**Option B: Full Version with Avatar**
```powershell
python DISHAMemory.py
```

**Option C: Easy Launcher**
```powershell
.\START_DISHA.ps1
```

---

## 📁 Project Structure

```
d:\DISHA\
├── DISHAMemory.py          # Full version with all features
├── disha_minimal.py        # Minimal text-only version
├── disha_avatar.py         # 3D avatar controller
├── emotion_engine.py       # Emotion detection & response
├── user_config.py          # User preferences
├── .env                    # API keys and configuration
├── requirements.txt        # Python dependencies
├── c001_f_costume_kouma/   # Live2D 3D model files
│   ├── *.moc3             # Model data
│   ├── *.model3.json      # Configuration
│   ├── motions/           # Emotion animations
│   └── textures/          # Character textures
└── models/
    └── emotion_resnet18.pt # Trained emotion detector
```

---

## 🎨 How It Works

```
1. User speaks/types → DISHA listens
                ↓
2. Emotion detection → Detects user's emotional state
                ↓
3. Avatar sync → Avatar shows matching emotion
                ↓
4. AI response → Gemini generates empathetic reply
                ↓
5. TTS + Animation → DISHA speaks with lip-sync
                ↓
6. Back to neutral → Avatar returns to idle state
```

---

## 💡 Use Cases

### 🏥 **Mental Health Support**
- Daily emotional check-ins
- Anxiety and stress relief
- Depression support
- Crisis intervention
- Therapeutic conversations

### 💼 **Startup Product**
- B2C mental health app
- Employee wellness platform
- Telehealth companion
- Educational counselor
- Senior care assistant

### 🎮 **VTuber / Streaming**
- Real-time emotion-responsive character
- Interactive AI companion
- Live streaming assistant
- Content creation tool

### 🧪 **Research & Development**
- Emotion AI testing
- Conversational AI research
- Human-AI interaction studies
- Fine-tuning experiments

---

## 🛠️ Configuration

### .env File Settings

```env
# Required
GEMINI_API_KEY=your_api_key_here

# Optional
LANGUAGE=en
WHISPERX_MODEL=small
DISABLE_NLI_MODEL=true
```

### user_config.json

```json
{
  "user_name": "Friend",
  "interaction_mode": "text"
}
```

---

## 📚 Documentation

- **`GEMINI_SETUP.md`** - Complete Gemini API setup guide
- **`GEMINI_MIGRATION.md`** - Migration from Groq to Gemini
- **`AVATAR_GUIDE.md`** - 3D avatar system documentation
- **`README.md`** - This file

---

## 🔧 System Requirements

### Minimum
- **OS**: Windows 10/11, Linux, macOS
- **Python**: 3.8 or higher
- **RAM**: 4GB
- **Internet**: Stable connection for API calls

### Recommended
- **RAM**: 8GB+ (for full version with voice)
- **GPU**: Optional (for faster emotion detection)
- **Microphone**: For voice mode
- **Speakers**: For TTS output

---

## 📦 Dependencies

### Core
- `google-generativeai` - Gemini Flash API
- `edge-tts` - Microsoft TTS voices
- `pygame` - Avatar visualization

### Voice Features
- `whisperx` - Speech recognition
- `sounddevice` - Audio I/O
- `SpeechRecognition` - Fallback STT

### AI/ML
- `torch` - PyTorch for emotion detection
- `transformers` - Optional NLI model

See `requirements.txt` for complete list.

---

## 🌟 Features in Detail

### Emotional Intelligence
- **7 Emotions**: Happy, Sad, Angry, Surprised, Fear, Disgust, Neutral
- **Crisis Detection**: Identifies severe distress
- **Context Awareness**: Remembers conversation history
- **Adaptive Tone**: Matches user's emotional state

### Voice Quality
- **Sara Voice**: `en-US-SaraNeural` - warm, personal, caring
- **Optimized Parameters**: Slower rate, higher pitch for empathy
- **Natural Pauses**: Breathing pauses for human-like speech
- **Emoji Processing**: Converts emojis to vocal expressions

### Avatar Animations
- **Emotion Sync**: Animations match detected emotions
- **Lip-Sync**: Speaking animations during TTS
- **Idle Behavior**: Natural movements when waiting
- **Smooth Transitions**: Fluid emotion changes

---

## 🚀 For Startups

### Why DISHA is Perfect for Your Startup

✅ **Free to Start** - 1,500 requests/day at zero cost  
✅ **Scalable** - Pay-as-you-grow pricing  
✅ **Customizable** - Fine-tune on your data  
✅ **Commercial License** - Sell as a product  
✅ **Complete Solution** - Voice, text, avatar, emotions  
✅ **Professional Quality** - Google's best AI  

### Business Model Ideas

1. **Subscription App** - $9.99/month for unlimited access
2. **B2B Platform** - $49/employee/month for companies
3. **Freemium** - Free basic, $19.99/month premium
4. **One-Time License** - $499 perpetual license
5. **White Label** - $999+ for custom branding

### Competitive Advantages

- **Visual Avatar** - Most competitors are text-only
- **Emotion AI** - Advanced emotional intelligence
- **Fine-Tunable** - Create unique personality
- **Low Cost** - Cheaper than hiring therapists
- **24/7 Availability** - Always there for users

---

## 🎯 Roadmap

### Current Version
- ✅ Google Gemini Flash integration
- ✅ 3D Live2D avatar with emotions
- ✅ Voice and text modes
- ✅ Emotion detection and sync
- ✅ Crisis intervention
- ✅ Conversation memory

### Upcoming Features
- [ ] Fine-tuned DISHA model on counseling data
- [ ] Multiple avatar styles
- [ ] Advanced lip-sync
- [ ] OBS Studio integration
- [ ] Mobile app (iOS/Android)
- [ ] Multi-language support
- [ ] User analytics dashboard

---

## 🐛 Troubleshooting

### DISHA won't start
- Check `GEMINI_API_KEY` in `.env` file
- Run `pip install -r requirements.txt`
- Try `python disha_minimal.py` first

### Avatar doesn't show
- Avatar will open in web browser automatically
- If not, check `disha_avatar_viewer.html`
- Ensure `pygame` is installed

### Slow responses
- Check internet connection
- Verify Gemini API key is valid
- Use `disha_minimal.py` for fastest experience

### No voice output
- Check speakers are connected
- Ensure `edge-tts` is installed
- Try text mode first

---

## 📄 License

This project is for educational and research purposes. The Live2D model is subject to its original license terms.

---

## 🙏 Acknowledgments

- **Google Gemini Flash** - Ultra-fast AI brain
- **Microsoft Edge TTS** - Natural voice synthesis
- **Live2D** - Beautiful 3D avatar technology
- **PyTorch** - Emotion detection framework

---

## 📞 Support

For setup help, see documentation:
- `GEMINI_SETUP.md` - API setup
- `AVATAR_GUIDE.md` - Avatar system
- `GEMINI_MIGRATION.md` - Migration guide

---

**Made with 💙 by passionate AIML engineers**

*DISHA - Your AI companion who truly cares*
