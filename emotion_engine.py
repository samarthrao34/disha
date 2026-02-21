import torch
import torch.nn as nn
import torchvision.models as models
import json
import os
import re

def create_emotionally_intelligent_system_prompt(base_lore):
    """Enhance base system prompt with emotional intelligence"""
    emotional_enhancement = """

EMOTIONAL INTELLIGENCE CORE:
- Recognize and validate emotions before giving advice
- Mirror emotional tone appropriately (gentle for sadness, calm for anxiety)
- Use empathetic phrases naturally throughout conversation
- Maintain unwavering positivity while respecting their feelings
- Build genuine connection through active listening cues
- Adapt response style to emotional state detected

RESPONSE PRINCIPLES:
- Start with emotional validation
- Share hope and encouragement naturally
- End with supportive engagement
- Be warm, human, and authentically caring
- Never sound clinical or robotic"""
    
    return base_lore + emotional_enhancement

class EmotionEngine:
    def __init__(self):
        try:
            self.model = models.resnet18(pretrained=False)
            num_classes = 7  # Based on emotion_labels.json
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
            
            # Load model weights with error handling
            model_path = os.path.join(os.path.dirname(__file__), 'models', 'emotion_resnet18.pt')
            if os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
                self.model.eval()
            else:
                # Fallback to rule-based only
                self.model = None
        except Exception:
            self.model = None
        
        # Load emotion labels
        try:
            with open(os.path.join(os.path.dirname(__file__), 'models', 'emotion_labels.json'), 'r') as f:
                self.emotion_labels = json.load(f)
        except Exception:
            self.emotion_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
        
        # Crisis detection patterns
        self.crisis_patterns = {
            'severe': ['suicide', 'kill', 'die', 'end it', 'hurt myself'],
            'moderate': ['hopeless', 'worthless', 'can\'t go on', 'giving up'],
            'mild': ['depressed', 'anxious', 'worried', 'scared']
        }
        
    def detect_emotion(self, text):
        """
        Detect emotion from text using combination of rule-based and model inference
        Returns dict with primary_emotion and confidence
        """
        # Map keywords to emotions for faster processing
        emotion_keywords = {
            'happy': ['joy', 'happy', 'excited', 'wonderful', 'great', 'amazing'],
            'sad': ['sad', 'depressed', 'unhappy', 'miserable', 'hurt'],
            'angry': ['angry', 'mad', 'frustrated', 'annoyed', 'upset'],
            'fear': ['scared', 'afraid', 'anxious', 'worried', 'terrified'],
            'surprise': ['shocked', 'surprised', 'amazed', 'unexpected'],
            'neutral': ['okay', 'fine', 'alright', 'normal'],
            'disgust': ['disgusted', 'gross', 'yuck', 'ew']
        }
        
        text = text.lower()
        emotion_scores = {emotion: 0 for emotion in self.emotion_labels}
        
        # Rule-based scoring
        for emotion, keywords in emotion_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    emotion_scores[emotion] += 1
        
        # Get highest scoring emotion
        primary_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
        total_matches = sum(emotion_scores.values())
        confidence = emotion_scores[primary_emotion] / max(total_matches, 1)
        
        return {
            'primary_emotion': primary_emotion,
            'confidence': confidence
        }
    
    def detect_crisis_indicators(self, text):
        """Check text for crisis indicators and return severity"""
        text = text.lower()
        
        for severity, patterns in self.crisis_patterns.items():
            if any(pattern in text for pattern in patterns):
                return True, severity
        
        return False, 'none'
    
    def get_crisis_response_prefix(self, severity):
        """Get appropriate crisis response based on severity"""
        responses = {
            'severe': "I'm very concerned about what you're saying and I want you to know that you're not alone. Please reach out to a crisis helpline right away - they are there 24/7 to help: 988. You can also text HOME to 741741.",
            'moderate': "I hear how difficult things are right now. Have you considered talking to a counselor or therapist? They can provide the professional support you deserve.",
            'mild': "I can tell you're going through a tough time. Remember that it's okay to ask for help, and there are people who want to support you."
        }
        return responses.get(severity, "")
    
    def get_emotional_tone_instructions(self, emotion):
        """Get tone guidance based on detected emotion"""
        tone_map = {
            'happy': "Match their joy with warm enthusiasm while keeping them grounded",
            'sad': "Be gentle and validating, offer hope without minimizing feelings",
            'angry': "Stay calm and understanding, validate feelings without amplifying anger",
            'fear': "Project calm confidence and reassurance, validate concerns",
            'surprise': "Share their sense of wonder while providing steady support",
            'disgust': "Acknowledge their feelings while redirecting to constructive views",
            'neutral': "Maintain warm engagement while exploring deeper feelings"
        }
        return tone_map.get(emotion, "Be warm and empathetic")
    
    def enhance_response_with_emotion(self, response, emotion, confidence):
        """Add appropriate emotional markers and tone based on detected emotion"""
        # Add subtle emotional indicators
        emotion_enhancers = {
            'happy': ['😊', '✨', '💕'],
            'sad': ['🫂', '💙', '✨'],
            'angry': ['I understand', 'that must be frustrating', 'let\'s work through this'],
            'fear': ['you\'re safe here', 'we\'ll face this together', '🫂'],
            'surprise': ['wow', 'I see why that would be surprising', '✨'],
            'disgust': ['that sounds challenging', 'let\'s focus on what helps', '💙'],
            'neutral': ['I\'m here', 'tell me more', '💫']
        }
        
        if confidence > 0.7:  # Only add enhancers if confident about emotion
            enhancers = emotion_enhancers.get(emotion, [])
            if enhancers and not any(e in response for e in enhancers):
                response = f"{response} {enhancers[0]}"
        
        return response
    
    def add_conversational_warmth(self, response):
        """Add warmth and natural conversation flow to responses"""
        import random
        
        # Add natural conversational elements with more variety
        warmth_phrases = {
            'start': ['You know,', 'I want you to know,', 'Here\'s the thing,', 'Listen,', 'Honestly,', 'Let me tell you,'],
            'middle': ['and that\'s okay', 'which is completely natural', 'that makes sense', 'and that\'s beautiful', 'which is totally valid'],
            'end': ['I\'m here for you', 'We\'re in this together', 'You\'re not alone', 'I believe in you', 'You\'ve got this', 'I\'m so proud of you']
        }
        
        # Add conversational fillers for more natural speech
        fillers = ['Well,', 'Hmm,', 'You see,', 'Actually,', 'I think,']
        
        # Avoid adding if already warm
        if any(phrase.lower() in response.lower() for phrases in warmth_phrases.values() for phrase in phrases):
            return response
        
        # Occasionally add warmth naturally
        if random.random() > 0.6 and not response.endswith('?'):
            response += f". {random.choice(warmth_phrases['end'])} 💙"
        
        # Add conversational filler at start occasionally for naturalness
        if random.random() > 0.8 and not any(response.startswith(f) for f in fillers):
            response = f"{random.choice(fillers)} {response[0].lower()}{response[1:]}"
        
        return response

class PersonalityEngine:
    def __init__(self):
        self.personality_traits = {
            'warmth': 0.9,
            'empathy': 0.95,
            'positivity': 0.85,
            'professionalism': 0.8
        }
    
    def add_emotional_expressiveness(self, response, emotion):
        """Add personality-consistent emotional expressions"""
        if not any(emoji in response for emoji in ['😊', '💙', '✨', '🫂']):
            if emotion in ['happy', 'neutral']:
                response += ' 😊'
            elif emotion in ['sad', 'fear']:
                response += ' 🫂'
        return response
    
    def adjust_for_relationship_building(self, response, is_first_interaction):
        """Adjust response style based on relationship stage"""
        if is_first_interaction:
            # More formal and reassuring for first interactions
            response = response.replace("I think", "I believe")
            response = response.replace("maybe", "perhaps")
            if not response.endswith(('?', '!', '😊', '💙', '✨', '🫂')):
                response += ' 😊'
        return response