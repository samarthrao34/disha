"""User Configuration Management for DISHA"""
import json
import os

class UserConfig:
    def __init__(self, config_file="user_config.json"):
        self.config_file = config_file
        self.config = self.load_config()
    
    def load_config(self):
        """Load user configuration from file"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
    
    def save_config(self):
        """Save user configuration to file"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except Exception:
            pass
    
    def is_first_time_user(self):
        """Check if this is a first-time user"""
        return 'user_name' not in self.config or not self.config.get('user_name')
    
    def get_user_name(self):
        """Get stored user name"""
        return self.config.get('user_name', '')
    
    def set_user_name(self, name):
        """Set user name"""
        self.config['user_name'] = name
        self.save_config()
    
    def get_interaction_mode(self):
        """Get preferred interaction mode (voice or text)"""
        return self.config.get('interaction_mode', 'voice')
    
    def set_interaction_mode(self, mode):
        """Set interaction mode"""
        self.config['interaction_mode'] = mode
        self.save_config()

def ask_user_name():
    """Ask for user's name"""
    print("\n" + "="*50)
    print("👋 Welcome to DISHA!")
    print("="*50)
    print("\nI'm DISHA, your compassionate mental health companion.")
    print("I'm here to listen, support, and walk with you through")
    print("whatever you're experiencing.\n")
    
    name = input("What would you like me to call you? ").strip()
    
    if not name:
        name = "Friend"
    
    print(f"\nIt's wonderful to meet you, {name}! 💙")
    print("I'm here for you whenever you need support.\n")
    
    return name

def select_interaction_mode():
    """Let user select interaction mode"""
    print("\n" + "="*50)
    print("How would you like to interact with me?")
    print("="*50)
    print("\n1. 🎤 Voice Mode - Talk to me naturally using your voice")
    print("2. ⌨️  Text Mode - Type your messages\n")
    
    while True:
        choice = input("Enter your choice (1 or 2): ").strip()
        
        if choice == "1":
            print("\n✅ Voice mode selected! I'll listen to your voice.")
            return "voice"
        elif choice == "2":
            print("\n✅ Text mode selected! Type whenever you're ready.")
            return "text"
        else:
            print("Please enter 1 for Voice or 2 for Text.")

def get_text_input(user_name):
    """Get text input from user in text mode"""
    try:
        text = input(f"\n{user_name}: ").strip()
        return text if text else None
    except (EOFError, KeyboardInterrupt):
        return None
