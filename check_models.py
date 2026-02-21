import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

print("\n" + "="*60)
print("  🔎 Checking Available Google Gemini Models...")
print("="*60 + "\n")

# Get API key
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ ERROR: GEMINI_API_KEY not found in .env file.")
    print("Please make sure your API key is correctly set in d:\\DISHA\\.env")
    exit()

try:
    # Configure the generative AI client
    genai.configure(api_key=api_key)

    print("✅ API Key found. Fetching models...\n")
    
    # List all available models
    all_models = genai.list_models()
    
    print("="*60)
    print("  🤖 Available Models for 'generateContent'")
    print("="*60)

    found_flash = False
    flash_model_name = ""

    for m in all_models:
        # Check if the model supports the 'generateContent' method
        if 'generateContent' in m.supported_generation_methods:
            model_name = m.name
            print(f"  - {model_name}")
            
            # Find the best "flash" model
            if 'flash' in model_name:
                found_flash = True
                flash_model_name = model_name # Store the first flash model found

    if found_flash:
        print("\n" + "="*60)
        print("  🎯 Recommended Model Found!")
        print("="*60)
        print(f"\n  ✅ The correct model name for DISHA is: {flash_model_name}\n")
    else:
        print("\n" + "="*60)
        print("  ⚠️ No 'flash' model found!")
        print("="*60)
        print("\n  Please use one of the models listed above.")

except Exception as e:
    print(f"\n❌ An error occurred while fetching models: {e}")
    print("\nPlease check the following:")
    print("  1. Your internet connection.")
    print("  2. The validity of your GEMINI_API_KEY in the .env file.")
    print("  3. Google AI services status.")

print("\n" + "="*60)
print("  Check complete.")
print("="*60 + "\n")
