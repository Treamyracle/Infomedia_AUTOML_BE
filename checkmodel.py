import google.generativeai as genai
import os
from dotenv import load_dotenv

# 1. Load Environment Variables
load_dotenv()

# 2. Configure API Key
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ Error: GEMINI_API_KEY not found in .env file.")
else:
    genai.configure(api_key=api_key)
    
    print(f"✅ API Key found: {api_key[:5]}...*****")
    print("\n🔎 Fetching available models from Google AI Studio...\n")

    try:
        # 3. List all models
        count = 0
        for m in genai.list_models():
            # We usually only care about models that support 'generateContent'
            if 'generateContent' in m.supported_generation_methods:
                print(f"🔹 Name: {m.name}")
                print(f"   Display Name: {m.display_name}")
                print(f"   Description: {m.description}")
                print(f"   Input Token Limit: {m.input_token_limit}")
                print(f"   Output Token Limit: {m.output_token_limit}")
                print("-" * 40)
                count += 1
        
        if count == 0:
            print("⚠️ No models found with 'generateContent' capability.")
        else:
            print(f"\n✨ Total available generative models: {count}")

    except Exception as e:
        print(f"❌ Failed to connect to Google API: {e}")