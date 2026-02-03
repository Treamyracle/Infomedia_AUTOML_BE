import pandas as pd
import json
import os
from dotenv import load_dotenv
import google.generativeai as genai
import traceback
# --- PERBAIKAN: Tambahkan import typing di sini ---
from typing import List, Union, Dict 

# Import prompt template dari file prompts.py
from app.services.prompts import generate_feature_engineering_prompt

# 1. LOAD ENVIRONMENT VARIABLES
load_dotenv() 

# 2. KONFIGURASI GOOGLE GEMINI
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    print("⚠️ WARNING: GEMINI_API_KEY tidak ditemukan di .env! Fitur AI tidak akan berjalan.")
else:
    # Menggunakan model Gemini Pro
    genai.configure(api_key=API_KEY)

def get_llm_response(prompt_text: str) -> str:
    """
    Fungsi khusus untuk memanggil API Google Gemini.
    """
    if not API_KEY:
        raise ValueError("API Key belum disetting.")

    try:
        # Menggunakan model Gemini Pro (pastikan nama model benar)
        model = genai.GenerativeModel('gemma-3-27b-it') 
        
        # Kirim prompt
        response = model.generate_content(prompt_text)
        
        # Ambil text balasan
        return response.text
    except Exception as e:
        print(f"❌ Error saat memanggil Gemini: {e}")
        return "[]"

def generate_features_plan(df: pd.DataFrame, description: str = "Dataset User"):
    """
    Step A: Mengirim data ke LLM dan meminta saran fitur.
    Mengembalikan List of Dictionary (JSON) dari LLM.
    """
    print("🤖 AI Feature Engineer sedang berpikir...")
    
    # 1. Generate Prompt
    prompt_text = generate_feature_engineering_prompt(description, df)
    
    # 2. Call LLM
    try:
        response_text = get_llm_response(prompt_text)
        print("   📩 Terima balasan dari AI.")
    except Exception as e:
        print(f"   ⚠️ Gagal koneksi ke AI: {e}")
        return []
    
    # 3. Parsing JSON
    try:
        clean_text = response_text.replace("```json", "").replace("```", "").strip()
        features_plan = json.loads(clean_text)
        
        if not isinstance(features_plan, list):
            print("   ⚠️ Format balasan AI bukan List JSON.")
            return []
            
        return features_plan

    except json.JSONDecodeError:
        print(f"❌ Gagal parsing output JSON. Raw text:\n{response_text[:200]}...")
        return []

def execute_feature_code(df: pd.DataFrame, features_plan: List[Union[Dict, object]]) -> pd.DataFrame:
    """
    Step B: EXECUTOR TOOL.
    Menjalankan kode saran dari LLM ke DataFrame asli secara aman.
    Support input berupa List of Dicts ATAU List of Pydantic Models.
    """
    if not features_plan:
        print("   ⚠️ Tidak ada rencana fitur untuk dieksekusi.")
        return df, []

    print(f"⚙️ Menerapkan {len(features_plan)} fitur baru...")
    
    safe_locals = {'df': df, 'pd': pd} 
    
    report = []
    
    for item in features_plan:
        # Cek tipe item (Dict atau Pydantic Model)
        if hasattr(item, 'model_dump'):
            item_dict = item.model_dump()
        elif hasattr(item, 'dict'):
            item_dict = item.dict()
        elif isinstance(item, dict):
            item_dict = item
        else:
            item_dict = item.__dict__

        col_name = item_dict.get('name', 'Unknown_Feature')
        expr = item_dict.get('expression', '')
        
        if not expr:
            continue
            
        try:
            if "import" in expr or "os." in expr or "sys." in expr or "open(" in expr:
                raise ValueError("Kode ditolak (Unsafe/Berbahaya)")

            full_code = f"df['{col_name}'] = {expr}"
            
            exec(full_code, {}, safe_locals)
            
            df[col_name] = safe_locals['df'][col_name]
            
            print(f"   ✅ Created: {col_name}")
            report.append({"name": col_name, "status": "Success"})
            
        except Exception as e:
            print(f"   ⚠️ Failed: {col_name} - {str(e)}")
            report.append({"name": col_name, "status": "Failed", "error": str(e)})
            
    return df, report