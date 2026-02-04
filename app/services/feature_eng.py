import pandas as pd
import numpy as np  # [FIX 1] Import Numpy
import json
import os
from dotenv import load_dotenv
import google.generativeai as genai
import traceback
from typing import List, Union, Dict 

# Import prompt template
from app.services.prompts import generate_feature_engineering_prompt

# 1. LOAD ENVIRONMENT VARIABLES
load_dotenv() 

# 2. KONFIGURASI GOOGLE GEMINI
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    print("⚠️ WARNING: GEMINI_API_KEY tidak ditemukan di .env! Fitur AI tidak akan berjalan.")
else:
    genai.configure(api_key=API_KEY)

def get_llm_response(prompt_text: str) -> str:
    """
    Fungsi khusus untuk memanggil API Google Gemini.
    """
    if not API_KEY:
        raise ValueError("API Key belum disetting.")

    try:
        # [FIX 2] Gunakan model yang stabil, cepat, dan murah untuk task ini
        # gemini-1.5-flash sangat direkomendasikan untuk tugas coding sederhana/JSON
        model = genai.GenerativeModel('gemini-2.5-flash') 
        
        # Set temperature rendah agar output konsisten (deterministic)
        generation_config = genai.GenerationConfig(temperature=0.2)
        
        response = model.generate_content(prompt_text, generation_config=generation_config)
        return response.text
    except Exception as e:
        print(f"❌ Error saat memanggil Gemini: {e}")
        return "[]"

def generate_features_plan(df: pd.DataFrame, description: str = "Dataset User") -> List[Dict]:
    """
    Step A: Mengirim data ke LLM dan meminta saran fitur.
    """
    print("🤖 AI Feature Engineer sedang berpikir...")
    
    # [FIX 3] Deteksi Target Column secara otomatis (asumsi kolom terakhir)
    # Ini penting agar Prompt tahu fitur apa yang relevan dibuat
    target_col = df.columns[-1] if not df.empty else None
    
    # Generate Prompt dengan Target Context
    prompt_text = generate_feature_engineering_prompt(description, df, target_col=target_col)
    
    # Call LLM
    try:
        response_text = get_llm_response(prompt_text)
        print("   📩 Terima balasan dari AI.")
    except Exception as e:
        print(f"   ⚠️ Gagal koneksi ke AI: {e}")
        return []
    
    # Parsing JSON (Robust)
    try:
        # Bersihkan markdown code block jika ada
        clean_text = response_text.replace("```json", "").replace("```", "").strip()
        
        # [FIX 4] Kadang LLM memberi teks sebelum JSON, cari kurung siku pertama dan terakhir
        start_idx = clean_text.find('[')
        end_idx = clean_text.rfind(']') + 1
        
        if start_idx != -1 and end_idx != -1:
            clean_text = clean_text[start_idx:end_idx]
            
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
    """
    if not features_plan:
        print("   ⚠️ Tidak ada rencana fitur untuk dieksekusi.")
        return df, []

    print(f"⚙️ Menerapkan {len(features_plan)} fitur baru...")
    
    # [FIX 5] Tambahkan 'np' (NumPy) ke safe_locals agar rumus matematika jalan
    safe_locals = {'df': df, 'pd': pd, 'np': np} 
    
    report = []
    
    for item in features_plan:
        # Handle Pydantic vs Dict
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
            # Security Check Basic
            if "import" in expr or "os." in expr or "sys." in expr or "open(" in expr:
                raise ValueError("Kode ditolak (Unsafe/Berbahaya)")

            # Eksekusi Kode
            full_code = f"df['{col_name}'] = {expr}"
            
            # Exec menggunakan safe_locals yang sudah ada numpy
            exec(full_code, {}, safe_locals)
            
            # Ambil hasil
            df[col_name] = safe_locals['df'][col_name]
            
            print(f"   ✅ Created: {col_name}")
            report.append({"name": col_name, "status": "Success"})
            
        except Exception as e:
            print(f"   ⚠️ Failed: {col_name} - {str(e)}")
            report.append({"name": col_name, "status": "Failed", "error": str(e)})
            
    return df, report