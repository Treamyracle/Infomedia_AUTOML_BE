import pandas as pd
import json
import os
from dotenv import load_dotenv
import google.generativeai as genai
import traceback

# Import prompt template dari file prompts.py yang sudah dibuat sebelumnya
from app.services.prompts import generate_feature_engineering_prompt

# 1. LOAD ENVIRONMENT VARIABLES
load_dotenv() # Membaca file .env

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
        # Menggunakan model Gemini Pro (gratis & cepat)
        model = genai.GenerativeModel('gemma-3-1b-it')
        
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
    
    # 1. Generate Prompt (dari prompts.py)
    prompt_text = generate_feature_engineering_prompt(description, df)
    
    # 2. Call LLM (REAL API CALL)
    try:
        response_text = get_llm_response(prompt_text)
        print("   📩 Terima balasan dari AI.")
    except Exception as e:
        print(f"   ⚠️ Gagal koneksi ke AI: {e}")
        return []
    
    # 3. Parsing JSON
    try:
        # Bersihkan format markdown ```json ... ``` jika ada
        clean_text = response_text.replace("```json", "").replace("```", "").strip()
        
        features_plan = json.loads(clean_text)
        
        # Validasi sederhana: pastikan hasil berupa list
        if not isinstance(features_plan, list):
            print("   ⚠️ Format balasan AI bukan List JSON.")
            return []
            
        return features_plan

    except json.JSONDecodeError:
        print(f"❌ Gagal parsing output JSON. Raw text:\n{response_text[:200]}...")
        return []

def execute_feature_code(df: pd.DataFrame, features_plan: list) -> pd.DataFrame:
    """
    Step B: EXECUTOR TOOL.
    Menjalankan kode saran dari LLM ke DataFrame asli secara aman.
    """
    if not features_plan:
        print("   ⚠️ Tidak ada rencana fitur untuk dieksekusi.")
        return df, []

    print(f"⚙️ Menerapkan {len(features_plan)} fitur baru...")
    
    # Context aman: hanya boleh akses df, pandas (pd), dan numpy (np jika perlu)
    safe_locals = {'df': df, 'pd': pd} 
    
    report = []
    
    for item in features_plan:
        col_name = item.get('name', 'Unknown_Feature')
        expr = item.get('expression', '')
        
        # Skip jika ekspresi kosong
        if not expr:
            continue
            
        try:
            # Security Check Sederhana
            if "import" in expr or "os." in expr or "sys." in expr or "open(" in expr:
                raise ValueError("Kode ditolak (Unsafe/Berbahaya)")

            # EKSEKUSI KODE PANDAS
            # Syntax: df['Nama'] = Rumus
            full_code = f"df['{col_name}'] = {expr}"
            
            # Jalankan di lingkungan terisolasi (safe_locals)
            exec(full_code, {}, safe_locals)
            
            # Update df utama dengan hasil dari safe_locals
            df[col_name] = safe_locals['df'][col_name]
            
            print(f"   ✅ Created: {col_name}")
            report.append({"name": col_name, "status": "Success"})
            
        except Exception as e:
            print(f"   ⚠️ Failed: {col_name} - {str(e)}")
            report.append({"name": col_name, "status": "Failed", "error": str(e)})
            
    return df, report