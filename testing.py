# File: backend/test_step_1_4.py

import sys
import os

# Import modul services
# Pastikan nama file di app/services adalah ingestion.py, cleaning.py, dll.
from app.services import ingestion1
from app.services import cleaning2
from app.services import selection3
from app.services import feature_eng4

# --- KONFIGURASI ---
FILE_TEST = "data/HousingData.csv"  # Ganti dengan path file CSV/Excel kamu

# Tentukan kolom target untuk Step 3 (Selection). 
# Sesuaikan dengan nama kolom asli di CSV kamu (misal: 'ripeness' atau 'class').
# Jika null, script akan otomatis mengambil kolom terakhir.
TARGET_COL = "ripeness" 

def run_test():
    try:
        # ==========================================
        # STEP 1: INGESTION (Load Data)
        # ==========================================
        print("\n🔵 [STEP 1] Testing Ingestion...")
        df = ingestion1.load_data(FILE_TEST)
        print(f"   ✅ Data Loaded: {df.shape} (Baris, Kolom)")
        print("   Preview Data Mentah:\n", df.head(3))

        # Auto-detect target jika belum diset
        target = TARGET_COL if TARGET_COL else df.columns[-1]
        print(f"   🎯 Target Column: {target}")

        # ==========================================
        # STEP 2: CLEANING (Auto Clean)
        # ==========================================
        print("\n🟡 [STEP 2] Testing Cleaning...")
        df = cleaning2.auto_clean(df)
        print(f"   ✅ Data Cleaned: {df.shape}")
        
        # ==========================================
        # STEP 3: SELECTION (Feature Selection)
        # ==========================================
        print("\n🟠 [STEP 3] Testing Feature Selection...")
        # Kita butuh nama kolom target agar tidak ikut terhapus
        df = selection3.select_features(df, target=target)
        print(f"   ✅ Features Selected: {df.shape}")
        
        # ==========================================
        # STEP 4: FEATURE ENGINEERING (LLM Agent)
        # ==========================================
        print("\n🟣 [STEP 4] Testing AI Feature Engineering...")
        
        # A. Minta Saran ke LLM (Brain)
        print("   🧠 Menghubungi AI (Google Gemini)... mohon tunggu...")
        dataset_desc = "Data kematangan alpukat berdasarkan fisik buah."
        features_plan = feature_eng4.generate_features_plan(df, description=dataset_desc)
        
        if not features_plan:
            print("   ⚠️ AI tidak memberikan saran (Cek API Key / Limit).")
        else:
            print(f"   💡 AI Menyarankan {len(features_plan)} fitur baru:")
            for feat in features_plan:
                print(f"      - {feat.get('name')}: {feat.get('expression')}")

            # B. Eksekusi Saran (Executor)
            print("\n   ⚙️ Mengeksekusi kode Python dari AI...")
            df, report = feature_eng4.execute_feature_code(df, features_plan)
            
            success_count = sum(1 for r in report if r['status'] == 'Success')
            print(f"   ✅ Berhasil membuat {success_count} fitur baru.")
            
            print("\n📊 FINAL DATA SHAPE:", df.shape)
            print("Preview Data Akhir (dengan fitur baru):\n", df.head(3))

    except Exception as e:
        print(f"\n❌ ERROR TERJADI: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_test()