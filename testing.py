# File: backend/test_full_pipeline.py

import sys
import os
import pandas as pd
import warnings
import logging

# Matikan warning system & log PyCaret agar terminal bersih
warnings.filterwarnings('ignore')
logging.getLogger('pycaret').setLevel(logging.ERROR)

# --- IMPORT MODUL SERVICES (1-7) ---
# Pastikan nama file di folder app/services/ sesuai dengan nama import ini
from app.services import ingestion1
from app.services import cleaning2
from app.services import selection3
from app.services import feature_eng4
from app.services import modeling5
from app.services import ensembling6
from app.services import evaluation7

# --- KONFIGURASI TEST ---
FILE_TEST = "data/HousingData.csv"   # Ganti dengan dataset kamu
TARGET_COL = "MEDV"                  # Kolom Target (Label)

def run_full_pipeline():
    print("🚀 MEMULAI FULL AUTOML PIPELINE TEST (Step 1 - 7) 🚀")
    print("=" * 60)
    
    try:
        # ==========================================
        # STEP 1: INGESTION
        # ==========================================
        print("\n🔵 [1] Ingestion Service")
        df = ingestion1.load_data(FILE_TEST)
        print(f"   ✅ Data Loaded: {df.shape}")

        # Tentukan target otomatis jika config kosong
        target = TARGET_COL if TARGET_COL else df.columns[-1]
        print(f"   🎯 Target: {target}")

        if target not in df.columns:
            raise ValueError(f"Target '{target}' tidak ada di dataset!")

        # ==========================================
        # STEP 2: CLEANING
        # ==========================================
        print("\n🟡 [2] Cleaning Service")
        df = cleaning2.auto_clean(df)
        print(f"   ✅ Data Cleaned: {df.shape}")

        # ==========================================
        # STEP 3: SELECTION
        # ==========================================
        print("\n🟠 [3] Feature Selection")
        df = selection3.select_features(df, target=target)
        print(f"   ✅ Features Selected: {df.shape}")

        # ==========================================
        # STEP 4: FEATURE ENGINEERING (AI)
        # ==========================================
        print("\n🟣 [4] AI Feature Engineering")
        # Deskripsi dataset manual (bisa dibuat dinamis nanti)
        desc = "Dataset prediksi harga rumah."
        
        # Panggil AI (Gemini)
        print("   🧠 Asking AI for features...")
        features_plan = feature_eng4.generate_features_plan(df, description=desc)
        
        if features_plan:
            print(f"   💡 AI Suggestions ({len(features_plan)}):")
            for f in features_plan:
                print(f"      - {f.get('name')}: {f.get('expression')}")
            
            # Eksekusi Kode
            df, report = feature_eng4.execute_feature_code(df, features_plan)
            success = sum(1 for r in report if r['status'] == 'Success')
            print(f"   ✅ Applied {success} new features.")
        else:
            print("   ⚠️ AI Skipped (No response/Quota exceeded).")
            
        print(f"   📊 Final Data Shape: {df.shape}")

        # ==========================================
        # STEP 5: MODELING
        # ==========================================
        print("\n🔴 [5] Modeling (Training)")
        print("   🚀 Training 3 algorithms (Linear, Tree, Boosting)...")
        
        # verbose=False sudah diatur di dalam modeling5.py agar tidak spam log
        train_res = modeling5.train_diverse_models(df, target=target)
        
        if train_res['status'] != 'success':
            raise ValueError(f"Training Failed: {train_res.get('message')}")
            
        models_list = train_res['models_list']
        task_type = train_res['task']
        metrics = train_res['metrics_report']
        
        print(f"   ✅ Task Detected: {task_type.upper()}")
        print(f"   🏆 Best Single Model: {metrics[0]['model_id'].upper()}")

        # ==========================================
        # STEP 6: ENSEMBLING
        # ==========================================
        print("\n🔵 [6] Ensembling (Voting/Stacking)")
        ensemble_res = ensembling6.ensemble_models(models_list, task_type)
        
        final_model = ensemble_res['final_model']
        ens_metrics = ensemble_res.get('metrics', {})
        
        if ensemble_res['status'] == 'success':
            print(f"   ✅ Ensemble Created using {len(models_list)} models.")
            # Tampilkan metrics ensemble
            score_key = 'accuracy' if task_type == 'classification' else 'r2'
            print(f"   🌟 Ensemble Score ({score_key}): {ens_metrics.get(score_key, 0):.4f}")
        else:
            print("   ⚠️ Ensembling Skipped (Not enough models). Using best single model.")

        # ==========================================
        # STEP 7: EVALUATION
        # ==========================================
        print("\n🟢 [7] Final Evaluation Report")
        eval_report = evaluation7.evaluate_model(final_model, task_type)
        
        print("-" * 40)
        print("📊 METRICS SUMMARY:")
        for k, v in eval_report.get('metrics', {}).items():
            print(f"   🔹 {k.ljust(10)}: {v:.4f}")
            
        if 'confusion_matrix' in eval_report:
            print(f"\n🧩 Confusion Matrix:\n{eval_report['confusion_matrix']}")
            
        print("-" * 40)
        print("\n✅✅✅ PIPELINE COMPLETE SUCCESSFULLY ✅✅✅")

    except Exception as e:
        print(f"\n❌ PIPELINE ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_full_pipeline()