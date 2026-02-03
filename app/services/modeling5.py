import pandas as pd
import os
import logging
from typing import List, Dict, Any

# Import PyCaret Modules
# Kita import di dalam fungsi atau global, tapi global lebih bersih
from pycaret.classification import (
    setup as clf_setup, 
    create_model as clf_create, 
    pull as clf_pull, 
    save_model as clf_save,
    get_config as clf_get_config,
    models as clf_models_list
)
from pycaret.regression import (
    setup as reg_setup, 
    create_model as reg_create, 
    pull as reg_pull, 
    save_model as reg_save,
    get_config as reg_get_config,
    models as reg_models_list
)

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Direktori untuk menyimpan model sementara (jika perlu)
MODEL_SAVE_DIR = "models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

def _detect_task_type(df: pd.DataFrame, target: str) -> str:
    """
    Mendeteksi apakah ini tugas Klasifikasi atau Regresi secara otomatis.
    Logic: Jika target unik < 20 (atau tipe object/bool) -> Klasifikasi.
    """
    if df[target].dtype == 'object' or df[target].dtype == 'bool':
        return "classification"
    
    # Jika numerik tapi unique values sedikit (misal rating 1-5), anggap klasifikasi
    if df[target].nunique() < 20:
        return "classification"
    
    return "regression"

def train_diverse_models(df: pd.DataFrame, target: str) -> Dict[str, Any]:
    """
    Melatih 3 model dari keluarga algoritma yang berbeda:
    1. Linear Model (Logistic Regression / Linear Regression) -> Baseline sederhana.
    2. Tree Based (Decision Tree) -> Mudah diinterpretasi.
    3. Boosting (LightGBM / XGBoost / CatBoost) -> Akurasi tinggi (SOTA).
    
    Returns:
        Dictionary berisi object model yang sudah dilatih dan info task.
    """
    logger.info(f"🚀 Memulai Training Model untuk target: {target}")
    
    # 1. Deteksi Task
    task = _detect_task_type(df, target)
    logger.info(f"   Task Type Detected: {task.upper()}")

    trained_models = []
    model_metrics = []

    try:
        # ==========================================
        # LOGIC CLASSIFICATION
        # ==========================================
        if task == "classification":
            # Setup Environment
            # fix_imbalance=True bagus untuk data tidak seimbang
            clf_setup(data=df, target=target, session_id=123, verbose=False)
            
            # Definisi 3 Model Diversifikasi
            # 'lr' = Logistic Regression
            # 'dt' = Decision Tree
            # 'lightgbm' = Light Gradient Boosting (Cepat & Akurat)
            model_ids = ['lr', 'dt', 'lightgbm']
            
            print(f"   Training models: {model_ids}...")
            
            for m_id in model_ids:
                try:
                    # Train Model
                    model = clf_create(m_id, verbose=False)
                    trained_models.append(model)
                    
                    # Ambil Metrics (Akurasi, AUC, dll)
                    metrics_df = clf_pull()
                    acc = metrics_df.iloc[0]['Accuracy']
                    print(f"     ✅ {m_id.upper()} Trained. Acc: {acc:.4f}")
                    
                    model_metrics.append({
                        "model_id": m_id,
                        "accuracy": acc,
                        "model_obj": model
                    })
                except Exception as e:
                    print(f"     ⚠️ Gagal train {m_id}: {str(e)}")

        # ==========================================
        # LOGIC REGRESSION
        # ==========================================
        elif task == "regression":
            reg_setup(data=df, target=target, session_id=123, verbose=False)
            
            # 'lr' = Linear Regression
            # 'dt' = Decision Tree Regressor
            # 'lightgbm' = LightGBM Regressor
            model_ids = ['lr', 'dt', 'lightgbm']
            
            print(f"   Training models: {model_ids}...")
            
            for m_id in model_ids:
                try:
                    model = reg_create(m_id, verbose=False)
                    trained_models.append(model)
                    
                    metrics_df = reg_pull()
                    r2 = metrics_df.iloc[0]['R2']
                    print(f"     ✅ {m_id.upper()} Trained. R2: {r2:.4f}")
                    
                    model_metrics.append({
                        "model_id": m_id,
                        "r2": r2,
                        "model_obj": model
                    })
                except Exception as e:
                    print(f"     ⚠️ Gagal train {m_id}: {str(e)}")

        else:
            raise ValueError("Unknown task type")

        # Sort model berdasarkan performa terbaik
        if task == 'classification':
            model_metrics.sort(key=lambda x: x['accuracy'], reverse=True)
            best_score = model_metrics[0]['accuracy']
        else:
            model_metrics.sort(key=lambda x: x['r2'], reverse=True)
            best_score = model_metrics[0]['r2']

        logger.info(f"🏁 Training Selesai. Best Model: {model_metrics[0]['model_id'].upper()} ({best_score:.4f})")

        return {
            "status": "success",
            "task": task,
            "models_list": trained_models,        # List object model (untuk ensembling)
            "metrics_report": model_metrics       # Data untuk report JSON
        }

    except Exception as e:
        logger.error(f"Error pada training logic: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}