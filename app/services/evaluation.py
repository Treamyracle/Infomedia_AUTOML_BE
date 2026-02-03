import logging
import pandas as pd
import numpy as np
from typing import Any, Dict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    r2_score, mean_squared_error, mean_absolute_error
)

# Import PyCaret
from pycaret.classification import predict_model as clf_predict, get_config as clf_get_config
from pycaret.regression import predict_model as reg_predict, get_config as reg_get_config

logger = logging.getLogger(__name__)

def evaluate_model(model: Any, task_type: str, X_test=None, y_test=None) -> Dict[str, Any]:
    """
    Step 7: Evaluation & Final Report.
    Menguji model final dan menghitung metrics secara manual menggunakan Scikit-Learn.
    """
    logger.info("📊 Memulai Evaluasi Final (Manual Calculation)...")
    
    evaluation_report = {}
    
    try:
        # ==========================================
        # 1. PREDIKSI PADA TEST SET (Hold-out)
        # ==========================================
        if task_type == "classification":
            # Predict pada data hold-out PyCaret (data=None)
            predictions = clf_predict(model, data=None, verbose=False)
            
            # Ambil Nama Kolom Target (Asli) & Prediksi
            y_true_col = clf_get_config('target_param')
            y_pred_col = 'prediction_label'
            
            # Pastikan kolom ada
            if y_true_col not in predictions.columns or y_pred_col not in predictions.columns:
                raise ValueError(f"Kolom target/prediksi tidak ditemukan. Cols: {predictions.columns}")

            y_true = predictions[y_true_col]
            y_pred = predictions[y_pred_col]
            
            # Hitung Metrics Manual (Lebih Aman & Akurat)
            # average='weighted' menangani binary maupun multiclass dengan baik
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            evaluation_report["metrics"] = {
                "Accuracy": float(acc),
                "Precision": float(prec),
                "Recall": float(rec),
                "F1": float(f1)
            }
            
            # Confusion Matrix
            cm = confusion_matrix(y_true, y_pred)
            evaluation_report["confusion_matrix"] = cm.tolist() 
            evaluation_report["classes"] = sorted(y_true.unique().tolist())
            
        elif task_type == "regression":
            predictions = reg_predict(model, data=None, verbose=False)
            
            y_true_col = reg_get_config('target_param')
            y_pred_col = 'prediction_label'
            
            y_true = predictions[y_true_col]
            y_pred = predictions[y_pred_col]
            
            # Hitung Metrics Manual
            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            
            evaluation_report["metrics"] = {
                "R2": float(r2),
                "RMSE": float(rmse),
                "MAE": float(mae)
            }
            
            # Sample Data untuk Scatter Plot
            sample_pred = predictions[[y_true_col, y_pred_col]].head(100)
            evaluation_report["prediction_sample"] = sample_pred.to_dict(orient="records")

        logger.info(f"✅ Evaluasi Selesai. Metrics: {evaluation_report['metrics']}")
        return evaluation_report

    except Exception as e:
        logger.error(f"❌ Evaluasi Gagal: {str(e)}")
        import traceback
        traceback.print_exc()
        # Return fallback jika gagal total
        return {"error": str(e), "metrics": {"Accuracy": 0.0}}