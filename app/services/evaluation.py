import logging
import pandas as pd
import numpy as np
from typing import Any, Dict
from sklearn.metrics import confusion_matrix

# Import PyCaret
from pycaret.classification import predict_model as clf_predict, pull as clf_pull, get_config as clf_get_config
from pycaret.regression import predict_model as reg_predict, pull as reg_pull, get_config as reg_get_config

logger = logging.getLogger(__name__)

def evaluate_model(model: Any, task_type: str, X_test=None, y_test=None) -> Dict[str, Any]:
    """
    Step 7: Evaluation & Final Report.
    Menguji model final pada data Hold-out dan menghasilkan metrics detail.
    
    Args:
        model: Model final (bisa single model atau ensemble).
        task_type: "classification" atau "regression".
        X_test, y_test: (Opsional) Data test eksternal. Jika None, pakai Hold-out set PyCaret.
        
    Returns:
        JSON/Dictionary berisi skor evaluasi dan data visualisasi (Confusion Matrix).
    """
    logger.info("📊 Memulai Evaluasi Final...")
    
    evaluation_report = {}
    
    try:
        # ==========================================
        # 1. PREDIKSI PADA TEST SET (Hold-out)
        # ==========================================
        if task_type == "classification":
            # Predict pada data hold-out (data=None artinya pakai data sisa split awal)
            predictions = clf_predict(model, data=None, verbose=False)
            metrics_df = clf_pull()
            
            # Ambil Metrics Utama
            # PyCaret return dataframe, kita ambil value baris pertama
            evaluation_report["metrics"] = {
                "Accuracy": float(metrics_df['Accuracy'].iloc[0]),
                "AUC": float(metrics_df['AUC'].iloc[0]),
                "Recall": float(metrics_df['Recall'].iloc[0]),
                "Precision": float(metrics_df['Precision'].iloc[0]),
                "F1": float(metrics_df['F1'].iloc[0])
            }
            
            # Buat Confusion Matrix Manual (agar bisa dikirim JSON)
            # PyCaret menyimpan kolom target asli dan 'prediction_label'
            y_true_col = clf_get_config('target_param')
            y_pred_col = 'prediction_label'
            
            if y_true_col in predictions.columns and y_pred_col in predictions.columns:
                cm = confusion_matrix(predictions[y_true_col], predictions[y_pred_col])
                # Convert numpy array ke list agar JSON serializable
                evaluation_report["confusion_matrix"] = cm.tolist() 
                evaluation_report["classes"] = sorted(predictions[y_true_col].unique().tolist())
            
        elif task_type == "regression":
            predictions = reg_predict(model, data=None, verbose=False)
            metrics_df = reg_pull()
            
            evaluation_report["metrics"] = {
                "R2": float(metrics_df['R2'].iloc[0]),
                "RMSE": float(metrics_df['RMSE'].iloc[0]),
                "MAE": float(metrics_df['MAE'].iloc[0]),
                "MSE": float(metrics_df['MSE'].iloc[0])
            }
            
            # Untuk regresi, kita kirim data Actual vs Predicted untuk Scatter Plot
            # Batasi 100 poin agar frontend tidak berat
            y_true_col = reg_get_config('target_param')
            y_pred_col = 'prediction_label'
            
            sample_pred = predictions[[y_true_col, y_pred_col]].head(100)
            evaluation_report["prediction_sample"] = sample_pred.to_dict(orient="records")

        logger.info("✅ Evaluasi Selesai.")
        return evaluation_report

    except Exception as e:
        logger.error(f"❌ Evaluasi Gagal: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}