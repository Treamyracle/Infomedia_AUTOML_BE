import logging
from typing import List, Any, Dict
from pycaret.classification import blend_models as clf_blend
from pycaret.regression import blend_models as reg_blend
from pycaret.classification import pull as clf_pull
from pycaret.regression import pull as reg_pull

# Setup Logging
logger = logging.getLogger(__name__)

def ensemble_models(models_list: List[Any], task_type: str) -> Dict[str, Any]:
    """
    Step 6: Model Ensembling.
    Menggabungkan Top 3 Model dari Step 5 menggunakan teknik Voting/Blending.
    
    Args:
        models_list: List object model yang sudah dilatih di Step 5.
        task_type: "classification" atau "regression".
        
    Returns:
        Dictionary berisi model hasil ensemble dan metrik performanya.
    """
    logger.info(f"🧩 Memulai Ensembling untuk {len(models_list)} model...")

    # Validasi: Butuh minimal 2 model untuk ensemble
    if len(models_list) < 2:
        logger.warning("⚠️ Jumlah model kurang dari 2. Tidak bisa Ensemble. Mengembalikan model terbaik saja.")
        return {
            "status": "skipped", 
            "final_model": models_list[0], 
            "message": "Not enough models to blend"
        }

    try:
        final_model = None
        metrics = {}

        if task_type == "classification":
            # Soft Voting: Mengambil rata-rata probabilitas prediksi
            final_model = clf_blend(estimator_list=models_list, verbose=False)
            
            # Ambil report akurasi blending
            metrics_df = clf_pull()
            # Biasanya baris 'Mean' atau baris pertama
            acc = metrics_df.iloc[0]['Accuracy']
            metrics = {"accuracy": acc, "auc": metrics_df.iloc[0]['AUC']}
            
        elif task_type == "regression":
            # Blending: Rata-rata nilai prediksi
            final_model = reg_blend(estimator_list=models_list, verbose=False)
            
            metrics_df = reg_pull()
            r2 = metrics_df.iloc[0]['R2']
            metrics = {"r2": r2, "rmse": metrics_df.iloc[0]['RMSE']}

        logger.info(f"✅ Ensembling Selesai. Metrics: {metrics}")

        return {
            "status": "success",
            "final_model": final_model,
            "metrics": metrics
        }

    except Exception as e:
        logger.error(f"❌ Ensembling Gagal: {str(e)}")
        # Fallback: Kembalikan model terbaik dari list jika blending gagal
        return {
            "status": "failed", 
            "final_model": models_list[0], 
            "error": str(e)
        }