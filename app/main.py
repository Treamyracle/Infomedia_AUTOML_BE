from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import pandas as pd
import logging

# Config & Schemas
from app.config import settings
from app.schemas import (
    UploadResponse, 
    TrainRequest, TrainResponse,
    FeatureSuggestRequest, FeatureSuggestResponse,
    FeatureApplyRequest, FeatureApplyResponse
)

# Services (The 7 Steps)
from app.services import (
    ingestion, 
    cleaning, 
    selection, 
    feature_eng, 
    modeling, 
    ensembling, 
    evaluation
)

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("automl_api")

app = FastAPI(title=settings.APP_NAME)

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "ML Server Running", "version": "2.0 Pro"}

# ==========================================
# 1. UPLOAD & INGESTION
# ==========================================
@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    try:
        # Save file
        file_path = os.path.join(settings.DATA_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Step 1: Ingestion Service
        df = ingestion.load_data(file_path)
        
        # Smart Target Suggestion (Ambil kolom terakhir sebagai default)
        suggested_target = df.columns[-1] if len(df.columns) > 0 else None
        
        return {
            "filename": file.filename,
            "columns": df.columns.tolist(),
            "suggested_target": suggested_target,
            "row_count": len(df)
        }
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# 4. FEATURE ENGINEERING (AI)
# ==========================================
@app.post("/features/suggest", response_model=FeatureSuggestResponse)
def suggest_features(request: FeatureSuggestRequest):
    """Meminta saran fitur baru ke LLM"""
    file_path = os.path.join(settings.DATA_DIR, request.filename)
    if not os.path.exists(file_path):
        raise HTTPException(404, "File not found")

    try:
        df = ingestion.load_data(file_path)
        # Step 4A: Call LLM
        plan = feature_eng.generate_features_plan(df, request.description)
        return {"plan": plan}
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@app.post("/features/apply", response_model=FeatureApplyResponse)
def apply_features(request: FeatureApplyRequest):
    """Menerapkan saran fitur dan menyimpan dataset baru"""
    file_path = os.path.join(settings.DATA_DIR, request.filename)
    if not os.path.exists(file_path):
        raise HTTPException(404, "File not found")

    try:
        df = ingestion.load_data(file_path)
        
        # Step 4B: Execute Code
        df_augmented, report = feature_eng.execute_feature_code(df, request.plan)
        
        # Simpan file baru agar tidak menimpa original
        new_filename = f"augmented_{request.filename}"
        new_path = os.path.join(settings.DATA_DIR, new_filename)
        df_augmented.to_csv(new_path, index=False)
        
        success_count = sum(1 for r in report if r['status'] == 'Success')
        
        return {
            "new_filename": new_filename,
            "new_columns": df_augmented.columns.tolist(),
            "message": f"Berhasil menambah {success_count} fitur baru."
        }
    except Exception as e:
        raise HTTPException(500, detail=str(e))

# ==========================================
# 5, 6, 7. TRAINING PIPELINE
# ==========================================
@app.post("/train", response_model=TrainResponse)
def train_pipeline(request: TrainRequest):
    """
    Menjalankan Full Pipeline:
    Clean -> Select -> Train (3 Models) -> Ensemble -> Evaluate
    """
    file_path = os.path.join(settings.DATA_DIR, request.filename)
    if not os.path.exists(file_path):
        raise HTTPException(404, "File not found")

    try:
        logger.info(f"Starting pipeline for {request.filename}...")
        
        # --- [Step 1] Reload Data ---
        df = ingestion.load_data(file_path)
        
        # --- [Step 2] Cleaning ---
        df = cleaning.auto_clean(df)
        
        # --- [Step 3] Selection ---
        df = selection.select_features(df, target=request.target_column)
        
        # --- [Step 5] Modeling ---
        # Note: Step 4 dilewati di sini karena dianggap sudah dilakukan via API /features/apply
        train_res = modeling.train_diverse_models(df, target=request.target_column)
        
        if train_res['status'] != 'success':
            raise ValueError(f"Training failed: {train_res.get('message')}")
            
        models_list = train_res['models_list']
        task_type = train_res['task']
        best_single_model_name = train_res['metrics_report'][0]['model_id']

        # --- [Step 6] Ensembling ---
        ensemble_res = ensembling.ensemble_models(models_list, task_type)
        final_model = ensemble_res['final_model']
        
        # --- [Step 7] Evaluation ---
        eval_report = evaluation.evaluate_model(final_model, task_type)
        
        # Prepare Response
        metrics = eval_report.get('metrics', {})
        main_score = metrics.get('Accuracy') if task_type == 'classification' else metrics.get('R2')
        
        return {
            "status": "success",
            "task_type": task_type,
            "accuracy_score": main_score or 0.0,
            "metrics_detail": metrics,
            "confusion_matrix": eval_report.get('confusion_matrix'),
            "prediction_sample": eval_report.get('prediction_sample'),
            "best_model_name": "Ensemble (Voting)" if ensemble_res['status'] == 'success' else best_single_model_name,
            "message": "Model berhasil dilatih dan dievaluasi."
        }

    except Exception as e:
        logger.error(f"Pipeline Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))