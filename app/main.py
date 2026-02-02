from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import pandas as pd
import traceback # Import ini untuk melihat error detail di console

from app.schemas import TrainRequest, TrainingResponse
from app.utils import detect_task_type, get_smart_column_suggestion
from app.ml_engine import engine

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

@app.get("/")
def read_root():
    return {"status": "ML Server Running"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_location = f"data/{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    df = pd.read_csv(file_location, nrows=5)
    
    return {
        "filename": file.filename,
        "columns": df.columns.tolist(),
        "suggested_target": get_smart_column_suggestion(df)
    }

@app.post("/train", response_model=TrainingResponse)
def train_model(request: TrainRequest):
    try:
        # Debugging: Print apa yang diterima server
        print(f"Training Request: {request}")

        df_full = pd.read_csv(f"data/{request.filename}")
        
        if request.task_type == "auto":
            task = detect_task_type(df_full, request.target_column)
        else:
            task = request.task_type
        
        # PASTIKAN BARIS INI BENAR:
        result = engine.train(
            filename=request.filename, 
            target=request.target_column, 
            task=task,
            model_id=request.model_choice # <-- Penting!
        )
        
        return result

    except Exception as e:
        # Ini akan mencetak error lengkap di terminal VS Code kamu
        traceback.print_exc()
        # Ini mengirim pesan error ke Frontend (Alert)
        raise HTTPException(status_code=500, detail=str(e))