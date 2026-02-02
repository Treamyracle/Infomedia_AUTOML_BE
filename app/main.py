from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import pandas as pd

from app.schemas import TrainRequest, TrainingResponse
from app.utils import detect_task_type, get_smart_column_suggestion
from app.ml_engine import engine

app = FastAPI()

# Enable CORS so your Vue/React frontend can talk to this
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure directories exist
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

@app.get("/")
def read_root():
    return {"status": "ML Server Running"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    1. Uploads the file
    2. Returns columns and a suggested target to the Frontend
    """
    file_location = f"data/{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Read just the headers to be fast
    df = pd.read_csv(file_location, nrows=5)
    
    return {
        "filename": file.filename,
        "columns": df.columns.tolist(),
        "suggested_target": get_smart_column_suggestion(df)
    }

@app.post("/train", response_model=TrainingResponse)
def train_model(request: TrainRequest):
    """
    1. Receives the filename and target column
    2. Auto-detects Classification vs Regression
    3. Trains the model
    """
    try:
        df_full = pd.read_csv(f"data/{request.filename}")
        
        # Auto-detect task if user didn't specify
        if request.task_type == "auto":
            task = detect_task_type(df_full, request.target_column)
        else:
            task = request.task_type

        # Run Training
        result = engine.train(request.filename, request.target_column, task)
        
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))