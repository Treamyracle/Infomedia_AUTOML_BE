from pydantic import BaseModel
from typing import List, Optional, Any, Dict

# --- Shared Models ---
class FeaturePlanItem(BaseModel):
    name: str
    expression: str
    rationale: Optional[str] = None

# --- Upload ---
class UploadResponse(BaseModel):
    filename: str
    columns: List[str]
    suggested_target: Optional[str] = None
    row_count: int

# --- Feature Engineering ---
class FeatureSuggestRequest(BaseModel):
    filename: str
    description: Optional[str] = "Dataset for machine learning"

class FeatureSuggestResponse(BaseModel):
    plan: List[FeaturePlanItem]

class FeatureApplyRequest(BaseModel):
    filename: str
    plan: List[FeaturePlanItem]

class FeatureApplyResponse(BaseModel):
    new_filename: str
    new_columns: List[str]
    message: str

# --- Training ---
class TrainRequest(BaseModel):
    filename: str
    target_column: str
    task_type: Optional[str] = "auto" # 'classification', 'regression'
    model_choice: Optional[str] = "auto" # 'auto' or specific algorithm

class TrainResponse(BaseModel):
    status: str
    task_type: str
    accuracy_score: float # Accuracy or R2 depending on task
    metrics_detail: Dict[str, float]
    confusion_matrix: Optional[List[List[int]]] = None # For classification
    prediction_sample: Optional[List[Dict[str, Any]]] = None # For regression
    best_model_name: str
    message: str