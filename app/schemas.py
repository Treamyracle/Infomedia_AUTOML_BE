from pydantic import BaseModel
from typing import Optional, List

class TrainRequest(BaseModel):
    filename: str
    target_column: str
    task_type: Optional[str] = "auto"  # 'classification', 'regression', or 'auto'
    model_choice: Optional[str] = "auto" # Specific model ID or 'auto'

class TrainingResponse(BaseModel):
    model_name: str
    accuracy: float
    model_path: str
    features: List[str]