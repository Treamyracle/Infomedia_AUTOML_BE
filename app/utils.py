import pandas as pd
import os

def detect_task_type(df: pd.DataFrame, target_col: str) -> str:
    """
    Heuristic to decide if it's Regression or Classification.
    """
    if target_col not in df.columns:
        raise ValueError(f"Column {target_col} not found")

    target_data = df[target_col]
    
    # 1. If it's a string/object -> Definitely Classification
    if target_data.dtype == 'object':
        return 'classification'
    
    # 2. If it's numeric, check unique values count
    unique_count = target_data.nunique()
    total_count = len(target_data)
    
    # If less than 20 unique values OR less than 5% unique, treat as Classifcation
    if unique_count < 20 or (unique_count / total_count < 0.05):
        return 'classification'
    
    return 'regression'

def get_smart_column_suggestion(df: pd.DataFrame) -> str:
    """
    Suggests the likely target column (usually the last one).
    """
    return df.columns[-1]