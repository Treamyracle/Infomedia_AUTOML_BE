import pandas as pd
from pycaret.classification import setup as clf_setup, compare_models as clf_compare, pull as clf_pull, save_model as clf_save
from pycaret.regression import setup as reg_setup, compare_models as reg_compare, pull as reg_pull, save_model as reg_save
import os

DATA_DIR = "data"
MODEL_DIR = "models"

class MLEngine:
    def train(self, filename: str, target: str, task: str):
        file_path = os.path.join(DATA_DIR, filename)
        df = pd.read_csv(file_path)
        
        # Determine setup based on task
        if task == "classification":
            s = clf_setup(data=df, target=target, silent=True, verbose=False)
            best_model = clf_compare()
            results = clf_pull() # Get the results table
            
            # Save the model
            model_name = f"{filename.split('.')[0]}_{task}"
            save_path = os.path.join(MODEL_DIR, model_name)
            clf_save(best_model, save_path)
            
            # Extract simple accuracy metric
            # In PyCaret classification, default metric is Accuracy
            accuracy = results.iloc[0]['Accuracy']
            
        elif task == "regression":
            s = reg_setup(data=df, target=target, silent=True, verbose=False)
            best_model = reg_compare()
            results = reg_pull()
            
            model_name = f"{filename.split('.')[0]}_{task}"
            save_path = os.path.join(MODEL_DIR, model_name)
            reg_save(best_model, save_path)
            
            # For regression, we might return R2 or RMSE
            accuracy = results.iloc[0]['R2'] # Using R2 as "accuracy" equivalent
            
        else:
            raise ValueError("Invalid task type")

        return {
            "model_name": str(best_model),
            "accuracy": float(accuracy),
            "model_path": f"{save_path}.pkl",
            "features": df.columns.drop(target).tolist()
        }

engine = MLEngine()