import pandas as pd
from pycaret.classification import setup as clf_setup, compare_models as clf_compare, create_model as clf_create, pull as clf_pull, save_model as clf_save
from pycaret.regression import setup as reg_setup, compare_models as reg_compare, create_model as reg_create, pull as reg_pull, save_model as reg_save
import os

DATA_DIR = "data"
MODEL_DIR = "models"

class MLEngine:
    # Tambahkan parameter model_id dengan default 'auto'
    def train(self, filename: str, target: str, task: str, model_id: str = "auto"):
        file_path = os.path.join(DATA_DIR, filename)
        df = pd.read_csv(file_path)
        
        # 1. Classification Logic
        if task == "classification":
            s = clf_setup(data=df, target=target, silent=True, verbose=False)
            
            # Cek apakah Auto atau Manual
            if model_id == "auto":
                best_model = clf_compare()
            else:
                best_model = clf_create(model_id) # Manual: Buat model spesifik (contoh: 'rf', 'dt')
                
            results = clf_pull()
            model_name = f"{filename.split('.')[0]}_{task}_{model_id}"
            save_path = os.path.join(MODEL_DIR, model_name)
            clf_save(best_model, save_path)
            
            # Ambil akurasi (Baris pertama jika Auto, atau hasil create jika Manual)
            try:
                accuracy = results.iloc[0]['Accuracy']
            except:
                accuracy = 0.0 # Fallback jika format table beda

        # 2. Regression Logic
        elif task == "regression":
            s = reg_setup(data=df, target=target, silent=True, verbose=False)
            
            if model_id == "auto":
                best_model = reg_compare()
            else:
                best_model = reg_create(model_id)
                
            results = reg_pull()
            model_name = f"{filename.split('.')[0]}_{task}_{model_id}"
            save_path = os.path.join(MODEL_DIR, model_name)
            reg_save(best_model, save_path)
            
            try:
                accuracy = results.iloc[0]['R2']
            except:
                accuracy = 0.0

        else:
            raise ValueError("Invalid task type")

        return {
            "model_name": str(best_model),
            "accuracy": float(accuracy),
            "model_path": f"{save_path}.pkl",
            "features": df.columns.drop(target).tolist()
        }

engine = MLEngine()