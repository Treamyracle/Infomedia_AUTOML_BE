import pandas as pd
from pycaret.classification import setup as clf_setup, compare_models as clf_compare, create_model as clf_create, pull as clf_pull, save_model as clf_save
from pycaret.regression import setup as reg_setup, compare_models as reg_compare, create_model as reg_create, pull as reg_pull, save_model as reg_save
import os

DATA_DIR = "data"
MODEL_DIR = "models"

class MLEngine:
    def train(self, filename: str, target: str, task: str, model_id: str = "auto"):
        file_path = os.path.join(DATA_DIR, filename)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {filename} tidak ditemukan di {DATA_DIR}")

        # 1. Load Data
        df = pd.read_csv(file_path)
        
        # 2. [Req User] Hapus semua baris yang ada nilai kosong (NaN)
        # Ini penting agar tidak error saat training
        df = df.dropna()

        # Cek jika data jadi kosong setelah dropna
        if df.empty:
            raise ValueError("Data kosong setelah menghapus baris NaN. Pastikan datasetmu bagus.")

        # === LOGIC CLASSIFICATION ===
        if task == "classification":
            # 3. [Fix Error] Hapus kelas yang datanya cuma 1 biji
            # PyCaret butuh minimal 2 data per kelas untuk split train/test
            if target in df.columns:
                class_counts = df[target].value_counts()
                # Ambil kelas yang muncul lebih dari 1 kali
                valid_classes = class_counts[class_counts > 1].index
                # Filter data cuma simpan kelas yang valid
                df = df[df[target].isin(valid_classes)]
            
            # Cek lagi takutnya habis difilter datanya habis
            if df.empty:
                raise ValueError("Data tidak cukup untuk Classification setelah filter kelas rare.")

            # Jalankan Setup (verbose=False pengganti silent=True)
            s = clf_setup(data=df, target=target, verbose=False)
            
            if model_id == "auto":
                best_model = clf_compare()
            else:
                best_model = clf_create(model_id)
                
            results = clf_pull()
            
            model_name = f"{filename.split('.')[0]}_{task}_{model_id}"
            save_path = os.path.join(MODEL_DIR, model_name)
            clf_save(best_model, save_path)
            
            try:
                accuracy = results.iloc[0]['Accuracy']
            except:
                accuracy = 0.0

        # === LOGIC REGRESSION ===
        elif task == "regression":
            s = reg_setup(data=df, target=target, verbose=False)
            
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
            raise ValueError(f"Task type tidak valid: {task}")

        return {
            "model_name": str(best_model),
            "accuracy": float(accuracy),
            "model_path": f"{save_path}.pkl",
            "features": df.columns.drop(target).tolist()
        }

engine = MLEngine()