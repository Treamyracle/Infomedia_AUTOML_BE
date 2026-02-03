import pandas as pd
import numpy as np

def select_features(df: pd.DataFrame, target: str, correlation_threshold: float = 0.95) -> pd.DataFrame:
    """
    Melakukan seleksi fitur secara statistik:
    1. Low Variance Filter: Hapus kolom yang isinya sama semua.
    2. High Correlation Filter: Hapus fitur yang duplikat/redundant.
    
    Args:
        df: Dataframe input
        target: Nama kolom target (agar tidak ikut terhapus)
        correlation_threshold: Batas korelasi untuk menghapus fitur redundan (default 0.95)
    """
    print("🔍 Memulai Feature Selection...")
    initial_cols = len(df.columns)
    
    # 1. LOW VARIANCE FILTER
    # Hapus kolom yang hanya punya 1 nilai unik (tidak ada variasi = tidak ada informasi)
    for col in df.columns:
        if df[col].nunique() <= 1:
            print(f"   - Drop {col} (Low Variance/Constant)")
            df = df.drop(columns=[col])

    # 2. HIGH CORRELATION FILTER (Multicollinearity)
    # Hanya hitung korelasi pada kolom numerik
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Jangan libatkan target dalam pengecekan redundansi antar fitur
    if target in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=[target])
    
    corr_matrix = numeric_df.corr().abs()
    
    # Ambil segitiga atas dari matrix korelasi (karena simetris)
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Cari kolom yang korelasinya > threshold
    to_drop = [column for column in upper.columns if any(upper[column] > correlation_threshold)]
    
    if to_drop:
        print(f"   - Drop Redundant Features (> {correlation_threshold*100}%): {to_drop}")
        df = df.drop(columns=to_drop)
    
    dropped_count = initial_cols - len(df.columns)
    print(f"✅ Seleksi Selesai. {dropped_count} fitur dibuang. Sisa: {len(df.columns)} kolom.")
    
    return df