import pandas as pd
import numpy as np

def auto_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Melakukan pembersihan data otomatis:
    1. Drop baris yang terlalu banyak kosong (>50%).
    2. Imputasi (Isi nilai kosong): Median untuk angka, Mode untuk teks.
    3. Hapus Outlier menggunakan metode IQR.
    """
    initial_rows = len(df)
    print("🧹 Memulai Auto Cleaning...")

    # 1. DROP ROW JIKA KOSONG > 50%
    # Threshold: minimal 50% kolom harus terisi agar baris dipertahankan
    threshold = int(0.5 * len(df.columns))
    df = df.dropna(thresh=threshold)
    print(f"   - Drop baris kosong parah: {initial_rows - len(df)} baris dihapus.")

    # 2. IMPUTASI (MENGISI NILAI KOSONG)
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            # Jika Numerik (Angka) -> Isi dengan Median (Nilai tengah, lebih tahan outlier drpd Mean)
            if pd.api.types.is_numeric_dtype(df[col]):
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
            
            # Jika Kategorikal/Teks -> Isi dengan Mode (Paling sering muncul) atau "Unknown"
            else:
                if len(df[col].mode()) > 0:
                    mode_val = df[col].mode()[0]
                    df[col] = df[col].fillna(mode_val)
                else:
                    df[col] = df[col].fillna("Unknown")
    
    print("   - Imputasi nilai kosong selesai.")

    # 3. HAPUS OUTLIER (Metode IQR)
    # Hanya terapkan pada kolom numerik
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    rows_before_outlier = len(df)

    for col in numeric_cols:
        # Lewati kolom yang isinya biner (0/1) atau kategori angka (misal ID)
        if df[col].nunique() < 10: 
            continue

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Filter data
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    print(f"   - Hapus Outlier: {rows_before_outlier - len(df)} baris dihapus.")
    print(f"✅ Cleaning Selesai. Data akhir: {df.shape[0]} baris.")
    
    return df