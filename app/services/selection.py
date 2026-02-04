import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold

def select_features(df: pd.DataFrame, target: str, correlation_threshold: float = 0.95) -> pd.DataFrame:
    """
    Melakukan seleksi fitur 'Smart' Best Practice:
    1. Quasi-Constant Filter: Hapus jika 1 nilai mendominasi > 99%.
    2. Relevance Filter: Hapus fitur yang korelasi ke targetnya sangat rendah (< 0.01).
    3. Smart Correlation Filter: Hapus fitur duplikat, tapi PERTAHANKAN yang 
       korelasinya lebih tinggi terhadap Target.
    """
    print("🔍 Memulai Advanced Feature Selection...")
    initial_cols = len(df.columns)
    
    # Pisahkan fitur dan target sementara
    if target in df.columns:
        y = df[target]
        X = df.drop(columns=[target])
    else:
        print("⚠️ Target tidak ditemukan, menjalankan mode unsupervised.")
        y = None
        X = df

    # ==========================================
    # 1. QUASI-CONSTANT FILTER (Lebih canggih dari nunique <= 1)
    # ==========================================
    # Menghapus kolom yang > 99% isinya sama. 
    # Contoh: Kolom 'Negara' isinya 'Indonesia' semua, cuma 1 baris 'Malaysia'. Ini noise.
    constant_cols = []
    for col in X.columns:
        # Jika nilai terbanyak muncul > 99% dari total data
        if X[col].value_counts(normalize=True).iloc[0] > 0.99:
            constant_cols.append(col)
    
    if constant_cols:
        print(f"   - Drop Quasi-Constant Features (>99% sama): {constant_cols}")
        X = X.drop(columns=constant_cols)

    # ==========================================
    # PERSIAPAN KORELASI
    # ==========================================
    # Kita butuh data numerik untuk korelasi
    numeric_X = X.select_dtypes(include=[np.number])
    
    # Hitung korelasi fitur ke TARGET (Relevansi)
    target_corr = {}
    if y is not None and pd.api.types.is_numeric_dtype(y):
        # Ini menghitung korelasi setiap kolom di X terhadap y
        target_corr = numeric_X.corrwith(y).abs()
        
        # 2. RELEVANCE FILTER (Opsional tapi bagus)
        # Hapus fitur yang tidak ada hubungannya sama sekali dengan target
        low_corr_features = target_corr[target_corr < 0.01].index.tolist()
        if low_corr_features:
            print(f"   - Drop Low Relevance Features (Corr to Target < 0.01): {low_corr_features}")
            X = X.drop(columns=low_corr_features)
            numeric_X = numeric_X.drop(columns=low_corr_features, errors='ignore')
            # Update target_corr setelah drop
            target_corr = target_corr.drop(labels=low_corr_features, errors='ignore')

    # ==========================================
    # 3. SMART CORRELATION FILTER (Redundancy)
    # ==========================================
    # Hitung korelasi antar fitur (A vs B)
    corr_matrix = numeric_X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    to_drop = set()

    # Iterasi setiap pasangan fitur yang korelasinya tinggi
    for column in upper.columns:
        for row in upper.index:
            if upper.loc[row, column] > correlation_threshold:
                # Ditemukan pasangan duplikat: Fitur A (row) dan Fitur B (column)
                
                # LOGIKA SMART: Bandingkan korelasi mereka terhadap Target
                if len(target_corr) > 0:
                    score_A = target_corr.get(row, 0)
                    score_B = target_corr.get(column, 0)
                    
                    # Buang yang skor korelasinya ke target LEBIH KECIL
                    if score_A < score_B:
                        to_drop.add(row)    # Buang A, pertahankan B
                    else:
                        to_drop.add(column) # Buang B, pertahankan A
                else:
                    # Fallback jika target bukan numerik/tidak ada: Buang kolom kedua
                    to_drop.add(column)

    if to_drop:
        print(f"   - Drop Redundant Features (Smart Drop): {list(to_drop)}")
        X = X.drop(columns=list(to_drop))

    # ==========================================
    # 4. FINISHING
    # ==========================================
    # Gabungkan kembali dengan target
    if y is not None:
        df_final = pd.concat([X, y], axis=1)
    else:
        df_final = X
        
    dropped_count = initial_cols - len(df_final.columns)
    print(f"✅ Seleksi Selesai. {dropped_count} fitur dibuang. Sisa: {len(df_final.columns)} kolom.")
    
    return df_final