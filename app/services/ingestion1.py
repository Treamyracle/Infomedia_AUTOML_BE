import pandas as pd
import os

def load_data(file_path: str) -> pd.DataFrame:
    """
    Membaca file CSV atau Excel dan mengembalikannya sebagai Pandas DataFrame.
    Menangani berbagai error encoding dan format.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File tidak ditemukan di path: {file_path}")

    file_ext = os.path.splitext(file_path)[1].lower()

    try:
        # === HANDLING CSV ===
        if file_ext == '.csv':
            try:
                # Coba utf-8 dulu (standar modern)
                df = pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError:
                # Jika gagal, coba latin1 (sering terjadi di file Excel lama yg di-save as CSV)
                print(f"⚠️ Warning: Gagal baca {file_path} dengan UTF-8, mencoba Latin-1...")
                df = pd.read_csv(file_path, encoding='latin1')
        
        # === HANDLING EXCEL ===
        elif file_ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        
        else:
            raise ValueError(f"Format file '{file_ext}' tidak didukung. Harap gunakan .csv atau .xlsx")

        # === OPTIONAL: AUTO DATE PARSING ===
        # Coba deteksi kolom tanggal otomatis
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    # Coba convert sampel (tidak semua baris agar cepat)
                    pd.to_datetime(df[col].dropna().iloc[:10], errors='raise')
                    # Jika sukses, convert seluruh kolom
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except (ValueError, TypeError):
                    pass # Bukan tanggal, lanjut

        print(f"✅ Berhasil load data: {df.shape[0]} baris, {df.shape[1]} kolom.")
        return df

    except Exception as e:
        raise ValueError(f"Gagal membaca file: {str(e)}")