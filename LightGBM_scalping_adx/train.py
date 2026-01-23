import glob, os
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

# =====================
# CONFIGURATION
# =====================
SAVE_MODEL_NAME = "models/scalping_adx.pkl"
ADX_THRESHOLD = 50

def build_features(df):
    """Menghitung indikator ADX, Williams %R, dan SMA secara manual"""
    df.columns = [c.strip().upper() for c in df.columns]
    
    # 1. ATR (Average True Range) - Dasar perhitungan ADX
    high_low = df['HIGH'] - df['LOW']
    high_cp = np.abs(df['HIGH'] - df['CLOSE'].shift(1))
    low_cp = np.abs(df['LOW'] - df['CLOSE'].shift(1))
    tr = np.maximum(high_low, np.maximum(high_cp, low_cp))
    df['ATR'] = tr.rolling(14).mean()
    
    # 2. Williams %R Manual
    high_max = df['HIGH'].rolling(14).max()
    low_min = df['LOW'].rolling(14).min()
    df['WILLR'] = -100 * (high_max - df['CLOSE']) / (high_max - low_min + 1e-9)

    # 3. ADX Manual (DMP, DMN, ADX)
    plus_dm = df['HIGH'].diff()
    minus_dm = df['LOW'].diff()
    
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
    
    df['DMP'] = 100 * (pd.Series(plus_dm).rolling(14).mean() / (df['ATR'] + 1e-9))
    df['DMN'] = 100 * (pd.Series(minus_dm).rolling(14).mean() / (df['ATR'] + 1e-9))
    
    dx = 100 * np.abs(df['DMP'] - df['DMN']) / (df['DMP'] + df['DMN'] + 1e-9)
    df['ADX'] = dx.rolling(14).mean()

    # 4. SMA 50 & Jarak Harga
    df['SMA50'] = df['CLOSE'].rolling(50).mean()
    df['DIST_SMA50'] = (df['CLOSE'] - df['SMA50']) / (df['ATR'] + 1e-9)
    
    # 5. Fitur Tambahan (Dari referensi kode awal Anda)
    df['F1_MOMENTUM'] = (df['CLOSE'] - df['CLOSE'].shift(5)) / (df['ATR'] + 1e-9)
    vol_col = 'TICK_VOLUME' if 'TICK_VOLUME' in df.columns else 'VOLUME'
    if vol_col in df.columns:
        df['F4_REL_VOL'] = df[vol_col] / (df[vol_col].rolling(20).mean() + 1e-9)
    
    df['DATETIME'] = pd.to_datetime(df['DATETIME'])
    df['F6_HOUR'] = df['DATETIME'].dt.hour
    
    return df

def label_data_binary(df):
    """Memberi label 1 (Buy) atau 0 (Sell) berdasarkan kriteria video"""
    targets = np.full(len(df), -1)
    
    # Syarat BUY: ADX kuat, DI+ di atas, Williams %R Oversold (Koreksi dalam tren naik)
    buy_cond = (df['ADX'] > ADX_THRESHOLD) & (df['DMP'] > df['DMN']) & (df['WILLR'] < -80)
    
    # Syarat SELL: ADX kuat, DI- di atas, Williams %R Overbought (Koreksi dalam tren turun)
    sell_cond = (df['ADX'] > ADX_THRESHOLD) & (df['DMN'] > df['DMP']) & (df['WILLR'] > -20)
    
    targets[buy_cond] = 1
    targets[sell_cond] = 0
    
    df['target'] = targets
    # Hanya simpan data yang memenuhi kriteria strategi untuk training
    return df[df['target'] != -1].copy()

def train_model():
    if not os.path.exists("models"): os.makedirs("models")
    
    all_files = glob.glob("data/*.csv")
    if not all_files:
        print("Folder 'data/' kosong! Masukkan file CSV M5 Anda di sana.")
        return

    list_df = []
    for f in all_files:
        print(f"Loading: {os.path.basename(f)}...", end="\r")
        df = pd.read_csv(f, sep=None, engine='python')
        df = build_features(df)
        df = label_data_binary(df)
        list_df.append(df)

    if not list_df: return
    full_df = pd.concat(list_df)

    # AUDIT DATA
    # percentage = (len(full_df) / 100000) * 100 # Contoh estimasi jika data awal 100rb baris
    print(f"\n[INFO] Total Sinyal Ditemukan: {len(full_df)} baris.")
    print(f"[INFO] Proporsi Sinyal BUY: {(full_df['target']==1).sum()}")
    print(f"[INFO] Proporsi Sinyal SELL: {(full_df['target']==0).sum()}")

    # Fitur yang akan dipelajari model
    features = ['ADX', 'DMP', 'DMN', 'WILLR', 'DIST_SMA50', 'F1_MOMENTUM', 'F6_HOUR']
    if 'F4_REL_VOL' in full_df.columns: features.append('F4_REL_VOL')

    X = full_df[features]
    y = full_df['target'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model Binary Classification
    model = lgb.LGBMClassifier(
        objective='binary',
        n_estimators=1000,
        learning_rate=0.01,
        num_leaves=15,
        min_child_samples=10,
        class_weight='balanced',
        force_col_wise=True
    )
    
    model.fit(
        X_train, y_train, 
        eval_set=[(X_test, y_test)], 
        callbacks=[lgb.early_stopping(100)]
    )

    joblib.dump(model, SAVE_MODEL_NAME)
    print(f"\nModel berhasil disimpan di: {SAVE_MODEL_NAME}")

if __name__ == "__main__":
    train_model()