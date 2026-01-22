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
SAVE_MODEL_NAME = "models/market_structure_lgbm.pkl"

def fast_pivot_detection(df, window=5):
    highs = df['HIGH'].values
    lows = df['LOW'].values
    length = len(df)
    
    p_high = np.zeros(length)
    p_low = np.zeros(length)
    
    # Menggunakan loop yang lebih cepat dengan perbandingan langsung
    for i in range(window, length - window):
        # Ambil range sekitarnya sekali saja
        h_slice = highs[i-window : i+window+1]
        l_slice = lows[i-window : i+window+1]
        
        # Jika nilai bar i adalah yang tertinggi di h_slice
        if highs[i] == np.max(h_slice):
            p_high[i] = 1
        # Jika nilai bar i adalah yang terendah di l_slice
        if lows[i] == np.min(l_slice):
            p_low[i] = 2
            
    return p_high + p_low

def build_features(df):
    df.columns = [c.strip().upper() for c in df.columns]
    
    # Pre-calculate ATR secara manual agar persis dengan rumus MQL5 iATR
    high_low = df['HIGH'] - df['LOW']
    high_cp = np.abs(df['HIGH'] - df['CLOSE'].shift(1))
    low_cp = np.abs(df['LOW'] - df['CLOSE'].shift(1))
    tr = np.maximum(high_low, np.maximum(high_cp, low_cp))
    df['ATR'] = tr.rolling(14).mean()
    
    # Sinkronisasi Fitur dengan MQL5 (Penting!)
    df['F1_MOMENTUM'] = (df['CLOSE'] - df['CLOSE'].shift(5)) / (df['ATR'] + 1e-9)
    
    # F2 & F3: Gunakan shift(1) agar tidak mengambil High/Low bar yang sedang berjalan
    df['F2_DIST_HIGH'] = (df['HIGH'].rolling(50).max().shift(1) - df['CLOSE']) / (df['ATR'] + 1e-9)
    df['F3_DIST_LOW'] = (df['CLOSE'] - df['LOW'].rolling(50).min().shift(1)) / (df['ATR'] + 1e-9)
    
    # F4: Gunakan TICK_VOLUME jika ada
    vol_col = 'TICK_VOLUME' if 'TICK_VOLUME' in df.columns else 'VOLUME'
    df['F4_REL_VOL'] = df[vol_col] / (df[vol_col].rolling(20).mean() + 1e-9)
    
    df['F5_BODY_RATIO'] = (df['CLOSE'] - df['OPEN']) / (df['ATR'] + 1e-9)
    
    df['DATETIME'] = pd.to_datetime(df['DATETIME'])
    df['F6_HOUR'] = df['DATETIME'].dt.hour
    
    return df

def label_data(df, backcandles=40, window=5):
    df['ISPIVOT'] = fast_pivot_detection(df, window)
    
    targets = np.full(len(df), -1)
    high_vals = df['HIGH'].values
    low_vals = df['LOW'].values
    pivots = df['ISPIVOT'].values
    
    # Gunakan loop yang lebih efisien
    for candle in range(backcandles + window, len(df)):
        # Data yang tersedia hanya sampai (candle - window) untuk hindari repaint
        idx_pivots = np.where(pivots[candle-backcandles-window : candle-window] > 0)[0]
        
        if len(idx_pivots) < 4: continue
        
        # Ambil index absolut pivot
        abs_idx = idx_pivots + (candle-backcandles-window)
        
        # Filter Highs dan Lows terakhir
        last_highs = [high_vals[i] for i in abs_idx if pivots[i] == 1 or pivots[i] == 3]
        last_lows = [low_vals[i] for i in abs_idx if pivots[i] == 2 or pivots[i] == 3]

        if len(last_highs) >= 2 and len(last_lows) >= 2:
            h_curr, h_prev = last_highs[-1], last_highs[-2]
            l_curr, l_prev = last_lows[-1], last_lows[-2]
            
            if h_curr > h_prev and l_curr > l_prev: targets[candle] = 3 # HH
            elif h_curr < h_prev and l_curr < l_prev: targets[candle] = 0 # LL
            elif l_curr > l_prev: targets[candle] = 2 # HL
            elif h_curr < h_prev: targets[candle] = 1 # LH
            
    df['target'] = targets
    return df[df['target'] != -1].copy()

def train_model():
    all_files = glob.glob("data/*.csv")
    if not all_files: return

    list_df = []
    for f in all_files:
        print(f"Loading: {os.path.basename(f)}...", end="\r")
        df = pd.read_csv(f, sep=None, engine='python')
        df = build_features(df)
        df = label_data(df)
        list_df.append(df)
        print(f"Processed: {os.path.basename(f)}     ")

    full_df = pd.concat(list_df)
    features = ['F1_MOMENTUM', 'F2_DIST_HIGH', 'F3_DIST_LOW', 'F4_REL_VOL', 'F5_BODY_RATIO', 'F6_HOUR']
    
    X = full_df[features]
    y = full_df['target'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = lgb.LGBMClassifier(
        objective='multiclass', 
        num_class=4, 
        n_estimators=500, 
        learning_rate=0.05, 
        class_weight='balanced',
        force_col_wise=True
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[lgb.early_stopping(50)])

    joblib.dump(model, SAVE_MODEL_NAME)
    print("Selesai!")

if __name__ == "__main__":
    train_model()