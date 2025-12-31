import glob, os
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

# --- CONFIGURATION ---
HORIZON = 20
WINDOW = 10
RR_TARGET = 2.0 
# --- INDIVIDUAL PATTERN FUNCTIONS (From your GitHub Reference) ---

def detect_head_shoulder(df, window=10):
    df['high_roll_max'] = df['High'].rolling(window=window).max()
    df['low_roll_min'] = df['Low'].rolling(window=window).min()
    mask_hs = ((df['high_roll_max'] > df['High'].shift(1)) & (df['high_roll_max'] > df['High'].shift(-1)) & (df['High'] < df['High'].shift(1)) & (df['High'] < df['High'].shift(-1)))
    mask_inv_hs = ((df['low_roll_min'] < df['Low'].shift(1)) & (df['low_roll_min'] < df['Low'].shift(-1)) & (df['Low'] > df['Low'].shift(1)) & (df['Low'] > df['Low'].shift(-1)))
    df['is_hs'] = mask_hs.astype(int)
    df['is_ihs'] = mask_inv_hs.astype(int)
    return df

def detect_double_top_bottom(df, window=10, threshold=0.01):
    # Mencari titik tertinggi/terendah lokal
    df['local_max'] = df['High'].rolling(window=window, center=True).max()
    df['local_min'] = df['Low'].rolling(window=window, center=True).min()
    
    # Double Top: Dua puncak yang tingginya hampir sama dalam rentang waktu tertentu
    # Kita menggunakan logika: High saat ini adalah local_max dan High beberapa bar lalu juga local_max
    mask_dt = (df['High'] == df['local_max']) & (abs(df['High'] - df['High'].shift(window)) < (df['High'] * threshold))
    mask_db = (df['Low'] == df['local_min']) & (abs(df['Low'] - df['Low'].shift(window)) < (df['Low'] * threshold))
    
    df['is_double_top'] = mask_dt.astype(int)
    df['is_double_bottom'] = mask_db.astype(int)
    return df

def detect_triangle_pattern(df, window=10):
    df['high_roll_max'] = df['High'].rolling(window=window).max()
    df['low_roll_min'] = df['Low'].rolling(window=window).min()
    mask_asc = (df['high_roll_max'] >= df['High'].shift(1)) & (df['low_roll_min'] <= df['Low'].shift(1)) & (df['Close'] > df['Close'].shift(1))
    mask_desc = (df['high_roll_max'] <= df['High'].shift(1)) & (df['low_roll_min'] >= df['Low'].shift(1)) & (df['Close'] < df['Close'].shift(1))
    df['is_asc_tri'] = mask_asc.astype(int)
    df['is_desc_tri'] = mask_desc.astype(int)
    return df

def detect_wedge(df, window=10):
    df['high_roll_max'] = df['High'].rolling(window=window).max()
    df['low_roll_min'] = df['Low'].rolling(window=window).min()
    
    # Gunakan np.where untuk menentukan tren (lebih cepat dan bebas error KeyError)
    # Tren High: High sekarang > High (window-1) bar lalu
    high_start = df['High'].shift(window - 1)
    df['trend_high'] = np.where(df['High'] > high_start, 1, np.where(df['High'] < high_start, -1, 0))
    
    # Tren Low: Low sekarang > Low (window-1) bar lalu
    low_start = df['Low'].shift(window - 1)
    df['trend_low'] = np.where(df['Low'] > low_start, 1, np.where(df['Low'] < low_start, -1, 0))
    
    mask_wedge_up = (df['high_roll_max'] >= df['High'].shift(1)) & (df['low_roll_min'] <= df['Low'].shift(1)) & (df['trend_high'] == 1) & (df['trend_low'] == 1)
    mask_wedge_down = (df['high_roll_max'] <= df['High'].shift(1)) & (df['low_roll_min'] >= df['Low'].shift(1)) & (df['trend_high'] == -1) & (df['trend_low'] == -1)
    
    df['is_wedge_up'] = mask_wedge_up.astype(int)
    df['is_wedge_down'] = mask_wedge_down.astype(int)
    return df

def detect_channel(df, window=10):
    channel_range = 0.1
    df['high_roll_max'] = df['High'].rolling(window=window).max()
    df['low_roll_min'] = df['Low'].rolling(window=window).min()
    
    # Logika tren yang sama dengan wedge untuk konsistensi
    high_start = df['High'].shift(window - 1)
    low_start = df['Low'].shift(window - 1)
    trend_h = np.where(df['High'] > high_start, 1, np.where(df['High'] < high_start, -1, 0))
    trend_l = np.where(df['Low'] > low_start, 1, np.where(df['Low'] < low_start, -1, 0))
    
    mask_channel_up = (df['high_roll_max'] >= df['High'].shift(1)) & (df['low_roll_min'] <= df['Low'].shift(1)) & \
                      (df['high_roll_max'] - df['low_roll_min'] <= channel_range * (df['high_roll_max'] + df['low_roll_min'])/2) & \
                      (trend_h == 1) & (trend_l == 1)
    
    mask_channel_down = (df['high_roll_max'] <= df['High'].shift(1)) & (df['low_roll_min'] >= df['Low'].shift(1)) & \
                        (df['high_roll_max'] - df['low_roll_min'] <= channel_range * (df['high_roll_max'] + df['low_roll_min'])/2) & \
                        (trend_h == -1) & (trend_l == -1)
                        
    df['is_channel_up'] = mask_channel_up.astype(int)
    df['is_channel_down'] = mask_channel_down.astype(int)
    return df

# --- CORE PROCESSING ---

def build_all_features(df):
    df.columns = [c.strip().capitalize() for c in df.columns]
    
    # Jalankan semua fungsi deteksi
    df = detect_head_shoulder(df, window=WINDOW)
    df = detect_double_top_bottom(df, window=WINDOW, threshold=0.05)
    df = detect_triangle_pattern(df, window=WINDOW)
    df = detect_wedge(df, window=WINDOW)   
    df = detect_channel(df, window=WINDOW) 
    
    # Fitur tambahan untuk membantu AI
    df["BODY_SIZE"] = abs(df["Close"] - df["Open"]) / (df["High"] - df["Low"] + 1e-9)
    df["ATR"] = (df["High"] - df["Low"]).rolling(14).mean()    
    df["VOL_SHOCK"] = df["ATR"] / (df["ATR"].shift(10) + 1e-9)
    df["MOMENTUM"] = df["Close"].pct_change(5)
    range_denom = (df['High'].rolling(10).max() - df['Low'].rolling(10).min())
    df['REL_RANGE'] = (df['High'] - df['Low']) / (range_denom + 1e-9)

    # --- PEMBERSIHAN FINAL ---
    # Ganti inf/-inf dengan NaN, lalu isi NaN dengan 0
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    
    return df

def label_strategy(df):
    df['target'] = 0
    h, l, c, o = df['High'].values, df['Low'].values, df['Close'].values, df['Open'].values
    
    # Mencari Support/Resistance 10 bar terakhir untuk penempatan SL yang lebih logis
    lookback_sl = 10 
    df['support'] = df['Low'].rolling(window=lookback_sl).min()
    df['resistance'] = df['High'].rolling(window=lookback_sl).max()
    
    support = df['support'].values
    resistance = df['resistance'].values

    # Kondisi Pola
    bull = ((df['is_ihs'] == 1) | (df['is_double_bottom'] == 1) | (df['is_asc_tri'] == 1) | 
            (df['is_wedge_down'] == 1) | (df['is_channel_up'] == 1)).values
            
    bear = ((df['is_hs'] == 1) | (df['is_double_top'] == 1) | (df['is_desc_tri'] == 1) | 
            (df['is_wedge_up'] == 1) | (df['is_channel_down'] == 1)).values

    for i in range(len(df) - HORIZON):
        entry_price = o[i+1] # Open bar berikutnya
        
        if bull[i]:
            # SL di Support terendah 10 bar terakhir (logika SL di bawah swing low)
            sl = support[i] 
            # Jika Support terlalu dekat (kurang dari spread), beri jarak sedikit
            if entry_price - sl < (df['ATR'].values[i] * 0.5):
                sl = entry_price - (df['ATR'].values[i] * 1.5)
            
            tp = entry_price + (entry_price - sl) * RR_TARGET
            
            for j in range(i+1, i+HORIZON):
                if l[j] <= sl: break # Kena SL
                if h[j] >= tp:
                    df.at[df.index[i], 'target'] = 1 # Profit
                    break
                    
        elif bear[i]:
            # SL di Resistance tertinggi 10 bar terakhir
            sl = resistance[i]
            if sl - entry_price < (df['ATR'].values[i] * 0.5):
                sl = entry_price + (df['ATR'].values[i] * 1.5)
                
            tp = entry_price - (sl - entry_price) * RR_TARGET
            
            for j in range(i+1, i+HORIZON):
                if h[j] >= sl: break # Kena SL
                if l[j] <= tp:
                    df.at[df.index[i], 'target'] = 2 # Profit
                    break
    return df

def train_model():
    all_files = glob.glob("data_chatpatterns/*.csv")
    if not all_files: return print("Folder data tidak ditemukan!")

    data_list = []
    for f in all_files:
        print(f"Reading: {os.path.basename(f)}")
        df = pd.read_csv(f, sep=None, engine='python')
        df = build_all_features(df)
        df = label_strategy(df)
        data_list.append(df)

    full_df = pd.concat(data_list).dropna()
    
    # Saring hanya baris yang punya signal agar training fokus
    condition = (
        (full_df['is_hs']==1) | (full_df['is_ihs']==1) | 
        (full_df['is_double_top']==1) | (full_df['is_double_bottom']==1) |
        (full_df['is_asc_tri']==1) | (full_df['is_desc_tri']==1) |
        (full_df['is_wedge_up']==1) | (full_df['is_wedge_down']==1) |
        (full_df['is_channel_up']==1) | (full_df['is_channel_down']==1)
    )
    train_df = full_df[condition].copy()

    features = [
        "BODY_SIZE", "REL_RANGE", "ATR", 
        "is_hs", "is_ihs", 
        "is_double_top", "is_double_bottom", 
        "is_asc_tri", "is_desc_tri",
        "is_wedge_up", "is_wedge_down",    
        "is_channel_up", "is_channel_down","VOL_SHOCK", "MOMENTUM"
    ]

    train_df = train_df.replace([np.inf, -np.inf], np.nan).dropna(subset=features)
    
    X = train_df[features]
    y = train_df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # SOLUSI WARNING: Berikan bobot seimbang agar label Sell (2) terdeteksi
    weights = compute_sample_weight(class_weight='balanced', y=y_train)

    model = xgb.XGBClassifier(
        objective="multi:softprob", 
        num_class=3, 
        learning_rate=0.05,
        max_depth=6,
        n_estimators=400,
        subsample=0.7,
        colsample_bytree=0.7,
        gamma=0.1, 
        tree_method="hist", 
        eval_metric="logloss",
        min_child_weight=5,
        reg_lambda=1.5,
        reg_alpha=0.1,
        random_state=42
    )

    print("🚀 Memulai Training dengan pola grafik...")
    model.fit(X_train, y_train, sample_weight=weights)

    print("\n--- PERFORMANCE REPORT ---")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, zero_division=0))
    
    joblib.dump(model, "models/pattern_model.pkl")
    print("✅ Model Berhasil Disimpan!")

if __name__ == "__main__":
    train_model()