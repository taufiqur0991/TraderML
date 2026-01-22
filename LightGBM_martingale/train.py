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
HORIZON = 50  # Ditingkatkan agar trend lebih terlihat jelas (terutama M5/M15)

def build_martingale_trend(df):
    df.columns = [c.strip().upper() for c in df.columns]
    
    # 1. Trend Indicators
    df['EMA_50'] = df['CLOSE'].ewm(span=50).mean()
    df['EMA_200'] = df['CLOSE'].ewm(span=200).mean()
    
    # Slope EMA 200 (Normalized by price)
    df['SLOPE_200'] = df['EMA_200'].diff(5) / df['EMA_200']
    
    # Distance from EMA 200
    df['DIST_EMA200'] = (df['CLOSE'] - df['EMA_200']) / df['EMA_200']

    # 2. Volatility (ATR)
    tr = np.maximum(df["HIGH"] - df["LOW"], 
                    np.maximum(abs(df["HIGH"] - df["CLOSE"].shift(1)), 
                               abs(df["LOW"] - df["CLOSE"].shift(1))))
    df["ATR"] = tr.rolling(14).mean()
    df['RELATIVE_ATR'] = (df['HIGH'] - df['LOW']) / (df['ATR'] + 1e-9)

    # ADX Sederhana
    plus_dm = df['HIGH'].diff().clip(lower=0)
    df['ADX'] = (plus_dm.rolling(14).mean() / (df['ATR'] + 1e-9)) * 100

    # 3. Time & Momentum Features
    df["BODY_SIZE"] = abs(df["CLOSE"] - df["OPEN"]) / (df["HIGH"] - df["LOW"] + 1e-9)
    df['DAY'] = pd.to_datetime(df['DATETIME']).dt.dayofweek
    df['HOUR'] = pd.to_datetime(df['DATETIME']).dt.hour
    df['PREV_DIR'] = np.where(df['CLOSE'].shift(1) > df['OPEN'].shift(1), 1, -1)

    return df

def label_martingale_regime(df):
    df['target'] = 0
    close = df['CLOSE'].values
    atr = df['ATR'].values 
    
    for i in range(len(df) - HORIZON):
        price_change = close[i + HORIZON] - close[i]
        
        # Labeling berbasis Volatilitas (ATR) jauh lebih akurat untuk deteksi trend
        # Jika harga bergerak > 2x ATR, itu konfirmasi trend kuat
        if price_change > (atr[i] * 2.0): 
            df.at[df.index[i], 'target'] = 1 # Bullish
        elif price_change < -(atr[i] * 2.0):
            df.at[df.index[i], 'target'] = 2 # Bearish
        else:
            df.at[df.index[i], 'target'] = 0 # Sideways
            
    return df

def train_model():
    all_files = glob.glob("data/*.csv")
    if not all_files:
        print("Error: Folder 'data' kosong!")
        return

    data_list = []
    for f in all_files:
        print(f"Processing: {os.path.basename(f)}")
        df = pd.read_csv(f, sep=None, engine='python')
        
        try:
            df = build_martingale_trend(df)
            df = label_martingale_regime(df)
            data_list.append(df)
        except Exception as e:
            print(f"Skip {f}: {e}")
            continue

    if not data_list: return
    
    full_df = pd.concat(data_list).dropna()
    
    # --- PERBAIKAN LOGIKA BALANCING ---
    df_trending = full_df[full_df['target'] != 0]
    df_sideways = full_df[full_df['target'] == 0]
    
    # Ambil sample sideways secara dinamis agar tidak ValueError
    n_sample_sideways = min(len(df_sideways), len(df_trending) * 2)
    df_sideways_sampled = df_sideways.sample(n=n_sample_sideways, random_state=42)

    # Gabungkan dan urutkan kembali berdasarkan waktu agar Time Series Split valid
    train_df = pd.concat([df_trending, df_sideways_sampled]).sort_index()

    features = [
        "SLOPE_200", "DIST_EMA200", "ATR", "RELATIVE_ATR", 
        "ADX", "BODY_SIZE", "DAY", "HOUR", "PREV_DIR"
    ]
    
    # Pastikan X dan y diambil dari train_df (Data yang sudah seimbang)
    X = train_df[features]
    y = train_df['target']

    # Split data (shuffle=False sangat penting untuk data harga/forex)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Hitung bobot sampel
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

    model = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=3,
        n_estimators=1000, # Ditingkatkan
        learning_rate=0.03,
        max_depth=7,
        num_leaves=64,
        importance_type='gain',
        n_jobs=-1,
        random_state=42,
        verbosity=-1
    )

    print(f"Starting Training... (Trending: {len(df_trending)}, Sideways: {n_sample_sideways})")
    
    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(stopping_rounds=100)]
    )

    print("\n--- PERFORMANCE REPORT ---")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    if not os.path.exists('models'): os.makedirs('models')
    joblib.dump(model, "models/martingale_ai_model.pkl")
    print("Model Berhasil Disimpan!")

if __name__ == "__main__":
    train_model()