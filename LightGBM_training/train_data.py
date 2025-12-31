import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import glob, os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# --- CONFIGURATION ---
HORIZON = 25
WINDOW_SWING = 10
RR_TARGET = 2.0

def build_features_lgbm(df):
    df.columns = [c.strip().capitalize() for c in df.columns]

    # errors='coerce' menjaga agar jika ada format tanggal aneh tidak membuat crash
    df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
    df['hour'] = df['Time'].dt.hour
    
    # Indikator Dasar
    df['ema20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['ema50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['ema200'] = df['Close'].ewm(span=200, adjust=False).mean()
    
    # Fitur Trend (Categorical-like)
    df['trend_mode'] = 0 # Sideways
    df.loc[(df['ema20'] > df['ema50']) & (df['ema50'] > df['ema200']), 'trend_mode'] = 1 # Uptrend
    df.loc[(df['ema20'] < df['ema50']) & (df['ema50'] < df['ema200']), 'trend_mode'] = 2 # Downtrend
    
    # Market Structure & Fib
    df['sw_high'] = df['High'].rolling(window=WINDOW_SWING, center=True).max()
    df['sw_low'] = df['Low'].rolling(window=WINDOW_SWING, center=True).min()
    last_h = df['High'].where(df['High'] == df['sw_high']).ffill()
    last_l = df['Low'].where(df['Low'] == df['sw_low']).ffill()
    df['fib_retrace'] = (df['Close'] - last_l) / (last_h - last_l + 1e-9)
    # Tambahkan di build_features_lgbm
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['rsi'] = 100 - (100 / (1 + rs))
    df['is_london_ny'] = df['hour'].apply(lambda x: 1 if 14 <= x <= 22 else 0).astype(int)
    
    # Volatilitas
    df['atr'] = (df['High'] - df['Low']).rolling(14).mean()
    df['dist_ema20'] = (df['Close'] - df['ema20']) / df['atr'] # Jarak harga ke EMA
    
    return df.fillna(0)

def label_strategy(df):
    df['target'] = 0
    h, l, c = df['High'].values, df['Low'].values, df['Close'].values
    
    # Ambil sinyal dasar (Trend + Pullback)
    bull_base = (df['trend_mode'] == 1) & (df['fib_retrace'] >= 0.5) & (df['fib_retrace'] <= 0.75)
    bear_base = (df['trend_mode'] == 2) & (df['fib_retrace'] >= 0.25) & (df['fib_retrace'] <= 0.5)

    for i in range(len(df) - HORIZON):
        if bull_base.iloc[i]:
            entry = c[i]
            sl = df['sw_low'].values[i]
            risk = entry - sl
            if risk < df['atr'].values[i]: risk = df['atr'].values[i] * 2
            tp = entry + (risk * RR_TARGET)
            
            for j in range(i+1, i+HORIZON):
                if l[j] <= (entry - risk): break
                if h[j] >= tp:
                    df.at[df.index[i], 'target'] = 1
                    break
        elif bear_base.iloc[i]:
            entry = c[i]
            sl = df['sw_high'].values[i]
            risk = sl - entry
            if risk < df['atr'].values[i]: risk = df['atr'].values[i] * 2
            tp = entry - (risk * RR_TARGET)
            
            for j in range(i+1, i+HORIZON):
                if h[j] >= (entry + risk): break
                if l[j] <= tp:
                    df.at[df.index[i], 'target'] = 2
                    break
    return df

def train_lgbm():
    all_files = glob.glob("data_trading/*.csv")
    data_list = [label_strategy(build_features_lgbm(pd.read_csv(f))) for f in all_files]
    full_df = pd.concat(data_list)

    features = ['trend_mode', 'fib_retrace', 'hour', 'dist_ema20', 'atr']
    X = full_df[features]
    y = full_df['target']

    # LightGBM Dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    # 1. Hitung bobot secara otomatis
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights_dict = {classes[i]: weights[i] for i in range(len(classes))}
    
    # 2. Update Parameter (Hapus is_unbalance, ganti dengan class_weight)
    model = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=3,
        learning_rate=0.03,
        num_leaves=64,           # Ditambah agar model lebih kompleks
        min_data_in_leaf=20,     # Membantu model melihat pola kecil
        class_weight=class_weights_dict, # Memberi beban besar pada label 1 & 2
        n_estimators=500,        # Menambah durasi belajar
        importance_type='gain',
        verbosity=-1
    )

    model.fit(X_train, y_train)

    print("\n--- EVALUASI LIGHTGBM ---")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    # Cek Probabilitas (Untuk filter Precision)
    y_proba = model.predict_proba(X_test)
    threshold = 0.8
    # Prediksi baru berdasarkan threshold
    y_pred_strict = np.zeros_like(y_pred)
    for i in range(len(y_proba)):
        if y_proba[i, 1] > threshold:
            y_pred_strict[i] = 1
        elif y_proba[i, 2] > threshold:
            y_pred_strict[i] = 2
        else:
            y_pred_strict[i] = 0

    print("\n--- EVALUASI DENGAN THRESHOLD 80% ---")
    print(classification_report(y_test, y_pred_strict, zero_division=0))
    high_conf_buy = np.sum(y_proba[:, 1] > 0.7)
    print(f"Sinyal BUY dengan keyakinan > 70%: {high_conf_buy}")

    joblib.dump(model, "models/lgbm_trading_model.pkl")

if __name__ == "__main__":
    train_lgbm()