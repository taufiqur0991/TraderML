import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import glob, os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_sample_weight

# --- CONFIGURATION ---
HORIZON = 25      # Berapa candle ke depan untuk cek TP/SL
WINDOW_SWING = 10 # Window untuk mencari HH/HL
RR_TARGET = 2.0   # Risk Reward Ratio

# --- 1. CANDLESTICK DETECTION FUNCTIONS ---
def detect_candlesticks(df):
    """Mendeteksi Pin Bar dan Engulfing secara matematis"""
    body = abs(df['Close'] - df['Open'])
    candle_range = df['High'] - df['Low'] + 1e-9
    upper_wick = df['High'] - df[['Close', 'Open']].max(axis=1)
    lower_wick = df[['Close', 'Open']].min(axis=1) - df['Low']
    
    # 1. Bullish Engulfing
    df['is_bullish_engulfing'] = ((df['Close'] > df['Open']) & 
                                  (df['Close'].shift(1) < df['Open'].shift(1)) & 
                                  (df['Close'] > df['Open'].shift(1)) & 
                                  (df['Open'] < df['Close'].shift(1))).astype(int)
    
    # 2. Bearish Engulfing
    df['is_bearish_engulfing'] = ((df['Close'] < df['Open']) & 
                                  (df['Close'].shift(1) > df['Open'].shift(1)) & 
                                  (df['Close'] < df['Open'].shift(1)) & 
                                  (df['Open'] > df['Close'].shift(1))).astype(int)
    
    # 3. Pin Bar (Hammer/Shooting Star)
    # Body kecil (maks 30% dari total range) dan salah satu wick sangat panjang (min 60%)
    df['is_pinbar_up'] = ((lower_wick > (candle_range * 0.6)) & (body < (candle_range * 0.3))).astype(int)
    df['is_pinbar_down'] = ((upper_wick > (candle_range * 0.6)) & (body < (candle_range * 0.3))).astype(int)
    
    return df

# --- 2. MARKET STRUCTURE & TREND ---
def build_complete_features(df):
    df.columns = [c.strip().capitalize() for c in df.columns]
    
    # EMA Trend Filter
    df['ema20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['ema50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['ema200'] = df['Close'].ewm(span=200, adjust=False).mean()
    
    df['ema_uptrend'] = ((df['ema20'] > df['ema50']) & (df['ema50'] > df['ema200'])).astype(int)
    df['ema_downtrend'] = ((df['ema20'] < df['ema50']) & (df['ema50'] < df['ema200'])).astype(int)
    
    # Market Structure (Swing Points)
    df['sw_high'] = df['High'].rolling(window=WINDOW_SWING, center=True).max()
    df['sw_low'] = df['Low'].rolling(window=WINDOW_SWING, center=True).min()
    
    # Fibonacci Retracement (Pullback)
    last_h = df['High'].where(df['High'] == df['sw_high']).ffill()
    last_l = df['Low'].where(df['Low'] == df['sw_low']).ffill()
    df['fib_retrace'] = (df['Close'] - last_l) / (last_h - last_l + 1e-9)
    
    # Tambahkan deteksi candle
    df = detect_candlesticks(df)
    
    # Indikator Tambahan
    df["ATR"] = (df["High"] - df["Low"]).rolling(14).mean()
    df["VOL_SHOCK"] = df["ATR"] / (df["ATR"].shift(10) + 1e-9)
    
    return df.fillna(0)

# --- 3. LABELING STRATEGY (Logic Entry) ---
def label_strategy(df):
    df['target'] = 0
    h, l, c, o = df['High'].values, df['Low'].values, df['Close'].values, df['Open'].values
    
    # Syarat Entry sesuai Strategi Anda
    # Bull: Trend Up + Pullback (50-75%) + Konfirmasi (Engulfing/Pinbar)
    bull_signal = (
        (df['ema_uptrend'] == 1) & 
        (df['fib_retrace'] >= 0.5) & (df['fib_retrace'] <= 0.8) & 
        ((df['is_bullish_engulfing'] == 1) | (df['is_pinbar_up'] == 1))
    ).values

    # Bear: Trend Down + Pullback (25-50%) + Konfirmasi
    bear_signal = (
        (df['ema_downtrend'] == 1) & 
        (df['fib_retrace'] >= 0.2) & (df['fib_retrace'] <= 0.5) & 
        ((df['is_bearish_engulfing'] == 1) | (df['is_pinbar_down'] == 1))
    ).values

    for i in range(len(df) - HORIZON):
        if bull_signal[i]:
            entry = c[i]
            sl = df['sw_low'].values[i] # SL di HL sebelumnya
            risk = entry - sl
            if risk < (df['ATR'].values[i] * 0.5): risk = df['ATR'].values[i] * 1.5 # Safety
            tp = entry + (risk * RR_TARGET)
            
            for j in range(i+1, i+HORIZON):
                if l[j] <= (entry - risk): break 
                if h[j] >= tp:
                    df.at[df.index[i], 'target'] = 1 # BUY SUCCESS
                    break
                    
        elif bear_signal[i]:
            entry = c[i]
            sl = df['sw_high'].values[i] # SL di LH sebelumnya
            risk = sl - entry
            if risk < (df['ATR'].values[i] * 0.5): risk = df['ATR'].values[i] * 1.5
            tp = entry - (risk * RR_TARGET)
            
            for j in range(i+1, i+HORIZON):
                if h[j] >= (entry + risk): break
                if l[j] <= tp:
                    df.at[df.index[i], 'target'] = 2 # SELL SUCCESS
                    break
    return df

# --- 4. TRAINING ENGINE ---
def train_model():
    all_files = glob.glob("data_trading/*.csv") # Sesuaikan nama folder data Anda
    if not all_files: return print("File CSV tidak ditemukan!")

    data_list = []
    for f in all_files:
        df = pd.read_csv(f)
        df = build_complete_features(df)
        df = label_strategy(df)
        data_list.append(df)

    full_df = pd.concat(data_list)
    
    # Fitur yang akan dipelajari AI
    features = [
        'ema_uptrend', 'ema_downtrend', 'fib_retrace',
        'is_bullish_engulfing', 'is_bearish_engulfing', 
        'is_pinbar_up', 'is_pinbar_down',
        'VOL_SHOCK', 'ATR'
    ]

    # Hanya latih pada baris yang memiliki sinyal (agar model tidak malas)
    train_df = full_df[(full_df['ema_uptrend'] == 1) | (full_df['ema_downtrend'] == 1)].copy()
    
    X = train_df[features]
    y = train_df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Handle Class Imbalance (karena label 0 biasanya lebih banyak)
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.03,
        objective="multi:softprob",
        num_class=3,
        tree_method="hist",
        eval_metric="aucpr",
        random_state=42
    )

    print("🚀 Training sedang berjalan...")
    model.fit(X_train, y_train, sample_weight=sample_weights)

    print("\n--- HASIL EVALUASI ---")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    joblib.dump(model, "pullback_model.pkl")
    print("✅ Model Pullback Berhasil Disimpan!")

if __name__ == "__main__":
    train_model()