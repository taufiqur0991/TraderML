import glob, os
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib

from scipy.signal import argrelextrema
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

USE_FIB = False
RISK_REWARD = 1.1

# =========================
# LOAD & PREPARE DATA
# =========================
def load_all_csv(path="data_market_structure/*.csv"):
    all_df = []

    for f in glob.glob(path):
        print(f"📄 Processing {os.path.basename(f)}")

        df = pd.read_csv(f, sep="\t")

        # Uppercase semua nama kolom untuk konsistensi
        df.columns = [c.upper() for c in df.columns]

        # Extract symbol dari nama file (contoh: BTCUSDT_data.csv → BTCUSDT)
        df["SYMBOL"] = os.path.basename(f).split("_")[0]

        # Parse datetime dan set sebagai index
        df["DATETIME"] = pd.to_datetime(df["DATETIME"])
        df.set_index("DATETIME", inplace=True)
        df.sort_index(inplace=True)

        df = build_features(df)
        df["entry_mode"] = np.where(
            (df["choch_up"] == 1) & (df["pullback_buy"] == 1),
            1,
            0
        )
        df = build_risk_reward(df, rr=RISK_REWARD)
        df = build_target(df)

        df = df.dropna().reset_index(drop=True)
        all_df.append(df)

    return pd.concat(all_df, ignore_index=True)


# =========================
# RISK MANAGEMENT (SL / TP)
# =========================
def build_risk_reward(df, rr=RISK_REWARD):
    """
    Dual Stop Loss Strategy:
    - Structure SL: berdasarkan swing low/high
    - Rejection SL: berdasarkan candle low/high
    """
    
    # --- STRUCTURE SL ---
    df["sl_buy_struct"]  = df["swing_low"]
    df["sl_sell_struct"] = df["swing_high"]

    # --- REJECTION SL ---
    df["sl_buy_reject"]  = df["Low"]
    df["sl_sell_reject"] = df["High"]

    # --- FINAL SL (pilih berdasarkan entry mode) ---
    df["sl_buy"] = np.where(
        df["entry_mode"] == 1,
        df["sl_buy_reject"],
        df["sl_buy_struct"]
    )

    df["sl_sell"] = np.where(
        df["entry_mode"] == 1,
        df["sl_sell_reject"],
        df["sl_sell_struct"]
    )

    # --- SL DISTANCE ---
    df["sl_dist_buy"]  = df["Close"] - df["sl_buy"]
    df["sl_dist_sell"] = df["sl_sell"] - df["Close"]

    # Proteksi nilai negatif/nol agar tidak merusak perhitungan TP
    df.loc[df["sl_dist_buy"] <= 0, "sl_dist_buy"] = np.nan
    df.loc[df["sl_dist_sell"] <= 0, "sl_dist_sell"] = np.nan

    # --- TAKE PROFIT ---
    df["tp_buy"]  = df["Close"] + rr * df["sl_dist_buy"]
    df["tp_sell"] = df["Close"] - rr * df["sl_dist_sell"]

    return df


# =========================
# FEATURE ENGINEERING
# =========================
def build_features(df):
    # 1. EMA & ATR
    df["ema9"] = df["CLOSE"].ewm(span=9, adjust=False).mean()
    df["ema21"] = df["CLOSE"].ewm(span=21, adjust=False).mean()
    
    # Rename kolom untuk konsistensi (Close, High, Low, Open)
    df.rename(columns={
        "OPEN":"Open", "HIGH":"High", "LOW":"Low", "CLOSE":"Close"
    }, inplace=True)

    # Hitung Average True Range (ATR)
    tr = np.maximum(df["High"] - df["Low"], 
                    np.maximum(abs(df["High"] - df["Close"].shift(1)), 
                            abs(df["Low"] - df["Close"].shift(1))))
    df["atr"] = tr.rolling(14).mean()
    df["ema_diff"] = (df["ema9"] - df["ema21"]) / df["atr"] 
    df["ema_trend"] = np.where(df["ema9"] > df["ema21"], 1, -1)

    # 2. SWING HIGH / LOW (harus dihitung dulu sebelum dist_to_hh/dist_to_ll)
    n = 5
    highs = df["High"].values
    lows  = df["Low"].values

    df["swing_high"] = np.nan
    df["swing_low"]  = np.nan

    hi_idx = argrelextrema(highs, np.greater_equal, order=n)[0]
    lo_idx = argrelextrema(lows,  np.less_equal,  order=n)[0]

    df.iloc[hi_idx, df.columns.get_loc("swing_high")] = df.iloc[hi_idx]["High"].values
    df.iloc[lo_idx, df.columns.get_loc("swing_low")]  = df.iloc[lo_idx]["Low"].values

    df["swing_high"] = df["swing_high"].ffill()
    df["swing_low"]  = df["swing_low"].ffill()

    # 3. FITUR STRUKTUR (sekarang aman karena swing_high/low sudah ada)
    df["dist_to_hh"] = (df["swing_high"] - df["Close"]) / df["atr"]
    df["dist_to_ll"] = (df["Close"] - df["swing_low"]) / df["atr"]
    df["momentum"] = df["Close"].pct_change(5)
    df["is_hh"] = (df["swing_high"] > df["swing_high"].shift(1)).astype(int)
    df["is_ll"] = (df["swing_low"] < df["swing_low"].shift(1)).astype(int)

    # 4. RANGE (untuk ML context)
    lookback = 20
    df["recent_high"] = df["High"].rolling(lookback).max()
    df["recent_low"]  = df["Low"].rolling(lookback).min()
    df["range"] = (df["recent_high"] - df["recent_low"]).replace(0, np.nan)

    # 5. TREND CONTEXT
    df["trend_bias"] = (
        (df["swing_high"] > df["swing_high"].shift(1)) &
        (df["swing_low"]  > df["swing_low"].shift(1))
    ).astype(int)

    # 6. PRICE LOCATION (discount/premium zone)
    df["price_pos"] = (df["Close"] - df["recent_low"]) / df["range"]
    df["price_pos"] = df["price_pos"].clip(0, 1)

    # 7. CHANGE OF CHARACTER (CHoCH)
    df["choch_up"] = (
        (df["Close"] > df["swing_high"].shift(1)) &
        (df["swing_low"] < df["swing_low"].shift(1))
    ).astype(int)

    df["choch_down"] = (
        (df["Close"] < df["swing_low"].shift(1)) &
        (df["swing_high"] > df["swing_high"].shift(1))
    ).astype(int)

    # 8. BREAK OF STRUCTURE (BOS)
    df["bos_up"] = (df["Close"] > df["swing_high"].shift(1)).astype(int)
    df["bos_down"] = (df["Close"] < df["swing_low"].shift(1)).astype(int)

    # BOS konfirmasi ke-2
    df["bos2_up"] = (
        (df["bos_up"] == 1) &
        (df["bos_up"].shift(1) == 1)
    ).astype(int)

    df["bos2_down"] = (
        (df["bos_down"] == 1) &
        (df["bos_down"].shift(1) == 1)
    ).astype(int)

    # 9. PULLBACK (setelah BOS konfirmasi)
    df["pullback_buy"] = (
        (df["bos2_up"] == 1) &
        (df["Close"] < df["High"].rolling(5).max())
    ).astype(int)

    df["pullback_sell"] = (
        (df["bos2_down"] == 1) &
        (df["Close"] > df["Low"].rolling(5).min())
    ).astype(int)

    # 10. FIBONACCI RETRACEMENT
    swing_range = df["swing_high"] - df["swing_low"]

    df["fib_50"] = df["swing_high"] - 0.5 * swing_range
    df["fib_618"] = df["swing_high"] - 0.618 * swing_range

    df["in_fib_buy"] = (
        (df["Close"] <= df["fib_50"]) &
        (df["Close"] >= df["fib_618"])
    ).astype(int)

    df["in_fib_sell"] = (
        (df["Close"] >= df["fib_50"]) &
        (df["Close"] <= df["fib_618"])
    ).astype(int)

    # 11. FITUR POLARITY (penting untuk multi-class classification)
    df["direction_score"] = df["choch_up"] - df["choch_down"]
    df["bos_score"] = df["bos2_up"] - df["bos2_down"]
    df["pullback_score"] = df["pullback_buy"] - df["pullback_sell"]

    return df


def build_target(df, horizon=50):
    """
    Target labeling: 0 = Hold, 1 = Buy, 2 = Sell
    Berdasarkan apakah TP tercapai sebelum SL
    """
    highs = df['High'].values
    lows = df['Low'].values
    tp_b, sl_b = df['tp_buy'].values, df['sl_buy'].values
    tp_s, sl_s = df['tp_sell'].values, df['sl_sell'].values
    ema9, ema21 = df['ema9'].values, df['ema21'].values
    
    targets = np.zeros(len(df))

    for i in range(len(df) - horizon):
        # Ambil data future High/Low untuk horizon N candle ke depan
        f_high = highs[i+1 : i+horizon]
        f_low  = lows[i+1 : i+horizon]

        is_uptrend = ema9[i] > ema21[i]
        is_downtrend = ema9[i] < ema21[i]

        # BUY signal (hanya jika uptrend)
        if is_uptrend:
            tp_b_idx = np.where(f_high >= tp_b[i])[0]
            sl_b_idx = np.where(f_low <= sl_b[i])[0]
            first_tp = tp_b_idx[0] if len(tp_b_idx) > 0 else 999
            first_sl = sl_b_idx[0] if len(sl_b_idx) > 0 else 999
            
            if first_tp < first_sl and first_tp != 999:
                targets[i] = 1
                continue

        # SELL signal (hanya jika downtrend)
        if is_downtrend:
            tp_s_idx = np.where(f_low <= tp_s[i])[0]
            sl_s_idx = np.where(f_high >= sl_s[i])[0]
            first_tp = tp_s_idx[0] if len(tp_s_idx) > 0 else 999
            first_sl = sl_s_idx[0] if len(sl_s_idx) > 0 else 999
            
            if first_tp < first_sl and first_tp != 999:
                targets[i] = 2

    df['target'] = targets
    return df


# =========================
# TRAIN
# =========================
def train():
    print("📥 Loading data...")
    df = load_all_csv()
    df = df[df["target"].notna()]

    features = [
        "direction_score", "bos_score", "pullback_score",
        "ema_diff", "ema_trend",
        "choch_up", "choch_down", "bos2_up", "bos2_down",
        "pullback_buy", "pullback_sell", "range", "trend_bias",
        "price_pos", "atr",
        "dist_to_hh", "dist_to_ll", "momentum", "is_hh", "is_ll"
    ]

    if USE_FIB:
        features += ["in_fib_buy", "in_fib_sell"]

    X = df[features]
    y = df["target"]

    print(f"📊 Samples: {len(df)} | Positive: {y.sum()}")
    print(df["target"].value_counts(dropna=False))
    print("\nNaN check:")
    print(df[features].isna().sum())

    # Split data (70% train, 30% test) - JANGAN shuffle untuk time series
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        shuffle=False
    )

    print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")
    print("Distribusi Target di Training:")
    print(y_train.value_counts(normalize=True))

    # XGBoost Classifier untuk 3 kelas (0, 1, 2)
    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=1500,
        max_depth=6,
        learning_rate=0.01,
        subsample=0.9,
        colsample_bytree=0.9,        
        eval_metric="logloss",
        tree_method="hist",
        # device="cuda",  # Uncomment jika pakai GPU
        random_state=42
    )

    print("\n🚀 Training...")
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    model.fit(X_train, y_train)

    # Evaluasi pada test set
    print("\n📈 Evaluation (TEST SET):")
    y_pred = model.predict(X_test, validate_features=False)
    print(classification_report(y_test, y_pred, zero_division=0))

    # Save model
    os.makedirs("models", exist_ok=True)
    name = "structure_fib_xgb.pkl" if USE_FIB else "structure_nonfib_xgb.pkl"
    joblib.dump(model, f"models/{name}")
    print(f"\n✅ Model saved: models/{name}")


if __name__ == "__main__":
    train()