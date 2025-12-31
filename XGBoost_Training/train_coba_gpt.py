import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score
from sklearn.utils.class_weight import compute_sample_weight

# ==========================================================
# CONFIG (WAJIB PAHAM)
# ==========================================================
HORIZON = 30
WINDOW_SWING = 10
RR_TARGET = 2.5
MAX_DRAWDOWN_RATIO = 0.3

# ==========================================================
# CANDLESTICK PATTERN
# ==========================================================
def detect_candles(df):
    body = abs(df['Close'] - df['Open'])
    rng = df['High'] - df['Low'] + 1e-9
    up = df['High'] - df[['Close','Open']].max(axis=1)
    lo = df[['Close','Open']].min(axis=1) - df['Low']

    df['bull_engulf'] = (
        (df['Close'] > df['Open']) &
        (df['Close'].shift(1) < df['Open'].shift(1)) &
        (df['Close'] > df['Open'].shift(1)) &
        (df['Open'] < df['Close'].shift(1))
    ).astype(int)

    df['bear_engulf'] = (
        (df['Close'] < df['Open']) &
        (df['Close'].shift(1) > df['Open'].shift(1)) &
        (df['Close'] < df['Open'].shift(1)) &
        (df['Open'] > df['Close'].shift(1))
    ).astype(int)

    df['pin_up'] = ((lo > rng*0.6) & (body < rng*0.3)).astype(int)
    df['pin_dn'] = ((up > rng*0.6) & (body < rng*0.3)).astype(int)

    return df

# ==========================================================
# FEATURE ENGINEERING
# ==========================================================
def build_features(df):
    df.columns = [c.strip().capitalize() for c in df.columns]

    df['ema20']  = df['Close'].ewm(span=20).mean()
    df['ema50']  = df['Close'].ewm(span=50).mean()
    df['ema200'] = df['Close'].ewm(span=200).mean()

    df['uptrend']   = ((df['ema20'] > df['ema50']) & (df['ema50'] > df['ema200'])).astype(int)
    df['downtrend'] = ((df['ema20'] < df['ema50']) & (df['ema50'] < df['ema200'])).astype(int)

    df['sw_high'] = df['High'].rolling(WINDOW_SWING, center=True).max()
    df['sw_low']  = df['Low'].rolling(WINDOW_SWING, center=True).min()

    last_h = df['High'].where(df['High']==df['sw_high']).ffill()
    last_l = df['Low'].where(df['Low']==df['sw_low']).ffill()

    df['fib'] = (df['Close'] - last_l) / (last_h - last_l + 1e-9)

    df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
    df['vol_shock'] = df['ATR'] / (df['ATR'].shift(10) + 1e-9)

    df = detect_candles(df)
    return df.fillna(0)

# ==========================================================
# LABELING KETAT (INTI PROFIT)
# ==========================================================
def label_data(df):
    df['target'] = 0

    H = df['High'].values
    L = df['Low'].values
    C = df['Close'].values

    for i in range(len(df)-HORIZON):

        # ===== BUY SETUP =====
        if df.at[i,'uptrend'] and 0.5 <= df.at[i,'fib'] <= 0.8:
            if df.at[i,'bull_engulf'] or df.at[i,'pin_up']:
                entry = C[i]
                sl = df.at[i,'sw_low']
                risk = max(entry-sl, df.at[i,'ATR']*1.5)
                tp = entry + RR_TARGET*risk

                max_dd = 0
                for j in range(i+1,i+HORIZON):
                    max_dd = max(max_dd, (entry-L[j])/risk)
                    if L[j] <= entry-risk: break
                    if H[j] >= tp and max_dd <= MAX_DRAWDOWN_RATIO:
                        df.at[i,'target'] = 1
                        break

        # ===== SELL SETUP =====
        if df.at[i,'downtrend'] and 0.2 <= df.at[i,'fib'] <= 0.5:
            if df.at[i,'bear_engulf'] or df.at[i,'pin_dn']:
                entry = C[i]
                sl = df.at[i,'sw_high']
                risk = max(sl-entry, df.at[i,'ATR']*1.5)
                tp = entry - RR_TARGET*risk

                max_dd = 0
                for j in range(i+1,i+HORIZON):
                    max_dd = max(max_dd, (H[j]-entry)/risk)
                    if H[j] >= entry+risk: break
                    if L[j] <= tp and max_dd <= MAX_DRAWDOWN_RATIO:
                        df.at[i,'target'] = 2
                        break
    return df

# ==========================================================
# TRAINING
# ==========================================================
def train():
    files = glob.glob("data_trading/*.csv")
    dfs = []

    for f in files:
        df = pd.read_csv(f)
        df = build_features(df)
        df = label_data(df)
        dfs.append(df)

    data = pd.concat(dfs)

    FEATURES = [
        'uptrend','downtrend','fib',
        'bull_engulf','bear_engulf',
        'pin_up','pin_dn',
        'vol_shock','ATR'
    ]

    X = data[FEATURES]
    y = data['target']

    Xtr, Xts, ytr, yts = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    weights = compute_sample_weight('balanced', ytr)

    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=2,
        tree_method="hist",
        random_state=42
    )

    print("🚀 Training...")
    model.fit(Xtr, ytr, sample_weight=weights)

    pred = model.predict(Xts)

    print("\n--- EVALUATION ---")
    print(classification_report(yts, pred))

    entry_mask = pred != 0
    if entry_mask.sum() > 0:
        print("🎯 Precision ENTRY:",
              precision_score(yts[entry_mask], pred[entry_mask], average='macro'))

    joblib.dump(model,"pullback_xgb_clean.pkl")
    print("✅ Model disimpan")

# ==========================================================
if __name__ == "__main__":
    train()
