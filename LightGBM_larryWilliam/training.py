import glob, os
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

RR_RATIO = 1.5
HORIZON = 20

def build_larry_features(df):
    df.columns = [c.strip().upper() for c in df.columns]
    
    # --- Outside Bar Dasar ---
    df['is_outside_bull'] = ((df['LOW'] < df['LOW'].shift(1)) & 
                             (df['HIGH'] > df['HIGH'].shift(1)) & 
                             (df['CLOSE'] > df['HIGH'].shift(1))).astype(int)
    df['is_outside_bear'] = ((df['HIGH'] > df['HIGH'].shift(1)) & 
                             (df['LOW'] < df['LOW'].shift(1)) & 
                             (df['CLOSE'] < df['LOW'].shift(1))).astype(int)

    # --- ATR Dasar ---
    tr = np.maximum(df["HIGH"] - df["LOW"], 
                    np.maximum(abs(df["HIGH"] - df["CLOSE"].shift(1)), 
                               abs(df["LOW"] - df["CLOSE"].shift(1))))
    df["ATR"] = tr.rolling(14).mean() + 1e-9

    # --- RSI (Mengukur Jenuh Harga) ---
    delta = df['CLOSE'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))

    # --- ADX (Fixed: Tetap Menggunakan Pandas Series) ---
    plus_dm_raw = (df['HIGH'] - df['HIGH'].shift(1))
    minus_dm_raw = (df['LOW'].shift(1) - df['LOW'])
    
    plus_dm = plus_dm_raw.where((plus_dm_raw > minus_dm_raw) & (plus_dm_raw > 0), 0)
    minus_dm = minus_dm_raw.where((minus_dm_raw > plus_dm_raw) & (minus_dm_raw > 0), 0)
    
    # Sekarang .rolling() akan bekerja karena ini adalah Pandas Series
    df['ADX'] = (abs(plus_dm - minus_dm) / (plus_dm + minus_dm + 1e-9)).rolling(14).mean() * 100

    # --- Trend Filter (EMA 200) ---
    df['EMA_200'] = df['CLOSE'].ewm(span=200, adjust=False).mean()
    df['DIST_EMA_200'] = (df['CLOSE'] - df['EMA_200']) / df['ATR']

    # --- Body & Range ---
    df["BODY_SIZE"] = abs(df["CLOSE"] - df["OPEN"]) / (df['HIGH'] - df['LOW'] + 1e-9)
    df["RELATIVE_ATR"] = (df['HIGH'] - df['LOW']) / df['ATR']
    
    df['DATETIME'] = pd.to_datetime(df['DATETIME'])
    df['HOUR'] = df['DATETIME'].dt.hour
    df['DAY'] = df['DATETIME'].dt.dayofweek
    
    return df

def label_larry_strategy(df):
    df['target'] = 0
    h, l, c, o = df['HIGH'].values, df['LOW'].values, df['CLOSE'].values, df['OPEN'].values
    bull, bear = df['is_outside_bull'].values, df['is_outside_bear'].values
    
    for i in range(len(df) - HORIZON):
        entry_price = o[i+1]
        if bull[i] == 1:
            sl = l[i]
            tp = entry_price + (entry_price - sl) * RR_RATIO
            for j in range(i+1, i+HORIZON):
                if l[j] <= sl: break 
                if h[j] >= tp:
                    df.at[df.index[i], 'target'] = 1
                    break
        elif bear[i] == 1:
            sl = h[i]
            tp = entry_price - (sl - entry_price) * RR_RATIO
            for j in range(i+1, i+HORIZON):
                if h[j] >= sl: break
                if l[j] <= tp:
                    df.at[df.index[i], 'target'] = 2
                    break
    return df

def train_model():
    all_files = glob.glob("data/*.csv")
    if not all_files:
        print("Error: Folder 'data' kosong!")
        return

    data_list = []
    for f in all_files:
        print(f"Reading: {os.path.basename(f)}")
        df = pd.read_csv(f, sep=None, engine='python')
        try:
            df = build_larry_features(df)
            df = label_larry_strategy(df)
            data_list.append(df)
        except Exception as e:
            print(f"Skip file {f}: {e}")
            continue

    if not data_list: return
    full_df = pd.concat(data_list).dropna()
    
    train_df = full_df[(full_df['is_outside_bull'] == 1) | (full_df['is_outside_bear'] == 1)].copy()
    
    features = ["BODY_SIZE", "RELATIVE_ATR", "RSI", "ADX", "DIST_EMA_200", 
                "HOUR", "DAY", "is_outside_bull", "is_outside_bear"]
    
    X = train_df[features]
    y = train_df['target'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)    

    # Tuning: Learning Rate lebih kecil, Estimators lebih banyak
    model = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=3,
        n_estimators=3000,
        learning_rate=0.01,
        max_depth=5,
        num_leaves=20,
        min_child_samples=100,
        class_weight='balanced',
        random_state=42,
        importance_type='gain',
        force_col_wise=True,
        verbosity=-1
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(stopping_rounds=100)]
    )

    print("\n--- PERFORMANCE REPORT (FIXED) ---")
    y_probs = model.predict_proba(X_test)
    
    final_preds = []
    for p in y_probs:
        if p[1] > 0.50:
            final_preds.append(1)
        elif p[2] > 0.50:
            final_preds.append(2)
        else:
            final_preds.append(0)
            
    print(classification_report(y_test, final_preds))
    
    joblib.dump(model, "models/larry_model_lgbm.pkl")
    print("Model LightGBM Berhasil Disimpan!")

if __name__ == "__main__":
    train_model()