import glob, os
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

RR_RATIO = 1.5
HORIZON = 20

def build_larry_features(df):
    # 1. Bersihkan nama kolom dari spasi dan ubah ke kapital
    df.columns = [c.strip().upper() for c in df.columns]
    
    # Debug: Print kolom untuk memastikan (opsional)
    # print("Kolom terdeteksi:", df.columns.tolist())

    # 2. Larry Williams Logic (Outside Bar)
    # Gunakan nama kolom kapital sesuai hasil strip()
    df['is_outside_bull'] = ((df['LOW'] < df['LOW'].shift(1)) & 
                             (df['HIGH'] > df['HIGH'].shift(1)) & 
                             (df['CLOSE'] > df['HIGH'].shift(1))).astype(int)

    df['is_outside_bear'] = ((df['HIGH'] > df['HIGH'].shift(1)) & 
                             (df['LOW'] < df['LOW'].shift(1)) & 
                             (df['CLOSE'] < df['LOW'].shift(1))).astype(int)

    # 3. Fitur Pendukung
    tr = np.maximum(df["HIGH"] - df["LOW"], 
                    np.maximum(abs(df["HIGH"] - df["CLOSE"].shift(1)), 
                               abs(df["LOW"] - df["CLOSE"].shift(1))))
    df["ATR"] = tr.rolling(14).mean()
    
    df["BODY_SIZE"] = abs(df["CLOSE"] - df["OPEN"]) / (df["HIGH"] - df["LOW"] + 1e-9)
    # Relatif range 10 candle terakhir
    df["REL_RANGE"] = (df["HIGH"] - df["LOW"]) / (df["HIGH"].rolling(10).max() - df["LOW"].rolling(10).min() + 1e-9)
    # Tambahkan fitur hari (0=Senin, 4=Jumat)
    # Pola Larry Williams sering beda performanya di awal/akhir pekan
    df['DAY'] = pd.to_datetime(df['DATETIME']).dt.dayofweek

    # Tambahkan fitur volatility relatif (apakah candle signal jauh lebih besar dari rata-rata?)
    df['RELATIVE_ATR'] = (df['HIGH'] - df['LOW']) / df['ATR']

    # Tambahkan trend barisan (apakah 3 candle sebelumnya naik semua atau turun semua?)
    df['PREV_DIR'] = np.where(df['CLOSE'].shift(1) > df['OPEN'].shift(1), 1, -1)
    df['VOL_CHANGE'] = df['VOLUME'].pct_change() # Apakah volume naik saat Outside Bar?
    df['HOUR'] = pd.to_datetime(df['DATETIME']).dt.hour # Jika data < Daily
    
    return df

def label_larry_strategy(df):
    df['target'] = 0
    # Pastikan ambil nilai dalam huruf kapital
    h, l, c, o = df['HIGH'].values, df['LOW'].values, df['CLOSE'].values, df['OPEN'].values
    bull, bear = df['is_outside_bull'].values, df['is_outside_bear'].values
    
    for i in range(len(df) - HORIZON):
        entry_price = o[i+1] # Entry di open bar berikutnya
        
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
    all_files = glob.glob("data_larry_williams/*.csv")
    if not all_files:
        print("Error: Folder 'data' kosong atau tidak ditemukan!")
        return

    data_list = []
    for f in all_files:
        print(f"Reading: {os.path.basename(f)}")
        # sep=None dengan engine='python' akan otomatis mendeteksi Tab atau Koma
        df = pd.read_csv(f, sep=None, engine='python')
        
        try:
            df = build_larry_features(df)
            df = label_larry_strategy(df)
            data_list.append(df)
        except KeyError as e:
            print(f"Skip file {f} karena error kolom: {e}")
            continue

    if not data_list: return
    
    full_df = pd.concat(data_list).dropna()
    
    # Filter: Hanya latih data yang ada signal Larry Williams
    train_df = full_df[(full_df['is_outside_bull'] == 1) | (full_df['is_outside_bear'] == 1)].copy()
    
    features = ["BODY_SIZE", "REL_RANGE", "is_outside_bull", "is_outside_bear"]
    X = train_df[features]
    y = train_df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = xgb.XGBClassifier(
        objective="multi:softprob", 
        num_class=3, 
        learning_rate=0.05, # Naikkan sedikit agar tidak terlalu lambat
        max_depth=4,        # JAUH LEBIH AMAN. Mencegah hafalan (overfitting)
        n_estimators=400,   # Cukup 300 untuk menangkap pola utama
        subsample=0.7,
        colsample_bytree=0.7,
        gamma=0.1,          # Menambahkan penalti agar model tidak ceroboh
        tree_method="hist", 
        eval_metric="logloss",
        min_child_weight=5,
        reg_lambda=1.5,
        reg_alpha=0.1,
        random_state=42
    )
    # model = xgb.XGBClassifier(
    #     objective="multi:softprob", 
    #     num_class=3, 
    #     learning_rate=0.01,
    #     max_depth=9,            
    #     n_estimators=1500,      
    #     subsample=0.9,          
    #     colsample_bytree=0.9,   
    #     tree_method="hist",   
    #     random_state=42
    # )
    model.fit(X_train, y_train)

    print("\n--- PERFORMANCE REPORT ---")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    joblib.dump(model, "models/larry_model.pkl")
    print("✅ Model Saved!")

if __name__ == "__main__":
    train_model()