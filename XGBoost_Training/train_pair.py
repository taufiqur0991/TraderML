import glob, os
import pandas as pd
import joblib
import xgboost as xgb
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from scipy.signal import argrelextrema
from sklearn.metrics import accuracy_score

def build_features(df, dxy_df=None):
    # --- Indikator Lokal (Pasti Ada) ---
    df['ema9'] = EMAIndicator(df['Close'], window=9).ema_indicator()
    df['ema21'] = EMAIndicator(df['Close'], window=21).ema_indicator()
    df['rsi'] = RSIIndicator(df['Close'], window=14).rsi()
    
    # Market Structure
    n = 5
    # Gunakan .values untuk kecepatan dan menghindari index mismatch saat pencarian peak
    high_vals = df['High'].values
    low_vals = df['Low'].values
    
    df['last_peak'] = df['High'].iloc[argrelextrema(high_vals, np.greater_equal, order=n)[0]]
    df['last_valley'] = df['Low'].iloc[argrelextrema(low_vals, np.less_equal, order=n)[0]]
    df['last_peak'] = df['last_peak'].ffill()
    df['last_valley'] = df['last_valley'].ffill()
    
    df['is_HH'] = (df['last_peak'] > df['last_peak'].shift(1)).astype(int)
    df['is_HL'] = (df['last_valley'] > df['last_valley'].shift(1)).astype(int)
    df['swing_range'] = (df['last_peak'] - df['last_valley']).replace(0, 0.0001)
    df['dist_fibo'] = df['Close'] - (df['last_peak'] - (0.618 * df['swing_range']))

    # --- INTEGRASI DXY DENGAN TOLERANSI ---
    if dxy_df is not None:
        # Gunakan 'nearest' agar jika selisih beberapa detik tetap nyambung
        df = pd.merge_asof(df.sort_index(), 
                           dxy_df[['CLOSE']].rename(columns={'CLOSE': 'dxy_price'}).sort_index(), 
                           left_index=True, right_index=True, 
                           direction='nearest', 
                           tolerance=pd.Timedelta('1hour')) # Toleransi 1 jam
        
        df['dxy_rsi'] = RSIIndicator(df['dxy_price'], window=14).rsi()
    else:
        df['dxy_price'] = df['Close']
        df['dxy_rsi'] = 50.0

    # Mengisi kekosongan tanpa menghapus baris (PENTING!)
    df['dxy_price'] = df['dxy_price'].ffill().bfill()
    df['dxy_rsi'] = df['dxy_rsi'].ffill().fillna(50)

    # --- Target ---
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # Dropna dilakukan TERAKHIR dan sangat hati-hati
    df.dropna(subset=['ema21', 'rsi', 'target'], inplace=True) 
    return df

def prepare_datetime(df):
    # Bersihkan kolom Date dan Time dari karakter aneh (seperti titik jadi strip)
    df['DATE'] = df['DATE'].astype(str).str.replace('.', '-', regex=False)
    df['DATETIME'] = pd.to_datetime(df['DATE'] + ' ' + df['TIME'], errors='coerce')
    df.set_index('DATETIME', inplace=True)
    df.sort_index(inplace=True)
    return df

def train_pair(pair_name):
    print(f"\n🚀 Training Spesifik untuk: {pair_name}")

    # 1. Load Data DXY (Sebagai Global Feature)
    dxy_df = None
    dxy_path = "data/DXY/DXY.csv" # Pastikan path data DXY benar
    if os.path.exists(dxy_path):
        dxy_df = pd.read_csv(dxy_path, sep="\t")
        dxy_df.columns = [c.replace("<", "").replace(">", "").upper() for c in dxy_df.columns]
        dxy_df = prepare_datetime(dxy_df)
        print(f"DEBUG: DXY Range: {dxy_df.index.min()} s/d {dxy_df.index.max()}")

    # 1. Load Data Pair
    path = f"data/{pair_name}/*.csv"
    files = glob.glob(path)
    
    if not files:
        print(f"Data untuk {pair_name} tidak ditemukan.")
        return

    dfs = [pd.read_csv(f, sep="\t") for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df.columns = [c.replace("<", "").replace(">", "").upper() for c in df.columns]
    df = prepare_datetime(df)
    print(f"DEBUG: {pair_name} Range: {df.index.min()} s/d {df.index.max()}")
    
    # Rename kolom agar standar
    df.rename(columns={'OPEN': 'Open', 'HIGH': 'High', 'LOW': 'Low', 'CLOSE': 'Close'}, inplace=True)

    # 2. Build Features
    df = build_features(df, dxy_df)
    print(f"DEBUG: Total baris setelah build_features & dropna: {len(df)}")
        
    features = ['ema9', 'ema21', 'rsi', 'is_HH', 'is_HL', 'dist_fibo', 'swing_range', 'dxy_price', 'dxy_rsi']
    X = df[features]
    y = df['target']

    # --- PENGECEKAN DATA ---
    unique_targets = y.unique()
    if len(unique_targets) < 2:
        print(f"⚠️ Skip {pair_name}: Data target tidak bervariasi (hanya ada {unique_targets}).")
        return

    if len(df) < 100:
        print(f"⚠️ Skip {pair_name}: Data terlalu sedikit setelah diproses ({len(df)} baris).")
        return

    # PENTING: Cek apakah X dan y masih aligned setelah dropna
    print(f"DEBUG: X shape: {X.shape}, y shape: {y.shape}")
    print(f"DEBUG: Features yang digunakan: {features}")
    print(f"DEBUG: Sample X (5 baris pertama):\n{X.head()}")
    
    # 3. XGBoost Model dengan early_stopping
    model = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=10,
        learning_rate=0.01,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric='logloss',
        tree_method='hist',
        random_state=42
    )

    print(f"🔄 Training model...")
    model.fit(X, y)
    
    # Verifikasi model setelah training
    print(f"DEBUG: Model n_estimators setelah fit: {model.n_estimators}")
    print(f"DEBUG: Model n_features_in_: {model.n_features_in_}")
    print(f"DEBUG: Model feature names: {model.feature_names_in_}")

    # 4. Save Model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, f"models/{pair_name}.pkl")
    
    acc = accuracy_score(y, model.predict(X))
    print(f"✅ Model {pair_name} saved. Accuracy: {acc:.2%}")
    
    # Test load model
    print(f"🔍 Testing load model...")
    loaded_model = joblib.load(f"models/{pair_name}.pkl")
    print(f"   Loaded model n_features: {loaded_model.n_features_in_}")
    print(f"   Loaded model n_estimators: {loaded_model.n_estimators}")