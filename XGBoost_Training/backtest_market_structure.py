import os, glob
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import mplfinance as mpf

from scipy.signal import argrelextrema

# ================= CONFIGURATION =================
DATA_DIR = "data_market_structure"
MODEL_PATH = "models/structure_nonfib_xgb.pkl"
OUTPUT_DIR = "backtest_market_structure"

# Threshold Simetris: 
# Buy jika prob > 0.65 | Sell jika prob < 0.35 (atau prob_sell > 0.65)
TH = 0.65             
INITIAL_BALANCE = 1000
LOT = 0.1               
SCREENSHOT_COUNT = 5    
RISK_REWARD = 1.1  # risk reward FIX
CONF_GAP = 0.10 # Batas minimal selisih probabilitas Buy/Sell

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_price_csv(path):
    df = pd.read_csv(path, sep="\t")

    # NORMALISASI KOLOM
    df.columns = (
        df.columns
        .str.strip()
        .str.upper()
    )

    if "DATETIME" not in df.columns:
        raise ValueError(f"DATETIME not found. Columns: {df.columns.tolist()}")

    df["DATETIME"] = pd.to_datetime(df["DATETIME"])
    df.set_index("DATETIME", inplace=True)

    return df


# =========================
# SL / TP STRUCTURE (RR)
# =========================
def build_sl_tp(df, rr=RISK_REWARD, atr_mult=1.2, min_sl_pips=10):
    """
    FINAL SL LOGIC:
    SL = max(
        structure/reject SL,
        ATR * atr_mult,
        min_sl_pips
    )
    TP = RR * SL
    """

    df = df.copy()

    # ===== ATR =====
    tr = np.maximum(
        df["HIGH"] - df["LOW"],
        np.maximum(
            abs(df["HIGH"] - df["CLOSE"].shift(1)),
            abs(df["LOW"] - df["CLOSE"].shift(1))
        )
    )
    df["ATR"] = tr.rolling(14).mean()

    # ===== STRUCTURE SL =====
    df["sl_buy_struct"]  = df["swing_low"]
    df["sl_sell_struct"] = df["swing_high"]

    # ===== REJECTION SL =====
    df["sl_buy_reject"]  = df["LOW"]
    df["sl_sell_reject"] = df["HIGH"]

    # ===== BASE SL (MODE) =====
    df["sl_buy_base"] = np.where(
        df["entry_mode"] == 1,
        df["sl_buy_reject"],
        df["sl_buy_struct"]
    )

    df["sl_sell_base"] = np.where(
        df["entry_mode"] == 1,
        df["sl_sell_reject"],
        df["sl_sell_struct"]
    )

    # ===== SL DISTANCE =====
    df["sl_dist_buy"]  = df["CLOSE"] - df["sl_buy_base"]
    df["sl_dist_sell"] = df["sl_sell_base"] - df["CLOSE"]

    # ===== ATR FLOOR =====
    df["sl_dist_buy"] = np.maximum(df["sl_dist_buy"], df["ATR"] * atr_mult)
    df["sl_dist_sell"] = np.maximum(df["sl_dist_sell"], df["ATR"] * atr_mult)

    # ===== FINAL SL PRICE =====
    df["sl_buy"]  = df["CLOSE"] - df["sl_dist_buy"]
    df["sl_sell"] = df["CLOSE"] + df["sl_dist_sell"]

    # ===== TP =====
    df["tp_buy"]  = df["CLOSE"] + rr * df["sl_dist_buy"]
    df["tp_sell"] = df["CLOSE"] - rr * df["sl_dist_sell"]

    return df



# ================= MODULAR PARAMETERS =================
def get_symbol_config(symbol_name):
    """
    Menyesuaikan nilai Pip dan Target berdasarkan simbol secara modular.
    """
    s = symbol_name.upper()
    
    # GOLD (XAUUSD)
    if "XAU" in s or "GOLD" in s:
        return 0.01, 10.0, 120, 240  # PipStep 0.1, Value $10, SL 40 pips, TP 120 pips
    
    # JPY PAIRS (USDJPY, EURJPY)
    elif "JPY" in s:
        return 0.01, 10.0, 40, 100   
    
    # FOREX MAJORS (EURUSD, GBPUSD)
    else:
        return 0.00001, 10.0, 40, 100

def parse_symbol_tf(filename):
    name = os.path.basename(filename).replace(".csv", "")
    parts = name.split("_")
    symbol = parts[0]
    tf = parts[1]
    return symbol, tf, f"{symbol}_{tf}"

# ================= FEATURE ENGINEERING =================
def build_features(df):
    df = df.copy()
    # 1. Dasar
    df["ema9"] = df["CLOSE"].ewm(span=9, adjust=False).mean()
    df["ema21"] = df["CLOSE"].ewm(span=21, adjust=False).mean()
    tr = np.maximum(df["HIGH"] - df["LOW"], 
                    np.maximum(abs(df["HIGH"] - df["CLOSE"].shift(1)), 
                               abs(df["LOW"] - df["CLOSE"].shift(1))))
    df["atr"] = tr.rolling(14).mean()
    df["ema_diff"] = (df["ema9"] - df["ema21"]) / df["atr"]
    df["ema_trend"] = np.where(df["ema9"] > df["ema21"], 1, -1)

    # 2. Swing (Wajib SEBELUM dist_to_hh)
    n = 5
    hi = argrelextrema(df["HIGH"].values, np.greater_equal, order=n)[0]
    lo = argrelextrema(df["LOW"].values, np.less_equal, order=n)[0]
    df["swing_high"] = np.nan
    df["swing_low"]  = np.nan
    df.iloc[hi, df.columns.get_loc("swing_high")] = df.iloc[hi]["HIGH"]
    df.iloc[lo, df.columns.get_loc("swing_low")]  = df.iloc[lo]["LOW"]
    df["swing_high"] = df["swing_high"].ffill()
    df["swing_low"]  = df["swing_low"].ffill()

    # 3. Fitur Struktur Baru
    df["dist_to_hh"] = (df["swing_high"] - df["CLOSE"]) / df["atr"]
    df["dist_to_ll"] = (df["CLOSE"] - df["swing_low"]) / df["atr"]
    df["momentum"] = df["CLOSE"].pct_change(5)
    df["is_hh"] = (df["swing_high"] > df["swing_high"].shift(1)).astype(int)
    df["is_ll"] = (df["swing_low"] < df["swing_low"].shift(1)).astype(int)

    # 4. Market Structure Score
    df["bos2_up"] = ((df["CLOSE"] > df["swing_high"].shift(1)) & (df["CLOSE"].shift(1) > df["swing_high"].shift(2))).astype(int)
    df["bos2_down"] = ((df["CLOSE"] < df["swing_low"].shift(1)) & (df["CLOSE"].shift(1) < df["swing_low"].shift(2))).astype(int)
    df["choch_up"] = ((df["bos2_up"] == 1) & (df["bos2_down"].shift(1) == 1)).astype(int)
    df["choch_down"] = ((df["bos2_down"] == 1) & (df["bos2_up"].shift(1) == 1)).astype(int)
    
    lookback = 20
    df["recent_high"] = df["HIGH"].rolling(lookback).max()
    df["recent_low"] = df["LOW"].rolling(lookback).min()
    df["range"] = (df["recent_high"] - df["recent_low"]).replace(0, np.nan)
    df["price_pos"] = (df["CLOSE"] - df["recent_low"]) / df["range"]
    
    df["trend_bias"] = df["is_hh"] - df["is_ll"]
    df["pullback_buy"] = ((df["ema_trend"] == 1) & (df["price_pos"] < 0.5)).astype(int)
    df["pullback_sell"] = ((df["ema_trend"] == -1) & (df["price_pos"] > 0.5)).astype(int)

    df["direction_score"] = df["choch_up"] - df["choch_down"]
    df["bos_score"] = df["bos2_up"] - df["bos2_down"]
    df["pullback_score"] = df["pullback_buy"] - df["pullback_sell"]

    return df


# ================= MAIN BACKTEST =================
FEATURES = [
    "direction_score", "bos_score", "pullback_score",
    "ema_diff", "ema_trend",
    "choch_up", "choch_down", "bos2_up", "bos2_down",
    "pullback_buy", "pullback_sell", "range", "trend_bias",
    "price_pos", "atr",
    "dist_to_hh", "dist_to_ll", "momentum", "is_hh", "is_ll"
]

if not os.path.exists(MODEL_PATH):
    print(f"❌ File model tidak ditemukan!")
else:
    model = joblib.load(MODEL_PATH)
    
    for file in glob.glob(f"{DATA_DIR}/*.csv"):
        symbol, tf, run_id = parse_symbol_tf(file)
        PIP_STEP, SYMBOL_PIP_VALUE, SL_PIPS, TP_PIPS = get_symbol_config(symbol)
        
        print(f"\n🚀 Backtesting {symbol} - {tf} ")        
                
        df_raw = load_price_csv(file)
        df = build_features(df_raw)
        # =========================
        # ENTRY MODE (SAMA DENGAN TRAINING)
        # =========================
        df["entry_mode"] = np.where(
            (df["choch_up"] == 1) & (df["pullback_buy"] == 1),
            1,
            0
        )
        df = build_sl_tp(df, rr=RISK_REWARD)

        # print("COLUMNS:", df_raw.columns.tolist())
        # print("NaN check:")
        # print(df[FEATURES].isna().sum())
        
        # Prediksi Probabilitas
        X = df[FEATURES]
        proba = model.predict_proba(X)
        df["prob_loss"] = proba[:, 0] # Kelas 0
        df["prob_buy"]  = proba[:, 1] # Kelas 1
        df["prob_sell"] = proba[:, 2] # Kelas 2
        
        # Inisialisasi Kolom Trading
        df['result_pips'] = 0.0
        df['exit_type'] = 'NONE'
        df['side'] = 'NONE'
        df['entry_price'] = np.nan
        df['exit_price'] = np.nan
        df['exit_index_val'] = np.nan

        # Loop Transaksi
        i = 0
        while i < len(df)-1:
            p_buy  = df.iloc[i]["prob_buy"]
            p_sell = df.iloc[i]["prob_sell"]
            current_ema_trend = df.iloc[i]["ema_trend"] # Ambil trend saat ini

            side = "NONE"
            # BUY hanya jika prob tinggi DAN EMA trend naik (1)
            if p_buy >= TH and (p_buy - p_sell) >= CONF_GAP and current_ema_trend == 1:
                side = "BUY"
            # SELL hanya jika prob tinggi DAN EMA trend turun (-1)
            elif p_sell >= TH and (p_sell - p_buy) >= CONF_GAP and current_ema_trend == -1:
                side = "SELL"
            
            # Jika tidak ada sinyal, lanjut ke baris berikutnya
            if side == "NONE":
                i += 1
                continue
            
            entry_idx = i + 1
            entry_p = df.iloc[entry_idx]["OPEN"]
            
            # SL/TP HARUS KONSISTEN DENGAN ENTRY
            
            if side == "BUY":
                sl_p = df.iloc[entry_idx]["sl_buy"]
                tp_p = df.iloc[entry_idx]["tp_buy"]
            else: # SELL
                sl_p = df.iloc[entry_idx]["sl_sell"]
                tp_p = df.iloc[entry_idx]["tp_sell"]
            
            # PROTEKSI INVALID
            if np.isnan(sl_p) or np.isnan(tp_p):
                i += 1
                continue

            found_exit = False
            for j in range(entry_idx, len(df)):
                high_j, low_j = df.iloc[j]["HIGH"], df.iloc[j]["LOW"]
                
                if side == "BUY":
                    if low_j <= sl_p:
                        res = -((entry_p - sl_p) / PIP_STEP)
                        etype, exit_p = 'SL', sl_p
                        found_exit = True
                    elif high_j >= tp_p:
                        res = (tp_p - entry_p) / PIP_STEP
                        etype, exit_p = 'TP', tp_p
                        found_exit = True
                
                elif side == "SELL":
                    if high_j >= sl_p:
                        res = -((sl_p - entry_p) / PIP_STEP)
                        etype, exit_p = 'SL', sl_p
                        found_exit = True
                    elif low_j <= tp_p:
                        res = (entry_p - tp_p) / PIP_STEP
                        etype, exit_p = 'TP', tp_p
                        found_exit = True
                
                if found_exit:
                    idx_label = df.index[i]
                    df.at[idx_label, 'result_pips'] = res
                    df.at[idx_label, 'exit_type'] = etype
                    df.at[idx_label, 'side'] = side
                    df.at[idx_label, 'entry_price'] = entry_p
                    df.at[idx_label, 'exit_price'] = exit_p
                    df.at[idx_label, 'exit_index_val'] = j
                    i = j  # Lompat ke baris saat exit terjadi
                    break
            
            if not found_exit:
                i += 1

        run_dir = os.path.join(OUTPUT_DIR, run_id)
        os.makedirs(run_dir, exist_ok=True)   
        # ================= VISUALISASI & SUMMARY =================
        trades = df[df['exit_type'] != 'NONE'].copy()
        
        if len(trades) > 0:
            current_bal = INITIAL_BALANCE
            bal_hist, time_hist = [INITIAL_BALANCE], [df.index[0]]
            
            for idx, row in trades.iterrows():
                profit = row['result_pips'] * SYMBOL_PIP_VALUE * LOT
                current_bal += profit
                bal_hist.append(current_bal)
                time_hist.append(idx)
            
            # Plot Equity Curve
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
            ax1.plot(time_hist, bal_hist, color='#0000FF', label='Balance')
            ax1.set_title(f"Equity {symbol} | B:{len(trades[trades['side']=='BUY'])} S:{len(trades[trades['side']=='SELL'])}")
            
            # Drawdown
            bal_arr = np.array(bal_hist)
            peak = np.maximum.accumulate(bal_arr)
            dd = np.divide((peak - bal_arr), peak, out=np.zeros_like(peak), where=peak!=0)
            ax2.fill_between(time_hist, dd * 100, 0, color='#00FF00', alpha=0.6)
            ax2.set_ylabel("DD %")
            
            plt.tight_layout()            
            plt.savefig(os.path.join(run_dir, f"equity_{run_id}.png"))
            plt.close()

            # Screenshot Trade            
         
            samples = trades.sample(min(len(trades), SCREENSHOT_COUNT))
            
            for t_id, (idx, row) in enumerate(samples.iterrows(), 1):
                e_pos = df.index.get_loc(idx) + 1
                x_pos = int(row['exit_index_val'])
                chart_df = df.iloc[
                    max(0, e_pos-20):min(len(df), x_pos+20)
                ][['OPEN','HIGH','LOW','CLOSE']].copy()

                chart_df.rename(columns={
                    "OPEN": "Open",
                    "HIGH": "High",
                    "LOW": "Low",
                    "CLOSE": "Close"
                }, inplace=True)
                
                # Markers
                b_m, s_m, x_m = [np.full(len(chart_df), np.nan) for _ in range(3)]
                rel_e = e_pos - df.index.get_loc(chart_df.index[0])
                rel_x = x_pos - df.index.get_loc(chart_df.index[0])
                
                if row['side'] == 'BUY': b_m[rel_e] = row['entry_price']
                else: s_m[rel_e] = row['entry_price']
                x_m[rel_x] = row['exit_price']

                apds = [
                    mpf.make_addplot(b_m, type='scatter', marker='^', markersize=100, color='green'),
                    mpf.make_addplot(s_m, type='scatter', marker='v', markersize=100, color='red'),
                    mpf.make_addplot(x_m, type='scatter', marker='x', markersize=80, color='black')
                ]
                apds = [a for a in apds if not np.all(np.isnan(a['data']))]
                if len(chart_df) < 1500:
                    mpf.plot(chart_df, type='candle', style='charles', addplot=apds,
                            title=f"{symbol} | {row['side']} {int(row['result_pips'])} Pips #{t_id}",
                            savefig=os.path.join(run_dir, f"trade_{t_id}.png"))                            
                else:
                    mpf.plot(chart_df, type='line', style='charles', addplot=apds,
                            title=f"{symbol} | {row['side']} {int(row['result_pips'])} Pips #{t_id}",
                            savefig=os.path.join(run_dir, f"trade_{t_id}.png"))

            print(f"✅ Final Balance: ${round(current_bal, 2)} | Winrate: {round((trades['result_pips']>0).mean()*100, 1)}%")
        else:
            print(f"⚠️ Tidak ada trade yang memenuhi threshold {TH}")

    print("\n✨ Selesai! Cek folder:", OUTPUT_DIR)