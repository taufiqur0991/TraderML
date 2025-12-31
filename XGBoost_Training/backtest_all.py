import os
import glob
import pandas as pd
import numpy as np
import joblib
import mplfinance as mpf
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt

# =====================================================
# CONFIGURATION
# =====================================================
DATA_DIR = "data"
DXY_PATH = "data/DXY/DXY.csv" # Pastikan path ini benar
MODEL_DIR = "models"
OUTPUT_BASE = "backtest"
TH = 0.60 # Threshold probabilitas
INITIAL_BALANCE = 1000
PIP_VALUE = 10 
FIXED_LOT = 0.01
SCREENSHOT_COUNT = 10

os.makedirs(OUTPUT_BASE, exist_ok=True)

def get_pair_params(filename):
    if "XAUUSD" in filename.upper():
        return 0.01, 1, 20, 40 # Gold SL 20, TP 40 (RR 1:2)
    return 0.00001, 10000, 15, 30 # Forex SL 15, TP 30

def build_features_for_backtest(df, dxy_df=None):
    # Identik dengan kode Training
    df['ema9'] = EMAIndicator(df['Close'], 9).ema_indicator()
    df['ema21'] = EMAIndicator(df['Close'], 21).ema_indicator()
    df['rsi'] = RSIIndicator(df['Close'], 14).rsi()
    
    n = 5
    df['peak'] = df['High'].iloc[argrelextrema(df['High'].values, np.greater_equal, order=n)[0]]
    df['valley'] = df['Low'].iloc[argrelextrema(df['Low'].values, np.less_equal, order=n)[0]]
    df['last_peak'] = df['peak'].ffill()
    df['last_valley'] = df['valley'].ffill()
    
    df['is_HH'] = (df['last_peak'] > df['last_peak'].shift(n+1)).astype(int)
    df['is_HL'] = (df['last_valley'] > df['last_valley'].shift(n+1)).astype(int)
    
    df['swing_range'] = (df['last_peak'] - df['last_valley']).replace(0, 0.0001)
    df['fibo_618'] = df['last_peak'] - (0.618 * df['swing_range'])
    df['dist_fibo'] = df['Close'] - df['fibo_618']

    if dxy_df is not None:
        df = pd.merge_asof(df.sort_index(), 
                           dxy_df[['CLOSE']].rename(columns={'CLOSE': 'dxy_price'}).sort_index(), 
                           left_index=True, right_index=True, direction='nearest', 
                           tolerance=pd.Timedelta('1hour'))
        df['dxy_rsi'] = RSIIndicator(df['dxy_price'], window=14).rsi().fillna(50)
    else:
        df['dxy_price'], df['dxy_rsi'] = 100.0, 50.0

    df.ffill(inplace=True); df.bfill(inplace=True)
    return df

# 1. LOAD DXY SEKALI SAJA
dxy_df = None
if os.path.exists(DXY_PATH):
    dxy_df = pd.read_csv(DXY_PATH, sep="\t")
    dxy_df.columns = [c.replace("<", "").replace(">", "").upper() for c in dxy_df.columns]
    dxy_df['Datetime'] = pd.to_datetime(dxy_df['DATE'] + ' ' + dxy_df['TIME'].fillna('00:00:00'))
    dxy_df.set_index('Datetime', inplace=True)

# 2. PROSES SEMUA CSV
csv_files = []
for root, _, files in os.walk(DATA_DIR):
    for f in files:
        if f.endswith('.csv') and "DXY" not in f:
            csv_files.append(os.path.join(root, f))

for full_path in csv_files:
    file_name = os.path.basename(full_path)
    pair_name = file_name.split('_')[0].split('.')[0]
    POINT, PIP, SL_PIPS, TP_PIPS = get_pair_params(file_name)
    
    # LOAD & PREPARE
    df_raw = pd.read_csv(full_path, sep="\t")
    df_raw.columns = [c.replace("<", "").replace(">", "").capitalize() for c in df_raw.columns]
    df_raw['Datetime'] = pd.to_datetime(df_raw['Date'] + ' ' + df_raw['Time'])
    df_raw.set_index('Datetime', inplace=True)
    
    # BUILD FEATURES
    df = build_features_for_backtest(df_raw.copy(), dxy_df)
    
    # LOAD MODEL
    model_path = os.path.join(MODEL_DIR, f"{pair_name}.pkl")
    if not os.path.exists(model_path): continue
    model = joblib.load(model_path)
    
    # PREDICT
    # Gunakan urutan kolom yang SAMA dengan saat training
    features = ['ema9', 'ema21', 'rsi', 'is_HH', 'is_HL', 'dist_fibo', 'swing_range', 'dxy_price', 'dxy_rsi']
    X = df[features]
    proba = model.predict_proba(X)
    
    df['prob_buy'], df['prob_sell'] = proba[:, 1], proba[:, 0]
    df['signal'] = 'NO_TRADE'
    df.loc[df['prob_buy'] > TH, 'signal'] = 'BUY'
    df.loc[df['prob_sell'] > TH, 'signal'] = 'SELL'

    # 4. FAST BACKTEST
    df['result_pips'] = 0.0
    df['exit_type'] = 'NONE'
    df['entry_price'] = np.nan
    df['exit_price'] = np.nan
    df['exit_index_val'] = np.nan

    sig_indices = np.where(df['signal'] != 'NO_TRADE')[0]
    high_arr, low_arr, open_arr, spread_arr = df['High'].values, df['Low'].values, df['Open'].values, df['Spread'].values

    for i in sig_indices:
        if i >= len(df) - 1: continue
        entry_idx = i + 1
        sig = df.iloc[i]['signal']
        spread = spread_arr[entry_idx] * POINT
        
        if sig == 'BUY':
            entry_p = open_arr[entry_idx] + spread
            sl_p, tp_p = entry_p - SL_PIPS/PIP, entry_p + TP_PIPS/PIP
            f_low, f_high = low_arr[entry_idx:], high_arr[entry_idx:]
            sl_hit, tp_hit = np.where(f_low <= sl_p)[0], np.where(f_high >= tp_p)[0]
        else:
            entry_p = open_arr[entry_idx]
            sl_p, tp_p = entry_p + SL_PIPS/PIP, entry_p - TP_PIPS/PIP
            f_high, f_low = high_arr[entry_idx:], low_arr[entry_idx:]
            sl_hit, tp_hit = np.where(f_high >= sl_p)[0], np.where(f_low <= tp_p)[0]

        f_sl = sl_hit[0] if len(sl_hit) > 0 else 999999
        f_tp = tp_hit[0] if len(tp_hit) > 0 else 999999
        if f_sl == 999999 and f_tp == 999999: continue
        
        if f_sl < f_tp:
            res, etype, eidx, ex_p = -SL_PIPS, 'SL', entry_idx + f_sl, sl_p
        else:
            res, etype, eidx, ex_p = TP_PIPS, 'TP', entry_idx + f_tp, tp_p

        idx_label = df.index[i]
        df.at[idx_label, 'result_pips'], df.at[idx_label, 'exit_type'] = res, etype
        df.at[idx_label, 'entry_price'], df.at[idx_label, 'exit_price'] = entry_p, ex_p
        df.at[idx_label, 'exit_index_val'] = eidx

    # 5. SUMMARY & BALANCE SIMULATION PER PAIR
    trades = df[df['exit_type'] != 'NONE'].copy()
    
    print(f"\n=== RESULT FOR {pair_name} ===")
    if len(trades) > 0:
        current_balance = INITIAL_BALANCE
        equity_list = [INITIAL_BALANCE]
        balance_list = [INITIAL_BALANCE]
        # Gunakan waktu awal data sebagai titik nol
        time_list = [df.index[0]] 
        
        for idx, row in trades.iterrows():
            lot = FIXED_LOT
            pips = row['result_pips']
            profit = pips * PIP_VALUE * lot
            
            # Simulasi Equity (spike) vs Balance (garis lurus)
            # Dalam MT5, Equity turun dulu saat floating, baru balance update saat close
            current_balance += profit
            
            balance_list.append(current_balance)
            equity_list.append(current_balance) # Untuk simulasi sederhana
            time_list.append(idx)
        
        balance_arr = np.array(balance_list)
        peak = np.maximum.accumulate(balance_arr)
        drawdown_pct = (peak - balance_arr) / peak

        # --- PLOTTING ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), 
                                       gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        
        # Plot 1: Balance (Biru) & Drawdown Spikes (Hijau)
        # Sumbu X: time_list, Sumbu Y: balance_arr -> Panjangnya sekarang PASTI sama
        ax1.plot(time_list, balance_arr, color='#0000FF', label='Balance', linewidth=1.5)
        
        # Simulasi spike drawdown hijau seperti di gambar
        # Kita buat garis vertikal tipis untuk setiap trade yang rugi
        for i in range(len(balance_list)):
            if i > 0 and balance_list[i] < balance_list[i-1]:
                ax1.vlines(time_list[i], balance_list[i], balance_list[i-1], color='#008000', alpha=0.7)

        ax1.set_title(f"Strategy Tester: {pair_name}", fontsize=12, loc='left')
        ax1.set_ylabel("Value")
        ax1.grid(True, which='both', linestyle='--', alpha=0.5)
        ax1.legend(loc='upper left')

        # Plot 2: Drawdown / Deposit Load (Hijau Bawah)
        ax2.fill_between(time_list, drawdown_pct * 100, 0, color='#00FF00', alpha=0.6)
        ax2.set_ylabel("Drawdown %")
        ax2.set_ylim(0, max(drawdown_pct * 100) * 1.5 if max(drawdown_pct) > 0 else 5)
        ax2.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_BASE, f"Chart_{pair_name}.png"))
        
        # Output text tetap sama
        print(f"Total Trades    : {len(trades)}")
        print(f"Total Pips      : {round(trades['result_pips'].sum(), 2)}")
        print(f"Winrate         : {round((trades['result_pips'] > 0).mean() * 100, 2)} %")
        print("\n=== BALANCE SIMULATION ===")
        print(f"Initial Balance : {INITIAL_BALANCE}")
        print(f"Final Balance   : {round(current_balance, 2)}")
        print(f"Net Profit      : {round(current_balance - INITIAL_BALANCE, 2)}")
        print(f"Max Drawdown    : {round(max(drawdown_pct) * 100, 2)} %")

    # 6. RANDOM SCREENSHOTS
    if len(trades) > 0:
        pair_out_dir = os.path.join(OUTPUT_BASE, pair_name)
        os.makedirs(pair_out_dir, exist_ok=True)
        samples = trades.sample(min(len(trades), SCREENSHOT_COUNT))
        for tid, (idx, row) in enumerate(samples.iterrows(), 1):
            entry_i, exit_i = df.index.get_loc(idx) + 1, int(row['exit_index_val'])
            start, end = max(0, entry_i - 10), min(len(df), exit_i + 15)
            chart_df = df.iloc[start:end][['Open','High','Low','Close']].copy()
            rel_entry, rel_exit = entry_i - start, exit_i - start
            
            buy_m = np.full(len(chart_df), np.nan); sell_m = np.full(len(chart_df), np.nan); exit_m = np.full(len(chart_df), np.nan)
            if row['signal'] == 'BUY': buy_m[rel_entry] = row['entry_price']
            else: sell_m[rel_entry] = row['entry_price']
            exit_m[rel_exit] = row['exit_price']

            apds = [
                mpf.make_addplot(buy_m, type='scatter', marker='^', markersize=100, color='green'),
                mpf.make_addplot(sell_m, type='scatter', marker='v', markersize=100, color='red'),
                mpf.make_addplot(exit_m, type='scatter', marker='x', markersize=80, color='black')
            ]
            apds = [a for a in apds if not np.all(np.isnan(a['data']))]
            mpf.plot(chart_df, type='candle', style='yahoo', addplot=apds,
                     title=f"{pair_name} | {row['signal']} | {row['result_pips']} Pips",
                     savefig=os.path.join(pair_out_dir, f"trade_{tid}.png"),warn_too_much_data=10000)