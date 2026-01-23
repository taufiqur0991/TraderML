//+------------------------------------------------------------------+
//|                                              ADX_Scalping_AI.mq5 |
//+------------------------------------------------------------------+
#include <Trade\Trade.mqh>

#property tester_file "scalping_adx.onnx"

// Output ONNX untuk Binary Classification biasanya memberikan probabilitas untuk 2 kelas
struct OnnxProbability {
   long  labels[2]; // Index 0 dan 1
   float values[2]; // Probabilitas untuk 0 dan 1
};

input group "AI Configuration"
input string  InpModelName      = "scalping_adx.onnx";
input double  InpThreshold      = 0.80; // Ambang batas keyakinan model (80%)
input int     InpADXPeriod      = 14;

input group "Strategy Settings"
input double  InpLotSize        = 0.1;
input int     InpMagic          = 999999;
input int     InpStopLossPips   = 200;  // 20 pips
input int     InpTakeProfitPips = 400;  // 40 pips

input group "Trailing Settings"
input double    InpTrailingStepRR = 0.5; 

//--- Global Variables
long      m_onnx_handle = INVALID_HANDLE;
int       h_adx, h_ma_fast;
CTrade    m_trade;

// JUMLAH FITUR HARUS 8 (Match dengan Python)
#define NUM_FEATURES 8

//+------------------------------------------------------------------+
int OnInit() {
   m_onnx_handle = OnnxCreate(InpModelName, ONNX_DEFAULT);
   if(m_onnx_handle == INVALID_HANDLE) return(INIT_FAILED);

   // Indikator pendukung untuk fitur
   h_adx = iADX(_Symbol, _Period, InpADXPeriod);
   h_ma_fast = iMA(_Symbol, _Period, 50, 0, MODE_SMA, PRICE_CLOSE);

   if(h_adx == INVALID_HANDLE || h_ma_fast == INVALID_HANDLE) return(INIT_FAILED);

   // Set Input Shape
   long input_shape[] = {1, NUM_FEATURES};
   OnnxSetInputShape(m_onnx_handle, 0, input_shape);
   
   // Set Output Shape (LightGBM Binary: Label & Probabilities)
   long shape_label[] = {1}; 
   long shape_probs[] = {1, 2}; 
   OnnxSetOutputShape(m_onnx_handle, 0, shape_label);
   OnnxSetOutputShape(m_onnx_handle, 1, shape_probs);

   m_trade.SetExpertMagicNumber(InpMagic);
   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason) {
   if(m_onnx_handle != INVALID_HANDLE) OnnxRelease(m_onnx_handle);
   IndicatorRelease(h_adx);
   IndicatorRelease(h_ma_fast);
}

void OnTick() {
   ApplyTrailingRR();
   // Hanya eksekusi pada Bar Baru
   static datetime last_bar = 0;
   datetime current_bar = iTime(_Symbol, _Period, 0);
   if(last_bar == current_bar) return;
   last_bar = current_bar;

   // Jangan buka posisi jika sudah ada yang aktif
   if(PositionsTotal() > 0) return; 

   float inputs[NUM_FEATURES];
   if(!CalculateFeatures(inputs)) return;

   // Run ONNX
   long predicted_label[1]; 
   OnnxProbability prob_data[1];
   if(!OnnxRun(m_onnx_handle, ONNX_NO_CONVERSION, inputs, predicted_label, prob_data)) return;

   int signal = (int)predicted_label[0];
   float confidence = prob_data[0].values[signal];

   // Filter keyakinan AI
   if(confidence < InpThreshold) return;

   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);

   // Eksekusi Instan (Market Order) sesuai strategi Scalping
   if(signal == 1) { // BUY
      double sl = ask - InpStopLossPips * _Point;
      double tp = ask + InpTakeProfitPips * _Point;
      m_trade.Buy(InpLotSize, _Symbol, ask, sl, tp, "ADX AI Buy");
   }
   else if(signal == 0) { // SELL
      double sl = bid + InpStopLossPips * _Point;
      double tp = bid - InpTakeProfitPips * _Point;
      m_trade.Sell(InpLotSize, _Symbol, bid, sl, tp, "ADX AI Sell");
   }
}

//+------------------------------------------------------------------+
//| MENGHITUNG 8 FITUR (SESUAI URUTAN PYTHON)                         |
//+------------------------------------------------------------------+
bool CalculateFeatures(float &f[]) {
   ArrayResize(f, NUM_FEATURES);
   
   MqlRates rates[];
   ArraySetAsSeries(rates, true);
   if(CopyRates(_Symbol, _Period, 1, 60, rates) < 60) return false; 

   double adx_main[], dmp[], dmn[], ma_val[], atr_val;
   ArraySetAsSeries(adx_main, true);
   ArraySetAsSeries(dmp, true);
   ArraySetAsSeries(dmn, true);
   ArraySetAsSeries(ma_val, true);

   if(CopyBuffer(h_adx, 0, 1, 1, adx_main) <= 0) return false;
   if(CopyBuffer(h_adx, 1, 1, 1, dmp) <= 0) return false;
   if(CopyBuffer(h_adx, 2, 1, 1, dmn) <= 0) return false;
   if(CopyBuffer(h_ma_fast, 0, 1, 1, ma_val) <= 0) return false;

   // Hitung ATR Manual (untuk normalisasi seperti di Python)
   double sum_tr = 0;
   for(int i=0; i<14; i++) {
      double tr = MathMax(rates[i].high - rates[i].low, 
                  MathMax(MathAbs(rates[i].high - rates[i+1].close), 
                          MathAbs(rates[i].low - rates[i+1].close)));
      sum_tr += tr;
   }
   atr_val = sum_tr / 14.0;
   if(atr_val <= 0) atr_val = _Point;

   // PEMETAAN FITUR (Wajib urut sesuai fitur_list di Python)
   f[0] = (float)adx_main[0];                            // ADX
   f[1] = (float)dmp[0];                                 // DMP
   f[2] = (float)dmn[0];                                 // DMN
   f[3] = (float)CalculateWilliamsR(14);                 // WILLR
   f[4] = (float)((rates[0].close - ma_val[0]) / atr_val); // DIST_SMA50
   f[5] = (float)((rates[0].close - rates[5].close) / atr_val); // F1_MOMENTUM
   
   MqlDateTime dt;
   TimeToStruct(rates[0].time, dt);
   f[6] = (float)dt.hour;                                // F6_HOUR
   
   f[7] = (float)CalculateRelVol(20, rates[0].tick_volume); // F4_REL_VOL

   return true;
}

double CalculateWilliamsR(int period) {
   double high = iHigh(_Symbol, _Period, iHighest(_Symbol, _Period, MODE_HIGH, period, 1));
   double low  = iLow(_Symbol, _Period, iLowest(_Symbol, _Period, MODE_LOW, period, 1));
   double close = iClose(_Symbol, _Period, 1);
   if(high - low == 0) return -50;
   return -100 * (high - close) / (high - low);
}

double CalculateRelVol(int period, long current_vol) {
   long vol_buffer[];
   if(CopyTickVolume(_Symbol, _Period, 1, period, vol_buffer) > 0) {
      double sum = 0;
      for(int i=0; i<period; i++) sum += (double)vol_buffer[i];
      return (double)current_vol / (sum / period);
   }
   return 1.0;
}

void ApplyTrailingRR() {
   for(int i = PositionsTotal() - 1; i >= 0; i--) {
      // 1. Ambil tiket posisi berdasarkan index
      ulong ticket = PositionGetTicket(i);
      
      // 2. Pilih posisi tersebut agar data POSITION_... bisa diakses
      if(PositionSelectByTicket(ticket)) {
         
         // 3. Filter berdasarkan Magic Number dan Symbol
         if(PositionGetInteger(POSITION_MAGIC) == InpMagic && PositionGetString(POSITION_SYMBOL) == _Symbol) {
            
            double open  = PositionGetDouble(POSITION_PRICE_OPEN);
            double sl    = PositionGetDouble(POSITION_SL);
            double tp    = PositionGetDouble(POSITION_TP);
            long type    = PositionGetInteger(POSITION_TYPE);
            double price = (type == POSITION_TYPE_BUY) ? SymbolInfoDouble(_Symbol, SYMBOL_BID) : SymbolInfoDouble(_Symbol, SYMBOL_ASK);

            double full_dist = MathAbs(tp - open);
            if(full_dist <= 0) continue;

            double trigger_price_dist = full_dist * InpTrailingStepRR;
            double current_profit_dist = MathAbs(price - open);

            // Jika harga sudah menempuh jarak RR yang ditentukan
            if(current_profit_dist >= trigger_price_dist) {
               double new_sl = 0;
               double buffer = 30 * _Point; // Jarak aman dari harga open

               if(type == POSITION_TYPE_BUY) {
                  // Geser SL ke Open + 30% dari profit saat ini
                  new_sl = NormalizeDouble(open + (current_profit_dist * 0.3), _Digits);
                  if(new_sl > sl + _Point || sl == 0) {
                     m_trade.PositionModify(ticket, new_sl, tp);
                  }
               } 
               else if(type == POSITION_TYPE_SELL) {
                  new_sl = NormalizeDouble(open - (current_profit_dist * 0.3), _Digits);
                  if(new_sl < sl - _Point || sl == 0) {
                     m_trade.PositionModify(ticket, new_sl, tp);
                  }
               }
            }
         }
      }
   }
}