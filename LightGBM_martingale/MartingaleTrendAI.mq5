//+------------------------------------------------------------------+
//|                                 BabiNgepet_Sideway_Scalper_V7.mq5|
//|                                     Copyright 2026, zvickyhac    |
//+------------------------------------------------------------------+
#property copyright "zvickyhac"
#property version   "7.00"
#property tester_file "martingale_ai_model.onnx"

#include <Trade\Trade.mqh>

struct OnnxProbability {
   long   labels[3]; 
   float  values[3]; 
};

// --- Input Parameters ---
input string  section0 = "=== General Parameters ===";
input long    InpMagic               = 9191; 
input double  InpLotSize             = 0.08;
input int     TakeProfitTargetUSD    = 12;

input string  section1 = "=== AI & ONNX Config ===";
input string  InpModelName           = "martingale_ai_model.onnx";
input double  InpAISidewayThreshold  = 0.70; // Entry HANYA jika Sideway >= ini
input double  InpAISafetyThreshold   = 0.85; // Rem Marti jika Trend Lawan >= ini
#define NUM_FEATURES 9 

input string  section2 = "=== Envelopes (Scalper Mode) ===";
input int     InpEnvPeriod           = 8;     // Periode lebih pendek
input double  InpEnvDeviation        = 0.06;  // Deviasi lebih sempit untuk scalping
input ENUM_MA_METHOD InpEnvMethod    = MODE_SMA;

input string  section3 = "=== Martingale Strategy ===";
input double  LotMultiplier          = 2.0;         
input int     DistanceMartingalePips = 15;                                                                           
input int     MaxTradesInSeries      = 8;

input string  section4 = "=== Risk Management ===";
input double  StopLossByMarginPercent = 50.0;

// --- Global Variables ---
long      m_onnx_handle = INVALID_HANDLE;
int       m_atr_handle, m_ema200_handle, m_env_handle;
CTrade    m_trade;
double    s_currentLot  = 0.0;
ENUM_POSITION_TYPE s_seriesType = (ENUM_POSITION_TYPE)-1;

//+------------------------------------------------------------------+
int OnInit() {
   m_onnx_handle = OnnxCreate(InpModelName, ONNX_DEFAULT);
   if(m_onnx_handle == INVALID_HANDLE) return(INIT_FAILED);

   m_atr_handle    = iATR(_Symbol, _Period, 14);
   m_ema200_handle = iMA(_Symbol, _Period, 200, 0, MODE_EMA, PRICE_CLOSE);
   m_env_handle    = iEnvelopes(_Symbol, _Period, InpEnvPeriod, 0, InpEnvMethod, PRICE_CLOSE, InpEnvDeviation);

   long input_shape[] = {1, NUM_FEATURES};
   OnnxSetInputShape(m_onnx_handle, 0, input_shape);
   long shape_label[] = {1}, shape_probs[] = {1}; 
   OnnxSetOutputShape(m_onnx_handle, 0, shape_label);
   OnnxSetOutputShape(m_onnx_handle, 1, shape_probs);

   m_trade.SetExpertMagicNumber(InpMagic);
   s_currentLot = InpLotSize;
   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason) {
   if(m_onnx_handle != INVALID_HANDLE) OnnxRelease(m_onnx_handle);
   IndicatorRelease(m_atr_handle); IndicatorRelease(m_ema200_handle); IndicatorRelease(m_env_handle);
}

void OnTick() {
   static datetime last_bar = 0;
   if(last_bar == iTime(_Symbol, _Period, 0)) return;
   last_bar = iTime(_Symbol, _Period, 0);

   double totalProfitUSD = 0;
   int openTrades = CountOpenPositions(totalProfitUSD);

   if(openTrades > 0 && totalProfitUSD >= TakeProfitTargetUSD) {
      CloseAllPositions(); ResetSeries(); return;
   }
   if(CheckMarginStopLoss()) return;

   float inputs[NUM_FEATURES];
   if(!CalculateFeatures(inputs)) return;

   long predicted_label[1]; OnnxProbability prob_data[1];
   if(!OnnxRun(m_onnx_handle, ONNX_NO_CONVERSION, inputs, predicted_label, prob_data)) return;
   
   float prob_side = prob_data[0].values[0];
   float prob_bull = prob_data[0].values[1];
   float prob_bear = prob_data[0].values[2];

   double upper[], lower[];
   CopyBuffer(m_env_handle, 0, 1, 1, upper); CopyBuffer(m_env_handle, 1, 1, 1, lower);
   double price_close = iClose(_Symbol, _Period, 1);

   // --- LOGIKA SIDEWAYS SCALPER ---
   bool is_sideway = (prob_side >= InpAISidewayThreshold);

   if(openTrades == 0) {
      ResetSeries();
      // Buy di pita bawah HANYA saat Sideway tinggi
      if(price_close <= lower[0] && is_sideway) {
         ExecuteOrder(ORDER_TYPE_BUY, InpLotSize, "AI Sideway Buy");
      }
      // Sell di pita atas HANYA saat Sideway tinggi
      else if(price_close >= upper[0] && is_sideway) {
         ExecuteOrder(ORDER_TYPE_SELL, InpLotSize, "AI Sideway Sell");
      }
   }
   else if(openTrades < MaxTradesInSeries) {
      double lastPrice = GetLastTradePrice();
      double dist = MathAbs(SymbolInfoDouble(_Symbol, SYMBOL_BID) - lastPrice) / _Point;
      double reqDist = DistanceMartingalePips * ((_Digits==5 || _Digits==3) ? 10 : 1);

      if(dist >= reqDist) {
         // Tetap ada rem safety agar tidak marti konyol saat mendadak trend kuat
         if(s_seriesType == POSITION_TYPE_BUY && prob_bear >= InpAISafetyThreshold) return;
         if(s_seriesType == POSITION_TYPE_SELL && prob_bull >= InpAISafetyThreshold) return;

         double nextLot = NormalizeDouble(s_currentLot * LotMultiplier, 2);
         ExecuteOrder((s_seriesType == POSITION_TYPE_BUY) ? ORDER_TYPE_BUY : ORDER_TYPE_SELL, nextLot, "Marti Recovery");
      }
   }
}

bool CalculateFeatures(float &f[]) {
   ArrayResize(f, NUM_FEATURES);
   double atr[], ema200[], ema200_prev[];
   if(CopyBuffer(m_atr_handle,0,1,1,atr)<=0 || CopyBuffer(m_ema200_handle,0,1,1,ema200)<=0 || CopyBuffer(m_ema200_handle,0,6,1,ema200_prev)<=0) return false;
   double c0 = iClose(_Symbol, _Period, 1), o0 = iOpen(_Symbol, _Period, 1), h0 = iHigh(_Symbol, _Period, 1), l0 = iLow(_Symbol, _Period, 1);
   f[0] = (float)((ema200[0] - ema200_prev[0]) / (ema200[0] + 1e-9));
   f[1] = (float)((c0 - ema200[0]) / (ema200[0] + 1e-9));
   f[2] = (float)atr[0];
   f[3] = (float)((h0 - l0) / (atr[0] + 1e-9));
   f[4] = (float)((MathMax(0, h0 - iHigh(_Symbol, _Period, 2)) / (atr[0] + 1e-9)) * 100);
   f[5] = (float)(MathAbs(c0 - o0) / (h0 - l0 + 1e-9));
   MqlDateTime dt; TimeCurrent(dt); f[6] = (float)dt.day_of_week; f[7] = (float)dt.hour;
   f[8] = (iClose(_Symbol, _Period, 2) > iOpen(_Symbol, _Period, 2)) ? 1.0f : -1.0f;
   return true;
}

void ExecuteOrder(ENUM_ORDER_TYPE type, double lot, string comment) {
   double p = (type==ORDER_TYPE_BUY)?SymbolInfoDouble(_Symbol,SYMBOL_ASK):SymbolInfoDouble(_Symbol,SYMBOL_BID);
   if(m_trade.PositionOpen(_Symbol, type, lot, p, 0, 0, comment)) { s_currentLot = lot; s_seriesType = (type==ORDER_TYPE_BUY)?POSITION_TYPE_BUY:POSITION_TYPE_SELL; }
}

int CountOpenPositions(double &profit) {
   int count = 0; profit = 0;
   for(int i=PositionsTotal()-1; i>=0; i--) {
      if(PositionSelectByTicket(PositionGetTicket(i)) && PositionGetInteger(POSITION_MAGIC)==InpMagic) { count++; profit += PositionGetDouble(POSITION_PROFIT); }
   }
   return count;
}

double GetLastTradePrice() {
   datetime lt=0; double lp=0;
   for(int i=PositionsTotal()-1; i>=0; i--) {
      if(PositionSelectByTicket(PositionGetTicket(i)) && PositionGetInteger(POSITION_MAGIC)==InpMagic) {
         if(PositionGetInteger(POSITION_TIME)>lt) { lt=(datetime)PositionGetInteger(POSITION_TIME); lp=PositionGetDouble(POSITION_PRICE_OPEN); }
      }
   }
   return lp;
}

void CloseAllPositions() {
   for(int i=PositionsTotal()-1; i>=0; i--) { if(PositionSelectByTicket(PositionGetTicket(i)) && PositionGetInteger(POSITION_MAGIC)==InpMagic) m_trade.PositionClose(PositionGetTicket(i)); }
}

void ResetSeries() { s_currentLot = InpLotSize; s_seriesType = (ENUM_POSITION_TYPE)-1; }

bool CheckMarginStopLoss() {
   double eq = AccountInfoDouble(ACCOUNT_EQUITY), bal = AccountInfoDouble(ACCOUNT_BALANCE);
   if(bal > 0 && ((bal - eq)/bal)*100.0 >= StopLossByMarginPercent) { CloseAllPositions(); return true; }
   return false;
}