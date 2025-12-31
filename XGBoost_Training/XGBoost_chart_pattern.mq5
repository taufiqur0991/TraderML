#include <Trade\Trade.mqh>
#property tester_file "pattern_model.onnx"

// Input parameters
input long      InpMagic           = 123456;
input double    InpLotSize         = 0.1;
input double    InpRR              = 3.0;      
input string    InpModelName       = "pattern_model.onnx";
input double    InpThreshold       = 0.7;      
input int       InpAtrPeriod       = 14;
input int       InpExpireBars      = 4;
input string    config4            = "==== Input untuk Trailing ====";
input bool      InpUseTrailing     = true;
input double    InpTrailingStart   = 1.0;
input double    InpTrailingStep    = 0.5;   
input string    config5            = "==== Jangan trading jika ATR dibawah 30 pips ====";
input double    InputMinATR        = 0.0040;

// Global variables
long      m_onnx_handle = INVALID_HANDLE;
CTrade    m_trade;
string    m_current_pattern = "";

// JUMLAH FITUR WAJIB 15 (Sesuai list Python terakhir)
#define NUM_FEATURES 15 

int OnInit()
{
   m_onnx_handle = OnnxCreate(InpModelName, ONNX_DEFAULT);
   if(m_onnx_handle == INVALID_HANDLE) return(INIT_FAILED);

   // Setup Input Shape (1 row, 15 features)
   long input_shape[] = {1, NUM_FEATURES};
   if(!OnnxSetInputShape(m_onnx_handle, 0, input_shape)) return(INIT_FAILED);

   // Node 0: Label, Node 1: Probabilities
   long shape_label[] = {1}; 
   long shape_probs[] = {1, 3};
   OnnxSetOutputShape(m_onnx_handle, 0, shape_label);
   OnnxSetOutputShape(m_onnx_handle, 1, shape_probs);

   m_trade.SetExpertMagicNumber(InpMagic);
   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason)
{
   if(m_onnx_handle != INVALID_HANDLE) OnnxRelease(m_onnx_handle);
}

void OnTick()
{
   // 1. Kelola Trailing Stop untuk posisi yang sudah AKTIF
   if(InpUseTrailing) ApplyTrailingStop();

   // 2. Logika Bar Baru
   static datetime last_bar = 0;
   if(last_bar == iTime(_Symbol, _Period, 0)) return;
   last_bar = iTime(_Symbol, _Period, 0);

   // 3. Ambil Sinyal AI
   float inputs[NUM_FEATURES];
   if(!CalculatePatternFeatures(inputs)) return;
   if(!IdentifyPatternName(inputs)) return;

   long  predicted_label[1];    
   float predicted_probs[3];    
   if(!OnnxRun(m_onnx_handle, ONNX_NO_CONVERSION, inputs, predicted_label, predicted_probs)) return;

   // 4. CEK SINYAL & CLEANUP PENDING ORDER
   // Jika ada sinyal kuat baru (Buy/Sell), hapus pending order lama agar tidak tumpang tindih
   if(predicted_probs[1] > InpThreshold || predicted_probs[2] > InpThreshold)
   {
      CancelAllPendingOrders();
   }

   // 5. EKSEKUSI JIKA TIDAK ADA POSISI AKTIF
   if(PositionsTotal() == 0 && OrdersTotal() == 0)
   {
      if(predicted_probs[1] > InpThreshold)      ExecuteOrder(ORDER_TYPE_BUY, m_current_pattern);
      else if(predicted_probs[2] > InpThreshold) ExecuteOrder(ORDER_TYPE_SELL, m_current_pattern);
   }
}

// --- FUNGSI MENGHAPUS PENDING ORDER LAMA ---
void CancelAllPendingOrders()
{
   for(int i=OrdersTotal()-1; i>=0; i--)
   {
      ulong ticket = OrderGetTicket(i);
      if(OrderSelect(ticket))
      {
         if(OrderGetInteger(ORDER_MAGIC) == InpMagic)
         {
            m_trade.OrderDelete(ticket);
            Print("Pending Order Lama Dihapus karena Muncul Sinyal Baru.");
         }
      }
   }
}

bool IdentifyPatternName(float &f[])
{
   m_current_pattern = "";
   
   if(f[3] > 0)  m_current_pattern = "Head & Shoulders";
   else if(f[4] > 0)  m_current_pattern = "Inv Head & Shoulders";
   else if(f[5] > 0)  m_current_pattern = "Double Top";
   else if(f[6] > 0)  m_current_pattern = "Double Bottom";
   else if(f[7] > 0)  m_current_pattern = "Ascending Triangle";
   else if(f[8] > 0)  m_current_pattern = "Descending Triangle";
   else if(f[9] > 0)  m_current_pattern = "Wedge Up";
   else if(f[10] > 0) m_current_pattern = "Wedge Down";
   else if(f[11] > 0) m_current_pattern = "Channel Up";
   else if(f[12] > 0) m_current_pattern = "Channel Down";

   return (m_current_pattern != "");
}

// FUNGSI UTAMA: MENGHITUNG 15 FITUR
bool CalculatePatternFeatures(float &f[])
{
   ArrayResize(f, NUM_FEATURES);
   ArrayInitialize(f, 0.0f);

   // Ambil Data Dasar (Index 1 adalah bar yang baru tertutup)
   double h0 = iHigh(_Symbol, _Period, 1);
   double l0 = iLow(_Symbol, _Period, 1);
   double c0 = iClose(_Symbol, _Period, 1);
   double o0 = iOpen(_Symbol, _Period, 1);
   
   // 0: BODY_SIZE
   f[0] = (float)(MathAbs(c0 - o0) / (h0 - l0 + 1e-9));

   // 1: REL_RANGE
   int h_idx = iHighest(_Symbol, _Period, MODE_HIGH, 10, 1);
   int l_idx = iLowest(_Symbol, _Period, MODE_LOW, 10, 1);
   double r10 = iHigh(_Symbol, _Period, h_idx) - iLow(_Symbol, _Period, l_idx);
   f[1] = (float)((h0 - l0) / (r10 + 1e-9));

   // 2: ATR
   double atr[];
   int atr_h = iATR(_Symbol, _Period, InpAtrPeriod);
   CopyBuffer(atr_h, 0, 1, 1, atr);
   f[2] = (float)atr[0];

   // --- LOGIKA POLA (WINDOW 3) ---
   // Bar index: [1] adalah current, [2] adalah shift(1), [3] adalah shift(2)
   double h1 = iHigh(_Symbol, _Period, 2), l1 = iLow(_Symbol, _Period, 2);
   double h2 = iHigh(_Symbol, _Period, 3), l2 = iLow(_Symbol, _Period, 3);
   
   double roll_max = MathMax(h0, MathMax(h1, h2));
   double roll_min = MathMin(l0, MathMin(l1, l2));

   // 3 & 4: Head & Shoulders
   f[3] = (roll_max > h1 && roll_max > h2 && h0 < h1 && h0 < h2) ? 1.0f : 0.0f; // is_hs
   f[4] = (roll_min < l1 && roll_min < l2 && l0 > l1 && l0 > l2) ? 1.0f : 0.0f; // is_ihs

   // 5 & 6: Double Top/Bottom (Threshold 0.05)
   double range_avg = (h1 + l1) / 2.0;
   bool dt_range = (h1 - l1) <= 0.05 * range_avg;
   f[5] = (roll_max >= h1 && roll_max >= h2 && h0 < h1 && h0 < h2 && dt_range) ? 1.0f : 0.0f;
   f[6] = (roll_min <= l1 && roll_min <= l2 && l0 > l1 && l0 > l2 && dt_range) ? 1.0f : 0.0f;

   // 7 & 8: Triangle
   f[7] = (roll_max >= h1 && roll_min <= l1 && c0 > iClose(_Symbol, _Period, 2)) ? 1.0f : 0.0f; // asc
   f[8] = (roll_max <= h1 && roll_min >= l1 && c0 < iClose(_Symbol, _Period, 2)) ? 1.0f : 0.0f; // desc

   // 9 & 10: Wedge (Trend based)
   int trend_h = (h0 > h2) ? 1 : (h0 < h2 ? -1 : 0);
   int trend_l = (l0 > l2) ? 1 : (l0 < l2 ? -1 : 0);
   f[9]  = (roll_max >= h1 && roll_min <= l1 && trend_h == 1 && trend_l == 1) ? 1.0f : 0.0f;
   f[10] = (roll_max <= h1 && roll_min >= l1 && trend_h == -1 && trend_l == -1) ? 1.0f : 0.0f;

   // 11 & 12: Channel (Range 0.1)
   double chan_val = 0.1 * (roll_max + roll_min) / 2.0;
   bool chan_range = (roll_max - roll_min) <= chan_val;
   f[11] = (f[9] == 1.0f && chan_range) ? 1.0f : 0.0f;
   f[12] = (f[10] == 1.0f && chan_range) ? 1.0f : 0.0f;

   // 13: VOL_SHOCK (ATR / ATR 10 bar lalu)
   double atr10[];
   CopyBuffer(atr_h, 0, 11, 1, atr10);
   f[13] = (float)(atr[0] / (atr10[0] + 1e-9));

   // 14: MOMENTUM (pct_change 5 bar)
   double c5 = iClose(_Symbol, _Period, 6);
   f[14] = (float)((c0 - c5) / (c5 + 1e-9));

   return true;
}

void ApplyTrailingStop()
{
   for(int i=PositionsTotal()-1; i>=0; i--)
   {
      if(PositionSelectByTicket(PositionGetTicket(i)))
      {
         if(PositionGetInteger(POSITION_MAGIC) != InpMagic) continue;

         double entry   = PositionGetDouble(POSITION_PRICE_OPEN);
         double current = (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) ? SymbolInfoDouble(_Symbol, SYMBOL_BID) : SymbolInfoDouble(_Symbol, SYMBOL_ASK);
         double current_sl = PositionGetDouble(POSITION_SL);
         
         // Kita hitung jarak Risk awal untuk menentukan level RR
         // Kita asumsikan Risk awal disimpan atau bisa dihitung dari SL awal
         // Untuk mempermudah, kita gunakan ATR sebagai satuan jarak jika SL awal tidak terlacak
         double atr[];
         CopyBuffer(iATR(_Symbol, _Period, 14), 0, 0, 1, atr);
         double unit = atr[0] * 1.5; // Ini estimasi jarak risk awal

         if(PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY)
         {
            double profit_rr = (current - entry) / unit;
            if(profit_rr >= InpTrailingStart)
            {
               // Hitung level SL baru: (Profit RR - 1.0) agar memberi jarak 1 RR dari harga sekarang
               double new_sl = entry + ((profit_rr - 1.0) * unit);
               if(new_sl > current_sl + (10 * _Point)) // Hanya geser jika lebih baik
               {
                  m_trade.PositionModify(PositionGetTicket(i), NormalizeDouble(new_sl, _Digits), PositionGetDouble(POSITION_TP));
               }
            }
         }
         else // Untuk SELL
         {
            double profit_rr = (entry - current) / unit;
            if(profit_rr >= InpTrailingStart)
            {
               double new_sl = entry - ((profit_rr - 1.0) * unit);
               if(current_sl == 0 || new_sl < current_sl - (10 * _Point))
               {
                  m_trade.PositionModify(PositionGetTicket(i), NormalizeDouble(new_sl, _Digits), PositionGetDouble(POSITION_TP));
               }
            }
         }
      }
   }
}

void ExecuteOrder(ENUM_ORDER_TYPE type, string pattern_name)
{
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   
   double entry_p, sl, tp;
   int lookback = 20;

   // Hitung Waktu Expired (Current Time + Jumlah Bar * Detik dalam Period)
   datetime expiration = TimeCurrent() + InpExpireBars * PeriodSeconds(_Period);

   if(type == ORDER_TYPE_BUY)
   {
      entry_p = iHigh(_Symbol, _Period, 1) + (15 * _Point); 
      int low_idx = iLowest(_Symbol, _Period, MODE_LOW, lookback, 1);
      sl = iLow(_Symbol, _Period, low_idx);
      
      double risk = entry_p - sl;
      tp = entry_p + (risk * InpRR);
      
      if(ask < entry_p) 
      {
         m_trade.BuyStop(InpLotSize, NormalizeDouble(entry_p, _Digits), _Symbol, 
                         NormalizeDouble(sl, _Digits), NormalizeDouble(tp, _Digits), 
                         ORDER_TIME_SPECIFIED, expiration, pattern_name);
      }
   }
   else
   {
      entry_p = iLow(_Symbol, _Period, 1) - (15 * _Point);
      int high_idx = iHighest(_Symbol, _Period, MODE_HIGH, lookback, 1);
      sl = iHigh(_Symbol, _Period, high_idx);
      
      double risk = sl - entry_p;
      tp = entry_p - (risk * InpRR);
      
      if(bid > entry_p)
      {
         m_trade.SellStop(InpLotSize, NormalizeDouble(entry_p, _Digits), _Symbol, 
                          NormalizeDouble(sl, _Digits), NormalizeDouble(tp, _Digits), 
                          ORDER_TIME_SPECIFIED, expiration, pattern_name);
      }
   }
}