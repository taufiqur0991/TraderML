#include <Trade\Trade.mqh>
#property tester_file "larry_model.onnx" // Pastikan nama file sesuai hasil ekspor ONNX

// Input parameters
input long      InpMagic           = 123456;
input double    InpLotSize         = 0.5;
input double    InpRR              = 1.5;      // Larry Williams RR ideal
input string    InpModelName       = "larry_model.onnx";
input double    InpThreshold       = 0.6;      // Probability threshold
input int       InpAtrPeriod       = 14;

// Global variables
long      m_onnx_handle = INVALID_HANDLE;
int       m_atr_handle  = INVALID_HANDLE;
CTrade    m_trade;

// JUMLAH FITUR HARUS SAMA DENGAN PYTHON
// Features: ["BODY_SIZE", "REL_RANGE", "is_outside_bull", "is_outside_bear"]
#define NUM_FEATURES 4 

int OnInit()
{
   m_onnx_handle = OnnxCreate(InpModelName, ONNX_DEFAULT);
   if(m_onnx_handle == INVALID_HANDLE) return(INIT_FAILED);

   m_atr_handle = iATR(_Symbol, _Period, InpAtrPeriod);

   // Setup Input Shape (1 row, 4 features)
   long input_shape[] = {1, NUM_FEATURES};
   if(!OnnxSetInputShape(m_onnx_handle, 0, input_shape)) return(INIT_FAILED);

   // --- PERBAIKAN DI SINI ---
   // Node 0: Label (Biasanya Int64)
   long shape_label[] = {1}; 
   if(!OnnxSetOutputShape(m_onnx_handle, 0, shape_label)) Print("Warning Output 0: ", GetLastError());

   // Node 1: Probabilities (1 row, 3 classes)
   long shape_probs[] = {1, 3};
   if(!OnnxSetOutputShape(m_onnx_handle, 1, shape_probs)) 
   {
      Print("Gagal SetOutputShape Probabilities: ", GetLastError());
      return(INIT_FAILED);
   }

   m_trade.SetExpertMagicNumber(InpMagic);
   return(INIT_SUCCEEDED);
}

void OnTick()
{
   static datetime last_bar = 0;
   if(last_bar == iTime(_Symbol, _Period, 0)) return;
   last_bar = iTime(_Symbol, _Period, 0);

   if(PositionsTotal() > 0) return;

   // 1. Deteksi Strategi Larry Williams (Mekanik)
   if(!IsLarryOutsideBar()) return;

   // 2. Jika ada Outside Bar, baru panggil AI untuk filter
   float inputs[NUM_FEATURES];
   if(!CalculateFeatures(inputs)) return;

   // Gunakan tipe data yang kaku sesuai standar ONNX
   long  predicted_label[1];    // Node 0
   float predicted_probs[3];    // Node 1 (Hold, Buy, Sell)

   // Gunakan ONNX_NO_CONVERSION agar MT5 tidak mencoba merubah tipe data secara otomatis
   if(!OnnxRun(m_onnx_handle, ONNX_NO_CONVERSION, inputs, predicted_label, predicted_probs))
   {
      // Jika masih error, print error code-nya
      // Print("OnnxRun Error: ", GetLastError());
      return;
   }

   // Gunakan threshold pada probabilitas kelas 1 (Buy) atau kelas 2 (Sell)
   if(predicted_probs[1] > InpThreshold) 
   {
      Print("AI Signal BUY: ", predicted_probs[1]);
      ExecuteOrder(ORDER_TYPE_BUY);
   }
   else if(predicted_probs[2] > InpThreshold) 
   {
      Print("AI Signal SELL: ", predicted_probs[2]);
      ExecuteOrder(ORDER_TYPE_SELL);
   }
}

// LOGIKA STRATEGI LARRY WILLIAMS (Hampir sama dengan Python)
bool IsLarryOutsideBar()
{
   double h0 = iHigh(_Symbol, _Period, 1);
   double l0 = iLow(_Symbol, _Period, 1);
   double c0 = iClose(_Symbol, _Period, 1);
   
   double h1 = iHigh(_Symbol, _Period, 2);
   double l1 = iLow(_Symbol, _Period, 2);

   bool bull = (l0 < l1 && h0 > h1 && c0 > h1);
   bool bear = (h0 > h1 && l0 < l1 && c0 < l1);
   
   return (bull || bear);
}

bool CalculateFeatures(float &features[])
{
   ArrayResize(features, NUM_FEATURES);
   
   double h0 = iHigh(_Symbol, _Period, 1);
   double l0 = iLow(_Symbol, _Period, 1);
   double c0 = iClose(_Symbol, _Period, 1);
   double o0 = iOpen(_Symbol, _Period, 1);
   
   double h1 = iHigh(_Symbol, _Period, 2);
   double l1 = iLow(_Symbol, _Period, 2);

   // Hitung REL_RANGE (High-Low 10 candle terakhir)
   int high_idx = iHighest(_Symbol, _Period, MODE_HIGH, 10, 1);
   int low_idx  = iLowest(_Symbol, _Period, MODE_LOW, 10, 1);
   double range_10 = iHigh(_Symbol, _Period, high_idx) - iLow(_Symbol, _Period, low_idx);

   // Isi Fitur (Urutan harus sama dengan Python!)
   // 0: BODY_SIZE
   features[0] = (float)(MathAbs(c0 - o0) / (h0 - l0 + 0.000001));
   
   // 1: REL_RANGE
   features[1] = (float)((h0 - l0) / (range_10 + 0.000001));
   
   // 2: is_outside_bull
   features[2] = (l0 < l1 && h0 > h1 && c0 > h1) ? 1.0f : 0.0f;
   
   // 3: is_outside_bear
   features[3] = (h0 > h1 && l0 < l1 && c0 < l1) ? 1.0f : 0.0f;

   return true;
}

void ExecuteOrder(ENUM_ORDER_TYPE type)
{
   double price = (type == ORDER_TYPE_BUY) ? SymbolInfoDouble(_Symbol, SYMBOL_ASK) : SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double h0 = iHigh(_Symbol, _Period, 1);
   double l0 = iLow(_Symbol, _Period, 1);

   double sl, tp;
   if(type == ORDER_TYPE_BUY)
   {
      sl = l0; // SL di Low Bar Signal (Khas Larry Williams)
      tp = price + (price - sl) * InpRR;
      m_trade.Buy(InpLotSize, _Symbol, price, NormalizeDouble(sl, _Digits), NormalizeDouble(tp, _Digits));
   }
   else
   {
      sl = h0; // SL di High Bar Signal
      tp = price - (sl - price) * InpRR;
      m_trade.Sell(InpLotSize, _Symbol, price, NormalizeDouble(sl, _Digits), NormalizeDouble(tp, _Digits));
   }
}