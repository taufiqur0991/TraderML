import joblib
import json
import onnx
import numpy as np
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
from xgboost import XGBClassifier

# 1. REGISTRASI CONVERTER (Wajib untuk XGBoost)
update_registered_converter(
    XGBClassifier, 'XGBoostXGBClassifier',
    calculate_linear_classifier_output_shapes, convert_xgboost,
    options={'zipmap': [True, False], 'nocl': [True, False]} 
)

def prepare_model_for_onnx(model):
    """ Memperbaiki metadata model agar kompatibel dengan runtime ONNX MetaTrader """
    try:
        booster = model.get_booster()
        
        # FIX BASE SCORE
        config = json.loads(booster.save_config())
        bs_raw = config['learner']['learner_model_param']['base_score']
        if isinstance(bs_raw, str):
            clean_str = bs_raw.replace('[', '').replace(']', '').split(',')[0]
            model.base_score = float(clean_str)
        
        # RESET FEATURE NAMES
        # MetaTrader 5 mengirim data sebagai array murni (f0, f1, f2...), 
        # bukan nama kolom (ATR, is_hs...). Kita harus samakan.
        num_features = model.n_features_in_
        new_names = [f'f{i}' for i in range(num_features)]
        booster.feature_names = new_names
        model._Booster = booster
        
        print(f"🔧 Model prepared: {num_features} features mapped to f0-f{num_features-1}")
        return num_features
    except Exception as e:
        print(f"⚠️ Warning during preparation: {e}")
        return None

# 2. LOAD MODEL PATTERN
# Pastikan nama file sesuai dengan hasil training Anda
model = joblib.load('models/pattern_model.pkl')
num_features = prepare_model_for_onnx(model)

# 3. KONVERSI KE ONNX
# PENTING: FloatTensorType([None, num_features]) 
# num_features harus 15 sesuai jumlah kolom di training terakhir
initial_type = [('float_input', FloatTensorType([None, num_features]))]
options = {'zipmap': False} # MQL5 lebih mudah membaca output tanpa zipmap

print(f"🔄 Converting {num_features} features to ONNX...")
try:
    onnx_model = convert_sklearn(
        model, 
        initial_types=initial_type, 
        options=options,
        target_opset={'': 14, 'ai.onnx.ml': 3} # Opset 14 lebih stabil untuk MT5 terbaru
    )

    # 4. SIMPAN UNTUK MT5
    with open("models/pattern_model.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())
    
    print("-" * 30)
    print("✅ BERHASIL!")
    print(f"📁 File: models/pattern_model.onnx")
    print(f"📊 Jumlah Input Fitur: {num_features}")
    print("-" * 30)
    print("⚠️ PENTING: Urutan array di MQL5 harus seperti ini:")
    features_list = [
        "0: BODY_SIZE", "1: REL_RANGE", "2: ATR", 
        "3: is_hs", "4: is_ihs", "5: is_double_top", 
        "6: is_double_bottom", "7: is_asc_tri", "8: is_desc_tri",
        "9: is_wedge_up", "10: is_wedge_down", "11: is_channel_up", 
        "12: is_channel_down", "13: VOL_SHOCK", "14: MOMENTUM"
    ]
    for feat in features_list: print(feat)

except Exception as e:
    print(f"❌ Error saat konversi: {e}")