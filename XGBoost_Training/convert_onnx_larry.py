import joblib
import json
import onnx
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
from xgboost import XGBClassifier

# 1. REGISTRASI CONVERTER
update_registered_converter(
    XGBClassifier, 'XGBoostXGBClassifier',
    calculate_linear_classifier_output_shapes, convert_xgboost,
    options={'zipmap': [True, False], 'nocl': [True, False]} 
)

def prepare_model_for_onnx(model):
    """ Fix base_score DAN Reset Feature Names agar ONNX tidak bingung """
    try:
        booster = model.get_booster()
        
        # --- FIX BASE SCORE ---
        config = json.loads(booster.save_config())
        bs_raw = config['learner']['learner_model_param']['base_score']
        if isinstance(bs_raw, str):
            clean_str = bs_raw.replace('[', '').replace(']', '').split(',')[0]
            model.base_score = float(clean_str)
        
        # --- FIX FEATURE NAMES (KUNCI ERROR ANDA) ---
        # Memaksa XGBoost melupakan nama 'REL_RANGE' dkk, ganti jadi f0, f1...
        num_features = model.n_features_in_
        new_names = [f'f{i}' for i in range(num_features)]
        booster.feature_names = new_names
        model._Booster = booster # Update booster di dalam wrapper sklearn
        
        print(f"🔧 Model prepared: Base_score fixed & Features renamed to f0-f{num_features-1}")
    except Exception as e:
        print(f"⚠️ Warning during preparation: {e}")

# 2. LOAD & PREPARE
model = joblib.load('models/larry_model.pkl')
prepare_model_for_onnx(model)

# 3. KONVERSI
initial_type = [('float_input', FloatTensorType([None, 4]))]
options = {'zipmap': False} 

print("🔄 Converting to ONNX...")
try:
    onnx_model = convert_sklearn(
        model, 
        initial_types=initial_type, 
        options=options,
        target_opset={'': 12, 'ai.onnx.ml': 3}
    )

    # 4. SIMPAN
    with open("models/larry_model.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())
    print("✅ BERHASIL! File larry_model.onnx siap digunakan di MQL5.")

except Exception as e:
    print(f"❌ Error saat konversi: {e}")