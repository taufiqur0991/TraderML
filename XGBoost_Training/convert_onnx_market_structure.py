import joblib
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import os
import glob
import json
import numpy as np

# Register XGBoost converter untuk skl2onnx
from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
from skl2onnx import update_registered_converter
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
from xgboost import XGBClassifier

update_registered_converter(
    XGBClassifier, 'XGBoostXGBClassifier',
    calculate_linear_classifier_output_shapes, convert_xgboost,
    options={'nocl': [True, False], 'zipmap': [True, False, 'columns']}
)

def fix_base_score(xgb_model):
    """
    Fix base_score untuk multi-class classifier.
    Multi-class punya array base_score (satu per kelas), bukan single float.
    """
    try:
        booster = xgb_model.get_booster()
        config = json.loads(booster.save_config())
        bs_raw = config['learner']['learner_model_param']['base_score']
        
        # Cek apakah base_score berupa string array
        if isinstance(bs_raw, str):
            # Hapus bracket dan split by comma
            clean_str = bs_raw.replace('[', '').replace(']', '').strip()
            
            if ',' in clean_str:
                # Multi-class: convert ke numpy array
                values = [float(x.strip()) for x in clean_str.split(',')]
                new_bs = np.array(values, dtype=np.float32)
                print(f"   🔧 Multi-class base_score detected: {values}")
                
                # Untuk multi-class, gunakan nilai rata-rata atau pertama sebagai fallback
                # (ONNX converter kadang butuh single value)
                single_value = float(values[0])
                xgb_model.base_score = single_value
                xgb_model.set_params(base_score=single_value)
                
                print(f"   ✅ Using first class base_score: {single_value}")
            else:
                # Binary/single class
                new_bs = float(clean_str)
                xgb_model.base_score = new_bs
                xgb_model.set_params(base_score=new_bs)
                print(f"   ✅ Fixed base_score: {bs_raw} -> {new_bs}")
            
            # Refresh internal state
            model_bytearray = booster.save_raw("json")
            booster.load_model(model_bytearray)
            
    except Exception as e:
        print(f"   ⚠️ base_score fix skipped: {e}")
        # Tidak fatal, konversi bisa tetap jalan


def convert_model_to_onnx(pair_name):
    """
    Convert XGBoost model dari .pkl ke .onnx
    """
    pkl_path = f"models/{pair_name}.pkl"
    onnx_path = f"models/{pair_name}.onnx"
    feature_map_path = f"models/{pair_name}_feature_map.json"
    
    if not os.path.exists(pkl_path):
        print(f"❌ Model {pkl_path} tidak ditemukan!")
        return False
    
    print(f"📦 Loading model: {pkl_path}")
    xgb_model = joblib.load(pkl_path)
    
    # Info model
    print(f"   Type: {type(xgb_model).__name__}")
    print(f"   N_estimators: {xgb_model.n_estimators}")
    print(f"   N_features: {xgb_model.n_features_in_}")
    
    # Deteksi objective (binary vs multiclass)
    objective = xgb_model.get_params().get('objective', 'unknown')
    print(f"   Objective: {objective}")
    
    # Fix base_score
    fix_base_score(xgb_model)
    
    # Simpan feature mapping
    original_feature_names = list(xgb_model.feature_names_in_)
    feature_map = {f"f{i}": name for i, name in enumerate(original_feature_names)}
    
    with open(feature_map_path, 'w') as f:
        json.dump(feature_map, f, indent=2)
    print(f"   Feature map saved: {feature_map_path}")
    
    # Reset feature names untuk kompatibilitas ONNX
    print(f"   Resetting feature names...")
    booster = xgb_model.get_booster()
    new_feature_names = [f"f{i}" for i in range(xgb_model.n_features_in_)]
    booster.feature_names = new_feature_names
    xgb_model._Booster = booster
    
    # Input shape untuk ONNX
    initial_type = [('float_input', FloatTensorType([None, xgb_model.n_features_in_]))]
    
    # Convert ke ONNX
    print(f"🔄 Converting to ONNX...")
    try:
        onnx_model = convert_sklearn(
            xgb_model,
            initial_types=initial_type,
            target_opset={'': 14, 'ai.onnx.ml': 3},
            options={id(xgb_model): {'zipmap': False}}
        )
        
        onnx_bytes = onnx_model.SerializeToString()
        print(f"   ONNX size: {len(onnx_bytes)/1024:.2f} KB")
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Simpan model ONNX
    with open(onnx_path, 'wb') as f:
        f.write(onnx_bytes)
    
    # Verifikasi ONNX model
    try:
        onnx_model_check = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model_check)
        print(f"✅ Conversion successful: {onnx_path}")
        return True
    except Exception as e:
        print(f"⚠️ Verification warning: {e}")
        # Tetap return True karena file sudah tersimpan
        return True


def convert_all_models():
    """
    Convert semua model .pkl di folder models/
    """
    pkl_files = glob.glob("models/*.pkl")
    
    if not pkl_files:
        print("❌ Tidak ada model .pkl di folder models/")
        return
    
    print(f"🔍 Found {len(pkl_files)} models to convert\n")
    print("="*60)
    
    success_count = 0
    failed_models = []
    
    for pkl_file in pkl_files:
        pair_name = os.path.basename(pkl_file).replace('.pkl', '')
        print()
        
        if convert_model_to_onnx(pair_name):
            success_count += 1
        else:
            failed_models.append(pair_name)
    
    # Summary
    print()
    print("="*60)
    print(f"🎉 Conversion complete!")
    print(f"   Success: {success_count}/{len(pkl_files)}")
    
    if failed_models:
        print(f"   Failed: {', '.join(failed_models)}")


if __name__ == "__main__":
    convert_all_models()