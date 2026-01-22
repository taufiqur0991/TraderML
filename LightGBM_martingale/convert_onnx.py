import joblib
import onnxmltools
import numpy as np
# GUNAKAN INI: Tipe data dari onnxmltools, bukan dari skl2onnx
from onnxmltools.convert.common.data_types import FloatTensorType

def convert_pkl_to_onnx(pkl_path, onnx_path):
    try:
        # 1. Load model .pkl
        print(f"Loading model dari {pkl_path}...")
        model = joblib.load(pkl_path)

        # 2. Definisikan jumlah fitur (9 fitur sesuai train.py)
        num_features = 9 
        
        # Menggunakan FloatTensorType milik onnxmltools
        initial_type = [('float_input', FloatTensorType([None, num_features]))]

        # 3. Konversi model LightGBM ke ONNX
        print("Mengonversi model ke format ONNX...")
        
        # onnxmltools.convert_lightgbm secara internal memanggil fungsi 
        # yang ada di link github yang Anda berikan.
        onnx_model = onnxmltools.convert_lightgbm(
            model, 
            initial_types=initial_type, 
            target_opset=13
        )

        # 4. Simpan model ONNX
        onnxmltools.utils.save_model(onnx_model, onnx_path)
        print(f"Berhasil! Model ONNX disimpan di: {onnx_path}")

    except Exception as e:
        print(f"Terjadi kesalahan: {e}")

if __name__ == "__main__":
    PKL_FILE = "models/martingale_ai_model.pkl"
    ONNX_FILE = "models/martingale_ai_model.onnx"
    
    convert_pkl_to_onnx(PKL_FILE, ONNX_FILE)