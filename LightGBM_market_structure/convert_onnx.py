import joblib
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType

def convert_pkl_to_onnx(pkl_path, onnx_path):
    try:
        # 1. Load model .pkl
        print(f"Loading model dari {pkl_path}...")
        model = joblib.load(pkl_path)

        # 2. Definisikan jumlah fitur (WAJIB 9 agar match dengan MQL5)
        num_features = 6 
        
        # MQL5 membutuhkan float32
        initial_type = [('float_input', FloatTensorType([None, num_features]))]

        # 3. Konversi
        print("Mengonversi model ke format ONNX...")
        onnx_model = onnxmltools.convert_lightgbm(
            model, 
            initial_types=initial_type, 
            target_opset=13 # Opset 13 paling stabil untuk MT5
        )

        # 4. Simpan
        onnxmltools.utils.save_model(onnx_model, onnx_path)
        print(f"Berhasil! File ONNX siap digunakan: {onnx_path}")

    except Exception as e:
        print(f"Terjadi kesalahan: {e}")

if __name__ == "__main__":
    # Pastikan path ini sesuai dengan hasil train Anda
    PKL_FILE = "models/market_structure_lgbm.pkl" 
    ONNX_FILE = "models/market_structure.onnx"
    
    convert_pkl_to_onnx(PKL_FILE, ONNX_FILE)