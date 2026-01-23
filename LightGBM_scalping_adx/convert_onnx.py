import joblib
import onnxmltools
import os
from onnxmltools.convert.common.data_types import FloatTensorType

def convert_pkl_to_onnx(pkl_path, onnx_path):
    try:
        # 1. Load model .pkl
        if not os.path.exists(pkl_path):
            print(f"Error: File {pkl_path} tidak ditemukan!")
            return
        
        # 1. Load model .pkl
        print(f"Loading model dari {pkl_path}...")
        model = joblib.load(pkl_path)

        # 2. Definisikan jumlah fitur
        num_features = 8 
        
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
        print("-" * 30)
        print(f"Berhasil! File ONNX siap digunakan: {onnx_path}")
        print(f"Pastikan di MQL5 input array berukuran: {num_features}")
        print("-" * 30)

    except Exception as e:
        print(f"Terjadi kesalahan: {e}")

if __name__ == "__main__":
    # Pastikan path ini sesuai dengan hasil train Anda
    PKL_FILE = "models/scalping_adx.pkl" 
    ONNX_FILE = "models/scalping_adx.onnx"
    
    convert_pkl_to_onnx(PKL_FILE, ONNX_FILE)