import joblib
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType

def convert_larry_to_onnx():
    pkl_path = "models/larry_model_lgbm.pkl"
    onnx_path = "models/larry_model_lgbm.onnx"
    
    model = joblib.load(pkl_path)
    num_features = 9 # Larry menggunakan 9 fitur
    
    initial_type = [('float_input', FloatTensorType([None, num_features]))]
    
    # Konversi
    onnx_model = onnxmltools.convert_lightgbm(
        model, 
        initial_types=initial_type, 
        target_opset=13 
    )

    # --- TAMBAHKAN INI UNTUK MEMPERBAIKI ERROR 5802 ---
    # Memaksa nama output agar dikenali MQL5 secara standar
    onnx_model.graph.output[0].name = "label"
    onnx_model.graph.output[1].name = "probabilities"
    # --------------------------------------------------

    onnxmltools.utils.save_model(onnx_model, onnx_path)
    print(f"Berhasil konversi! Output 0: {onnx_model.graph.output[0].name}, Output 1: {onnx_model.graph.output[1].name}")

if __name__ == "__main__":
    convert_larry_to_onnx()