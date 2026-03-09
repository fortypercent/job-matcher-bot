"""Reassembles split ONNX model files on first run."""
import glob
import os

MODEL_DIR = os.path.join(os.path.dirname(__file__), "onnx_model_quantized")
ONNX_FILE = os.path.join(MODEL_DIR, "model_quantized.onnx")

def reassemble():
    if os.path.exists(ONNX_FILE):
        return  # already assembled
    parts = sorted(glob.glob(os.path.join(MODEL_DIR, "model_part_*")))
    if not parts:
        return
    print(f"Assembling {len(parts)} parts into {ONNX_FILE}...")
    with open(ONNX_FILE, "wb") as out:
        for part in parts:
            with open(part, "rb") as f:
                out.write(f.read())
    print("Done!")

if __name__ == "__main__":
    reassemble()
