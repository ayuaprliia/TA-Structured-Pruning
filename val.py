# ======= COLAB CELL: INFERENCE TIME VIA VAL (TEST SET) =======
from ultralytics import YOLO
import time

DATA_YAML = "/content/TA-Structured-Pruning/Wayang-Kulit-Detc-13/data.yaml"
IMGSZ = 640
BATCH = 8
DEVICE = 0

MODEL_10   = "/content/TA-Structured-Pruning/11s_10_best.pt"
MODEL_20   = "/content/TA-Structured-Pruning/11s_20_best.pt"

def measure_inference_time_val(model_path, data_yaml, imgsz=640, batch=8, device=0):
    model = YOLO(model_path)
    results = model.val(data=data_yaml, split="test", imgsz=imgsz, batch=batch, device=device, verbose=False)

    # results.speed: dict with keys like 'preprocess', 'inference', 'postprocess' (ms/img)
    speed = results.speed  # ms/img
    return speed["inference"]


time_10   = measure_inference_time_val(MODEL_10, DATA_YAML, IMGSZ, BATCH, DEVICE)
time_20   = measure_inference_time_val(MODEL_20, DATA_YAML, IMGSZ, BATCH, DEVICE)

print("Inference time (ms/img):")
print("Prune10 :", time_10)
print("Prune20 :", time_20)