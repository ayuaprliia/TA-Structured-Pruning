from ultralytics import YOLO

MODEL_PATH = "/content/TA-Structured-Pruning/YOLO11-s Weights/11s_50_best.pt"

model = YOLO(MODEL_PATH)
model.export(
    format="onnx",
    imgsz=640,
    opset=13,
    simplify=True,
    dynamic=False
)