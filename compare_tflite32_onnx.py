import cv2
import torch
import onnxruntime as ort
import tensorflow as tf
import numpy as np
from ultralytics import YOLO


# =========================================================
# CONFIG
# =========================================================
PYTORCH_MODEL_PATH = "/content/TA-Structured-Pruning/YOLO11-s Weights/11s_50_best.pt"
ONNX_MODEL_PATH = "/content/TA-Structured-Pruning/YOLO11-s Weights/11s_50_best.onnx"
TFLITE_MODEL_PATH = "/content/saved_model/11s_50_best_float32.tflite"
TEST_IMAGE_PATH = "/content/WAYANG LEMAH (21).jpg"
IMG_SIZE = 640


# =========================================================
# PREPROCESS
# =========================================================
def preprocess_image(
    image_path: str,
    img_size: int = 640
):
    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    raw_img = img.copy()

    img = cv2.resize(img, (img_size, img_size))
    img = img.astype(np.float32) / 255.0

    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    img_tensor = torch.from_numpy(img).float()

    return img_tensor, img.astype(np.float32), raw_img


# =========================================================
# PYTORCH INFERENCE
# =========================================================
def run_pytorch_inference(
    model_path: str,
    input_tensor: torch.Tensor
):
    model = YOLO(model_path)
    model.model.eval()

    with torch.no_grad():
        output = model.model(input_tensor)

    if isinstance(output, (list, tuple)):
        output = output[0]

    return output.cpu().numpy()


# =========================================================
# ONNX INFERENCE
# =========================================================
def run_onnx_inference(
    onnx_path: str,
    input_numpy: np.ndarray
):
    session = ort.InferenceSession(
        onnx_path,
        providers=['CPUExecutionProvider']
    )

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    output = session.run(
        [output_name],
        {input_name: input_numpy}
    )[0]

    return output


# =========================================================
# TFLITE INFERENCE
# =========================================================
def run_tflite_inference(
    tflite_path: str,
    input_numpy: np.ndarray
):
    interpreter = tf.lite.Interpreter(
        model_path=tflite_path
    )

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_tflite = np.transpose(
    input_numpy,
    (0, 2, 3, 1)
    )  # NCHW -> NHWC

    interpreter.set_tensor(
        input_details[0]['index'],
        input_tflite.astype(np.float32)
    )

    interpreter.invoke()

    output = interpreter.get_tensor(
        output_details[0]['index']
    )

    return output


# =========================================================
# COMPARISON
# =========================================================
def compare_outputs(
    reference_output: np.ndarray,
    test_output: np.ndarray,
    stage_name: str
):
    abs_diff = np.abs(reference_output - test_output)

    metrics = {
        "mean_abs_diff": float(np.mean(abs_diff)),
        "max_abs_diff": float(np.max(abs_diff)),
        "std_abs_diff": float(np.std(abs_diff)),
        "shape_match": reference_output.shape == test_output.shape
    }

    print("\n" + "=" * 60)
    print(f"VALIDATING {stage_name}")
    print("=" * 60)

    for k, v in metrics.items():
        print(f"{k}: {v}")

    mad = metrics["mean_abs_diff"]

    print("\nINTERPRETATION:")

    if mad < 1e-4:
        print("EXCELLENT: Hampir identik")
    elif mad < 1e-3:
        print("GOOD: Sangat dekat")
    elif mad < 1e-2:
        print("ACCEPTABLE: Sedikit deviasi")
    else:
        print("WARNING: Perbedaan besar")

    return metrics


# =========================================================
# MAIN
# =========================================================
def main():
    print("=" * 60)
    print("FULL VALIDATION PIPELINE")
    print("=" * 60)

    img_tensor, img_numpy, _ = preprocess_image(
        TEST_IMAGE_PATH,
        IMG_SIZE
    )

    print("\n[1] Running PyTorch inference...")
    pytorch_output = run_pytorch_inference(
        PYTORCH_MODEL_PATH,
        img_tensor
    )

    print("PyTorch output shape:", pytorch_output.shape)

    print("\n[2] Running ONNX inference...")
    onnx_output = run_onnx_inference(
        ONNX_MODEL_PATH,
        img_numpy
    )

    print("ONNX output shape:", onnx_output.shape)

    compare_outputs(
        pytorch_output,
        onnx_output,
        "ONNX vs PyTorch"
    )

    print("\n[3] Running TFLite inference...")
    tflite_output = run_tflite_inference(
        TFLITE_MODEL_PATH,
        img_numpy
    )

    print("TFLite output shape:", tflite_output.shape)

    compare_outputs(
        onnx_output,
        tflite_output,
        "TFLite vs ONNX"
    )


if __name__ == "__main__":
    main()