import os
import subprocess


def convert_onnx_to_savedmodel(
    onnx_path: str,
    output_dir: str
):
    """
    Convert ONNX model to TensorFlow SavedModel using onnx2tf.
    """

    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "onnx2tf",
        "-i", onnx_path,
        "-o", output_dir
    ]

    print("=" * 60)
    print("CONVERTING ONNX → TENSORFLOW SAVEDMODEL")
    print("=" * 60)

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    print(result.stdout)

    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("Conversion failed.")

    print("\n SavedModel generated at:", output_dir)

convert_onnx_to_savedmodel(
    onnx_path="/content/TA-Structured-Pruning/YOLO11-s Weights/11s_50_best.onnx",
    output_dir="/content/saved_model"
)