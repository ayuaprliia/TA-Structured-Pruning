from dotenv import load_dotenv
import os
from roboflow import Roboflow
from ultralytics import YOLO

# Load .env
load_dotenv()

API_KEY = os.getenv("ROBOFLOW_API_KEY")
if not API_KEY:
    raise ValueError("ROBOFLOW_API_KEY not found in .env")

# Download dataset (Roboflow)
rf = Roboflow(api_key=API_KEY)
project = rf.workspace("collision-detection").project("wayang-kulit-detc-xeos8")
dataset = project.version(13).download("yolov11")  

print("Dataset downloaded to:", dataset.location)

# Load your baseline model
model = YOLO("weights/baseline/best.pt")
print("Loaded model:", model)