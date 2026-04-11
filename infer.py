import os
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from ultralytics import YOLO

# =========================
# CONFIG
# =========================
IMAGE_PATH = "/content/test bima.jpg"

MODELS = {
    "Prune10": "/content/TA-Structured-Pruning/YOLO11-s Weights/11s_10_best.pt",
    "Prune20": "/content/TA-Structured-Pruning/YOLO11-s Weights/11s_20_best.pt",
    "Prune30": "/content/TA-Structured-Pruning/YOLO11-s Weights/11s_30_best.pt",
    "Prune40": "/content/TA-Structured-Pruning/YOLO11-s Weights/11s_40_best.pt",
    "Prune50": "/content/TA-Structured-Pruning/YOLO11-s Weights/11s_50_best.pt",
}

OUTDIR = "/content/inference_comparison"
os.makedirs(OUTDIR, exist_ok=True)

saved_images = []

# =========================
# RUN INFERENCE FOR ALL MODELS
# =========================
for name, model_path in MODELS.items():
    print(f"Running inference: {name}")

    model = YOLO(model_path)

    result = model.predict(
        source=IMAGE_PATH,
        save=True,
        project=OUTDIR,
        name=name,
        exist_ok=True,
        device=0,
        line_width=2
    )

    # predicted image path
    pred_img_path = os.path.join(OUTDIR, name, os.path.basename(IMAGE_PATH))
    saved_images.append((name, pred_img_path))


# =========================
# DISPLAY COLLAGE
# =========================
n = len(saved_images)
cols = 2
rows = math.ceil(n / cols)

fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 6))
axes = axes.flatten()

for ax, (name, img_path) in zip(axes, saved_images):
    img = mpimg.imread(img_path)
    ax.imshow(img)
    ax.set_title(name, fontsize=14)
    ax.axis("off")

# Hide unused subplot
for i in range(len(saved_images), len(axes)):
    axes[i].axis("off")

plt.tight_layout()
plt.show()

print(f"\n✅ Semua hasil inferensi disimpan di: {OUTDIR}")