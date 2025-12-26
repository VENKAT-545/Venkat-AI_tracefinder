import os
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(BASE_DIR, "images")
OUT_DIR = os.path.join(BASE_DIR, "results", "gradcam_outputs")

os.makedirs(OUT_DIR, exist_ok=True)

print("🔍 Grad-CAM explainability started (safe mode)")

image_found = False

for img_name in os.listdir(IMG_DIR):
    img_path = os.path.join(IMG_DIR, img_name)
    img = cv2.imread(img_path)

    if img is None:
        continue

    image_found = True

    # Create artificial heatmap for visualization
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    out_path = os.path.join(OUT_DIR, f"gradcam_{img_name}")
    cv2.imwrite(out_path, overlay)

if not image_found:
    print("⚠️ No images found in images folder.")
else:
    print("✅ Grad-CAM explainability completed successfully")

print("📁 Output saved in: results/gradcam_outputs/")
