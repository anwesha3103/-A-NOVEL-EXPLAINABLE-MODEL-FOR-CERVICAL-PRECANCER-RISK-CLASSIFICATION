"""
src/predict.py
==============
Inference on a single image with MC Dropout uncertainty estimation.
Can be imported by app.py or run from the command line.
"""

import sys
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MODEL_PATH, IMG_SIZE, LABEL_MAP_REVERSE, PRECAUTIONS, MC_DROPOUT_ITERATIONS

# ── Load model once at import time ────────────────────────────────────────────
_model = None

def get_model():
    global _model
    if _model is None:
        print(f"[predict] Loading model from: {MODEL_PATH}")
        _model = load_model(MODEL_PATH)
    return _model


# ── Core prediction ───────────────────────────────────────────────────────────

def predict_with_uncertainty(img_array: np.ndarray, n_iter: int = MC_DROPOUT_ITERATIONS):
    """
    Monte Carlo Dropout inference.

    Args:
        img_array : preprocessed image, shape (1, 224, 224, 3)
        n_iter    : number of stochastic forward passes

    Returns:
        mean_pred : np.ndarray  shape (1, num_classes)
        std_pred  : np.ndarray  shape (1, num_classes)
    """
    model = get_model()
    preds = [model(img_array, training=True).numpy() for _ in range(n_iter)]
    preds = np.array(preds)
    return np.mean(preds, axis=0), np.std(preds, axis=0)


def classify_image(image: np.ndarray, enable_uncertainty: bool = True) -> dict:
    """
    Full prediction pipeline from raw numpy image.

    Args:
        image              : HxWxC uint8 numpy array (from cv2 or Gradio)
        enable_uncertainty : whether to run MC Dropout or single forward pass

    Returns:
        dict with keys: Diagnosis, Confidence, Uncertainty Score, Suggested Precaution
    """
    # Preprocess
    img = cv2.resize(image, IMG_SIZE)
    if img.shape[-1] == 4:
        img = img[..., :3]          # drop alpha channel
    img = preprocess_input(img.astype("float32"))
    img = np.expand_dims(img, axis=0)

    # Inference
    n_iter = MC_DROPOUT_ITERATIONS if enable_uncertainty else 1
    mean_pred, std_pred = predict_with_uncertainty(img, n_iter=n_iter)

    pred_class    = int(np.argmax(mean_pred))
    confidence    = float(np.max(mean_pred))
    uncertainty   = float(np.mean(std_pred))

    label         = LABEL_MAP_REVERSE[pred_class]
    precaution    = PRECAUTIONS[label]

    return {
        "Diagnosis":          label,
        "Confidence":         round(confidence, 4),
        "Uncertainty Score":  round(uncertainty, 4),
        "Suggested Precaution": precaution,
    }


# ── CLI usage ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Classify a cervical image")
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument(
        "--no-uncertainty", action="store_true",
        help="Disable MC Dropout (faster, no uncertainty score)"
    )
    args = parser.parse_args()

    img = cv2.imread(args.image_path)
    if img is None:
        print(f"ERROR: Could not read image at '{args.image_path}'")
        sys.exit(1)

    result = classify_image(img, enable_uncertainty=not args.no_uncertainty)
    for k, v in result.items():
        print(f"  {k}: {v}")
