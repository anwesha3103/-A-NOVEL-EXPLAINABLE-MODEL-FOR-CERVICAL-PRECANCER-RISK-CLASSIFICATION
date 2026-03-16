"""
src/evaluate.py
===============
Evaluate a saved model on the full dataset:
  - Validation loss & accuracy
  - Per-case majority-vote predictions
  - Classification report
"""

import os
import sys
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import classification_report

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATA_DIR, MODEL_PATH, LABEL_MAP_REVERSE
from src.data_loader import (
    prepare_dataset,
    list_all_case_folders,
    load_case_images,
)


def evaluate_model():
    """Run overall accuracy evaluation on the full dataset."""
    print("[evaluate] Loading dataset...")
    X, y = prepare_dataset()

    X = preprocess_input(X.astype("float32"))

    print("[evaluate] Loading model...")
    model = load_model(MODEL_PATH)

    val_loss, val_accuracy = model.evaluate(X, y, verbose=1)
    print(f"\n[evaluate] Loss: {val_loss:.4f} | Accuracy: {val_accuracy:.4f}")

    # Classification report
    y_pred = np.argmax(model.predict(X, verbose=0), axis=1)
    labels = list(LABEL_MAP_REVERSE.values())
    print("\n[evaluate] Classification Report:")
    print(classification_report(y, y_pred, target_names=labels))


def predict_per_case(dataset_path: str = DATA_DIR):
    """Majority-vote prediction for every case folder."""
    model = load_model(MODEL_PATH)

    print("\n--- Per-case Predictions ---")
    for folder in sorted(list_all_case_folders(dataset_path)):
        folder_path = os.path.join(dataset_path, folder)
        images = load_case_images(folder_path)

        if len(images) == 0:
            print(f"  {folder}: [skipped – no valid images]")
            continue

        images = preprocess_input(images.astype("float32"))
        preds  = model.predict(images, verbose=0)
        pred_class = np.argmax(np.bincount(np.argmax(preds, axis=1)))
        label = LABEL_MAP_REVERSE[pred_class]
        print(f"  {folder}: {label}")


if __name__ == "__main__":
    evaluate_model()
    predict_per_case()
