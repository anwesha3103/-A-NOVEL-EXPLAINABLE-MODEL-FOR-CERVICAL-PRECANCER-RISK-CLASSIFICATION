"""
src/train.py
============
Full training pipeline:
  1. Load dataset
  2. Preprocess & split
  3. Apply SMOTE for class imbalance
  4. Build model
  5. Train with EarlyStopping
  6. Save model weights
"""

import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Make sure src/ imports work when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    MODEL_PATH, BATCH_SIZE, EPOCHS, EARLY_STOPPING_PATIENCE
)
from src.data_loader import prepare_dataset
from src.model import build_model


def run_training():
    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("[train] Loading dataset...")
    X, y = prepare_dataset()

    if len(X) == 0:
        print("[train] ERROR: No data found. Check DATA_DIR in config.py.")
        return

    print(f"[train] Dataset: {X.shape}, labels: {y.shape}")

    # ── 2. Preprocess ─────────────────────────────────────────────────────────
    X = preprocess_input(X)          # MobileNetV2-specific normalisation

    # ── 3. Train / Validation split ───────────────────────────────────────────
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[train] Train={len(X_train)}, Val={len(X_val)}")

    # ── 4. SMOTE (operate on flattened images, then reshape) ──────────────────
    try:
        from imblearn.over_sampling import SMOTE
        n_samples, h, w, c = X_train.shape
        X_flat = X_train.reshape(n_samples, -1)
        smote  = SMOTE(random_state=42)
        X_flat_res, y_train = smote.fit_resample(X_flat, y_train)
        X_train = X_flat_res.reshape(-1, h, w, c)
        print(f"[train] After SMOTE: {X_train.shape}")
    except Exception as e:
        print(f"[train] SMOTE skipped ({e}). Training with original distribution.")

    # ── 5. Class weights (fallback if SMOTE isn't used) ───────────────────────
    classes      = np.unique(y_train)
    class_weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, class_weights))
    print(f"[train] Class weights: {class_weight_dict}")

    # ── 6. Build model ────────────────────────────────────────────────────────
    model = build_model(num_classes=len(classes))

    # ── 7. Callbacks ──────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    # ── 8. Train ──────────────────────────────────────────────────────────────
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=callbacks,
    )

    print(f"\n[train] ✅ Model saved to: {MODEL_PATH}")
    print(f"[train] Best val accuracy: {max(history.history['val_accuracy']):.4f}")

    return history


if __name__ == "__main__":
    run_training()
