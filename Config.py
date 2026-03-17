"""
config.py
=========
Central config for all paths, hyperparameters, and constants.
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR        = os.path.join(BASE_DIR, "data", "raw", "IARC_ImageBank_VIA")
EXCEL_IMAGES    = os.path.join(DATA_DIR, "Cases - Images.xlsx")
EXCEL_META      = os.path.join(DATA_DIR, "Cases Meta data.xlsx")

MODEL_DIR       = os.path.join(BASE_DIR, "models")
MODEL_PATH      = os.path.join(MODEL_DIR, "cervical_model.h5")   # ← put your .h5 here

# ─── Image Settings ───────────────────────────────────────────────────────────

IMG_SIZE        = (224, 224)   # MobileNetV2 input size
IMG_CHANNELS    = 3

# ─── Training Hyperparameters ─────────────────────────────────────────────────

BATCH_SIZE      = 32
EPOCHS          = 30
LEARNING_RATE   = 0.0001
DROPOUT_RATE    = 0.5
EARLY_STOPPING_PATIENCE = 5

# ─── Class Labels ─────────────────────────────────────────────────────────────

LABEL_MAP = {
    "Negative":             0,
    "Positive":             1,
    "Suspicious of cancer": 2,
}

LABEL_MAP_REVERSE = {v: k for k, v in LABEL_MAP.items()}

PRECAUTIONS = {
    "Negative":             "✅ Continue regular check-ups and maintain a healthy lifestyle.",
    "Positive":             "⚠️ Follow up with your doctor for additional screening.",
    "Suspicious of cancer": "🚨 Seek immediate medical attention for further diagnosis.",
}

# ─── MC Dropout (Uncertainty) ─────────────────────────────────────────────────

MC_DROPOUT_ITERATIONS = 10   # number of stochastic forward passes

# ─── Gradio App ───────────────────────────────────────────────────────────────

APP_HOST  = "0.0.0.0"
APP_PORT  = 7860
APP_SHARE = False   # set True to get a public gradio.live link
