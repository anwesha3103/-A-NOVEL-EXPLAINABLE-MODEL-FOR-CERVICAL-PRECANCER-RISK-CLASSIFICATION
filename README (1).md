#  -A-NOVEL-EXPLAINABLE-MODEL-FOR-CERVICAL-PRECANCER-RISK-CLASSIFICATION


<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-MobileNetV2-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-UI-orange?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A-NOVEL-EXPLAINABLE-MODEL-FOR-CERVICAL-PRECANCER-RISK-CLASSIFICATION**  
*with Uncertainty Estimation via Monte Carlo Dropout*

[Features](#-features) • [Demo](#-demo) • [Installation](#-installation) • [Usage](#-usage) • [Model](#-model-architecture) • [Dataset](#-dataset)

</div>

---

## Overview

This project uses **transfer learning on MobileNetV2** to classify cervical images into three risk categories:

| Label | Description | Recommended Action |
|-------|-------------|-------------------|
| **Negative** | No abnormalities detected | Regular check-ups |
| **Positive** | Abnormalities present | Follow up with doctor |
| **Suspicious of Cancer** | High-risk indicators | Immediate medical attention |

What makes this classifier unique is its use of **Monte Carlo Dropout** — the model doesn't just predict a label, it also tells you *how confident* it is. Low confidence = flag for human review.

---

##  Features

-  **3-class cervical image classification** (Negative / Positive / Suspicious)
-  **Uncertainty estimation** via Monte Carlo Dropout (50 stochastic forward passes)
-  **Class imbalance handling** with SMOTE oversampling
-  **Transfer learning** on MobileNetV2 pretrained on ImageNet
-  **Gradio web interface** — upload an image, get instant results
-  **~98.8% validation accuracy** on IARC ImageBank VIA dataset
-  Fully modular Python codebase — easy to extend and retrain

---

##  Demo

Launch the web app and upload any cervical image:

```bash
python app.py
```

Then open **http://localhost:7860** in your browser.

> Set `APP_SHARE = True` in `config.py` to generate a public `gradio.live` link you can share with anyone.

**Example output:**
```
 Diagnosis:        Positive
 Confidence:       0.9241
 Uncertainty Score: 0.0183
 Precaution:       Follow up with your doctor for additional screening.
```

---

## 📁 Project Structure

```
cervical_cancer_classifier/
├── config.py                        ← All paths & hyperparameters
├── app.py                           ← Gradio web UI
├── requirements.txt                 ← All dependencies
├── README.md
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py               ← Image & metadata loading
│   ├── model.py                     ← MobileNetV2 architecture
│   ├── train.py                     ← Full training pipeline
│   ├── evaluate.py                  ← Metrics & per-case predictions
│   └── predict.py                   ← Inference + MC Dropout
│
├── data/
│   └── raw/
│       └── IARC_ImageBank_VIA/      ← Place your dataset here
│           ├── Cases - Images.xlsx
│           ├── Cases Meta data.xlsx
│           └── Case 001/ ... Case 186/
│
├── models/
│   └── cervical_model.h5            ← Trained model weights
│
└── notebooks/
    └── exploration.ipynb            ← Original Colab notebook
```

---

##  Installation

### 1 — Clone & set up environment

```bash
git clone https://github.com/yourusername/cervical-cancer-classifier.git
cd cervical-cancer-classifier

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2 — Place your dataset

Unzip `IARCImageBankVIA.zip` into `data/raw/`:

```
data/raw/IARC_ImageBank_VIA/
    ├── Cases - Images.xlsx
    ├── Cases Meta data.xlsx
    ├── Case 001/
    ├── Case 002/
    └── ...
```

### 3 — Place your trained model

Copy your `.h5` model file into `models/` and update `config.py`:

```python
MODEL_PATH = os.path.join(MODEL_DIR, "96%accuracy.h5")
```

---

##  Usage

### Launch the web app
```bash
python app.py
```

### Predict on a single image (CLI)
```bash
python -m src.predict path/to/image.jpg
```

### Evaluate model performance
```bash
python -m src.evaluate
```

### Retrain from scratch
```bash
python -m src.train
```

---

##  Model Architecture

```
Input (224×224×3)
       ↓
MobileNetV2 [frozen — ImageNet weights]
       ↓
Flatten
       ↓
Dense(128, ReLU)
       ↓
Dropout(0.5)  ← stays ON during inference for uncertainty estimation
       ↓
Dense(3, Softmax)
       ↓
Output: [Negative, Positive, Suspicious of cancer]
```

### Why Monte Carlo Dropout?

Standard neural networks output a single prediction with no sense of confidence. MC Dropout runs the model **50 times** with dropout active during inference — the **mean** gives the prediction, the **standard deviation** gives the uncertainty score.

> A high uncertainty score means the model is unsure — flagging these cases for expert review makes the system safer in clinical use.

---

##  Dataset

**IARC ImageBank VIA** — 186 patient cases, ~370 cervical images (before/after acetic acid application)

| Class | Description |
|-------|-------------|
| Negative | No VIA abnormality |
| Positive | VIA positive |
| Suspicious of Cancer | High-grade lesion suspected |

> Dataset source: International Agency for Research on Cancer (IARC)

---

##  Configuration

All key settings live in `config.py` — no need to touch other files:

```python
IMG_SIZE               = (224, 224)
BATCH_SIZE             = 32
EPOCHS                 = 30
LEARNING_RATE          = 0.0001
DROPOUT_RATE           = 0.5
MC_DROPOUT_ITERATIONS  = 10
APP_SHARE              = False    # True → public gradio.live link
```

---

##  Dependencies

```
tensorflow >= 2.12
keras >= 2.12
opencv-python
numpy
pandas
scikit-learn
imbalanced-learn      # SMOTE
gradio >= 4.0
openpyxl
matplotlib
```

---

##  Disclaimer

> This tool is intended for **research and educational purposes only**.  
> It is **not** a substitute for professional medical diagnosis.  
> Always consult a qualified healthcare provider for medical decisions.

---

## 🙌 Acknowledgements

- [IARC](https://www.iarc.who.int/) for the ImageBank VIA dataset
- [MobileNetV2](https://arxiv.org/abs/1801.04381) — Sandler et al., 2018
- [Gradio](https://gradio.app/) for the web interface
- [imbalanced-learn](https://imbalanced-learn.org/) for SMOTE implementation

---

<div align="center">

</div>
