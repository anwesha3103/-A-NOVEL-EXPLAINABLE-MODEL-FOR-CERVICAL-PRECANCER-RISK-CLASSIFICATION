Cervical Cancer Risk Classifier
MobileNetV2-based image classification for VIA (Visual Inspection with Acetic acid) cervical images.
Predicts Negative / Positive / Suspicious of cancer with uncertainty estimation via MC Dropout.

Project Structure
cervical_cancer_classifier/
│
├── config.py              ← ALL paths & hyperparameters (edit this first)
│
├── src/
│   ├── data_loader.py     ← dataset loading & metadata parsing
│   ├── model.py           ← MobileNetV2 architecture
│   ├── train.py           ← full training pipeline
│   ├── evaluate.py        ← evaluation & per-case predictions
│   └── predict.py         ← single-image inference (CLI or import)
│
├── app.py                 ← Gradio web UI
│
├── data/
│   └── raw/
│       └── IARC_ImageBank_VIA/     ← place your dataset here
│           ├── Cases - Images.xlsx
│           ├── Cases Meta data.xlsx
│           └── Case 001/ ... Case 186/
│
├── models/
│   └── cervical_model.h5   ← saved model (auto-created by train.py)
│
├── notebooks/
│   └── exploration.ipynb   ← original Colab notebook (reference)
│
└── requirements.txt

Quick Start
1 — Install dependencies
bashpython -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
2 — Place your dataset
Unzip IARCImageBankVIA.zip into data/raw/ so the structure becomes:
data/raw/IARC_ImageBank_VIA/
    Cases - Images.xlsx
    Cases Meta data.xlsx
    Case 001/
    Case 002/
    ...
3 — Place your trained model
Copy 96%accuracy.h5 (or any .h5 file) into models/ and update MODEL_PATH in config.py:
pythonMODEL_PATH = os.path.join(MODEL_DIR, "96%accuracy.h5")
4 — Train (if you want to retrain)
bashpython -m src.train
5 — Evaluate
bashpython -m src.evaluate
6 — Predict on a single image (CLI)
bashpython -m src.predict path/to/image.jpg
7 — Launch the web app
bashpython app.py
Open http://localhost:7860 in your browser.

Notes

MC Dropout: the Dropout layer stays active during inference to produce uncertainty scores.
SMOTE: applied automatically during training to handle class imbalance.
Set APP_SHARE = True in config.py to get a public gradio.live link.
