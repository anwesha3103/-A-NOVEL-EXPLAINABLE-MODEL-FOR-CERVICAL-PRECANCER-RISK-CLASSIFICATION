"""
app.py
======
Gradio web interface for the Cervical Cancer Risk Classifier.
Run with:  python app.py
"""

import gradio as gr

from config import APP_HOST, APP_PORT, APP_SHARE
from src.predict import classify_image


# ── Wrapper for Gradio ────────────────────────────────────────────────────────

def predict_for_gradio(image, enable_uncertainty: bool = True) -> str:
    """Gradio-compatible wrapper that returns a formatted Markdown string."""
    if image is None:
        return "⚠️ Please upload an image."

    result = classify_image(image, enable_uncertainty=enable_uncertainty)

    return (
        f"### 🩺 Diagnosis: **{result['Diagnosis']}**\n\n"
        f"**📊 Confidence:** `{result['Confidence']:.4f}`\n\n"
        f"**⚠️ Uncertainty Score:** `{result['Uncertainty Score']:.4f}`\n\n"
        f"**📌 Suggested Precaution:**\n{result['Suggested Precaution']}"
    )


# ── Gradio Interface ──────────────────────────────────────────────────────────

with gr.Blocks(theme=gr.themes.Soft()) as interface:
    gr.Markdown("""
    <div style='text-align:center;'>
        <h1>🧬 Cervical Cancer Risk Classifier</h1>
        <p style='color:#666;'>
            Upload a cervix image (VIA / Acetic Acid test) to detect potential risk levels.<br>
            The model uses <strong>MobileNetV2 + MC Dropout</strong> to estimate prediction uncertainty.
        </p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                label="📸 Upload Cervical Image (.jpg / .png)",
                type="numpy",
                image_mode="RGB",
                height=260,
            )
            uncertainty_toggle = gr.Checkbox(
                label="Enable Uncertainty Estimation (MC Dropout)",
                value=True,
            )
            analyze_btn = gr.Button("🔍 Analyze", variant="primary")

        with gr.Column(scale=1):
            results_output = gr.Markdown(label="Prediction Results")

    analyze_btn.click(
        fn=predict_for_gradio,
        inputs=[image_input, uncertainty_toggle],
        outputs=results_output,
    )

    gr.Markdown("""
    ---
    > ⚠️ **Disclaimer:** This tool is for research/educational purposes only.
    > It is **not** a substitute for professional medical diagnosis.
    """)


# ── Launch ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    interface.launch(
        server_name=APP_HOST,
        server_port=APP_PORT,
        share=APP_SHARE,
    )
