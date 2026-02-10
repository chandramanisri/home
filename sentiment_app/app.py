from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, request

from sentiment_app.model import IncrementalSentimentModel, MODEL_PATH, sentiment_label

app = Flask(__name__)

model: IncrementalSentimentModel | None = None


@app.get("/health")
def health():
    ready = model is not None
    return jsonify({"status": "ok", "model_loaded": ready, "model_path": str(MODEL_PATH)})


@app.post("/predict")
def predict():
    global model
    if model is None:
        return jsonify({"error": "Model not loaded. Train first."}), 400

    payload = request.get_json(force=True)
    text = payload.get("text", "").strip()
    if not text:
        return jsonify({"error": "text is required"}), 400

    score = model.predict_proba([text])[0]
    return jsonify(
        {
            "text": text,
            "sentiment_score": round(score, 4),
            "sentiment": sentiment_label(score),
        }
    )


def load_model_if_exists() -> None:
    global model
    if Path(MODEL_PATH).exists():
        model = IncrementalSentimentModel.load(MODEL_PATH)


if __name__ == "__main__":
    load_model_if_exists()
    app.run(host="0.0.0.0", port=8000)
