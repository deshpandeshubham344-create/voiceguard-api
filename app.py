from flask import Flask, request, jsonify
import os
import joblib
from features import extract_features

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load models
auth_model = joblib.load(os.path.join(BASE_DIR, "model/voiceguard_model.pkl"))
lang_model = joblib.load(os.path.join(BASE_DIR, "model/language_model.pkl"))

LANG_MAP_REV = {
    0: "tamil",
    1: "english",
    2: "hindi",
    3: "malayalam",
    4: "telugu"
}

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "VoiceGuard API running",
        "usage": "POST /detect with multipart/form-data (file=.wav)"
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/detect", methods=["POST"])
def detect():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if not file.filename.lower().endswith(".wav"):
        return jsonify({"error": "Only WAV files supported"}), 400

    temp_path = "temp.wav"

    try:
        file.save(temp_path)

        # --- Feature extraction (safe) ---
        features = extract_features(temp_path)

        # --- Prediction ---
        auth_pred = auth_model.predict([features])[0]
        auth_prob = float(auth_model.predict_proba([features])[0].max())
        classification = "AI_GENERATED" if auth_pred == 1 else "HUMAN"

        lang_pred = lang_model.predict([features])[0]
        language = LANG_MAP_REV.get(lang_pred, "unknown")

        return jsonify({
            "classification": classification,
            "confidence_score": auth_prob,
            "detected_language": language
        })

    except Exception as e:
        # Convert crash → readable error
        return jsonify({
            "error": "Processing failed",
            "details": str(e)
        }), 500

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
