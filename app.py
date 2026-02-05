from flask import Flask, request, jsonify
import os
import joblib
import wave
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
        "usage": "POST /detect with multipart/form-data",
        "accepted_format": "WAV (PCM, RIFF only)"
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

def is_valid_wav(path):
    """Ensure file is a real WAV (RIFF)"""
    try:
        with wave.open(path, "rb"):
            return True
    except wave.Error:
        return False

@app.route("/detect", methods=["POST"])
def detect():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if not file.filename.lower().endswith(".wav"):
        return jsonify({
            "error": "Invalid format",
            "details": "Only WAV audio files are supported. Convert MP3/MP4 to WAV before upload."
        }), 400

    temp_path = "temp.wav"

    try:
        file.save(temp_path)

        # 🔒 Hard WAV validation (prevents MP3 crash)
        if not is_valid_wav(temp_path):
            return jsonify({
                "error": "Invalid WAV file",
                "details": "File is not a valid RIFF/WAV audio. Do not rename MP3 to WAV."
            }), 400

        # --- Feature extraction ---
        features = extract_features(temp_path)

        # --- AI vs Human ---
        auth_pred = auth_model.predict([features])[0]
        auth_prob = float(auth_model.predict_proba([features])[0].max())
        classification = "AI_GENERATED" if auth_pred == 1 else "HUMAN"

        # --- Language ---
        lang_pred = lang_model.predict([features])[0]
        language = LANG_MAP_REV.get(lang_pred, "unknown")

        return jsonify({
            "classification": classification,
            "confidence_score": auth_prob,
            "detected_language": language
        })

    except Exception as e:
        return jsonify({
            "error": "Processing failed",
            "details": str(e)
        }), 500

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
