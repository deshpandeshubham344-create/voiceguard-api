from flask import Flask, request, jsonify
import os, base64, joblib, wave
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

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/detect", methods=["POST"])
def detect():
    temp_path = "temp.wav"

    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "Invalid JSON"}), 400

        # ✅ REQUIRED BY TESTER
        audio_b64 = data.get("audio_base64")
        if not audio_b64:
            return jsonify({"error": "audio_base64 missing"}), 400

        # ✅ Remove possible base64 header
        if "," in audio_b64:
            audio_b64 = audio_b64.split(",")[1]

        # Decode Base64 → WAV
        audio_bytes = base64.b64decode(audio_b64)
        with open(temp_path, "wb") as f:
            f.write(audio_bytes)

        # ✅ Strict WAV validation
        try:
            with wave.open(temp_path, "rb") as wf:
                if wf.getnchannels() < 1:
                    raise ValueError("Invalid WAV")
        except Exception:
            return jsonify({
                "error": "Invalid WAV file",
                "details": "File must be PCM WAV (not MP3 renamed to WAV)"
            }), 400

        # Feature extraction
        features = extract_features(temp_path)

        # AI vs Human
        auth_pred = auth_model.predict([features])[0]
        auth_prob = float(auth_model.predict_proba([features])[0].max())
        classification = "AI_GENERATED" if auth_pred == 1 else "HUMAN"

        # Language
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


