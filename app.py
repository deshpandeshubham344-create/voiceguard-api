
from flask import Flask, request, jsonify
import os
import joblib
from pydub import AudioSegment
from features import extract_features

app = Flask(__name__)

# Load models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
auth_model = joblib.load(os.path.join(BASE_DIR, "model/voiceguard_model.pkl"))
lang_model = joblib.load(os.path.join(BASE_DIR, "model/language_model.pkl"))

LANG_MAP_REV = {
    0: "tamil",
    1: "english",
    2: "hindi",
    3: "malayalam",
    4: "telugu"
}

def generate_explanation(classification, confidence):
    if classification == "AI_GENERATED":
        if confidence > 0.85:
            return "The voice shows highly uniform pitch and low background noise, which are common patterns in AI-generated speech."
        else:
            return "The voice contains synthetic characteristics, but with moderate confidence."
    else:
        if confidence > 0.85:
            return "The voice contains natural pauses, breathing patterns, and background noise typical of human speech."
        else:
            return "The voice appears human, but some synthetic traits were detected."

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "VoiceGuard API is running",
        "endpoint": "/detect",
        "note": "Supports MP4, MP3, WAV (auto converted to WAV)"
    })

@app.route("/detect", methods=["POST"])
def detect():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        input_path = "input_audio"
        output_path = "temp.wav"

        # Save uploaded file (any format)
        file.save(input_path)

        # Convert to WAV using ffmpeg via pydub
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(output_path, format="wav")

        # Extract features
        features = extract_features(output_path)

        # AI vs Human
        auth_pred = auth_model.predict([features])[0]
        auth_prob = auth_model.predict_proba([features])[0].max()
        classification = "AI_GENERATED" if auth_pred == 1 else "HUMAN"

        # Language
        lang_pred = lang_model.predict([features])[0]
        language = LANG_MAP_REV.get(lang_pred, "unknown")

        explanation = generate_explanation(classification, auth_prob)

        # Cleanup
        os.remove(input_path)
        os.remove(output_path)

        return jsonify({
            "classification": classification,
            "confidence_score": float(auth_prob),
            "detected_language": language,
            "explanation": explanation
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
