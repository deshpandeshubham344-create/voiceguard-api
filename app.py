from flask import Flask, request, jsonify
import os
import joblib
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
        "note": "Upload WAV audio file using form-data"
    })

@app.route("/detect", methods=["POST"])
def detect():
    # Expect file upload
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Save uploaded file
    temp_path = "temp.wav"
    file.save(temp_path)

    # Extract features
    features = extract_features(temp_path)

    # AI vs Human
    auth_pred = auth_model.predict([features])[0]
    auth_prob = auth_model.predict_proba([features])[0].max()
    classification = "AI_GENERATED" if auth_pred == 1 else "HUMAN"

    # Language
    lang_pred = lang_model.predict([features])[0]
    language = LANG_MAP_REV.get(lang_pred, "unknown")

    explanation = generate_explanation(classification, auth_prob)

    # Cleanup
    os.remove(temp_path)

    return jsonify({
        "classification": classification,
        "confidence_score": float(auth_prob),
        "detected_language": language,
        "explanation": explanation
    })

if __name__ == "__main__":
    app.run(debug=True)
