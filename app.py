import subprocess
print(subprocess.run(["which", "ffmpeg"], capture_output=True, text=True))
from flask import Flask, request, jsonify
import os, subprocess, joblib
from features import extract_features

app = Flask(__name__)

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
        return "The voice shows synthetic acoustic patterns."
    else:
        return "The voice shows natural human acoustic patterns."

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "VoiceGuard API running",
        "endpoint": "/detect",
        "method": "POST multipart/form-data with file"
    })

@app.route("/detect", methods=["POST"])
def detect():
    if "file" not in request.files:
        return jsonify({"error": "Send file as multipart/form-data with key 'file'"}), 400

    file = request.files["file"]
    input_path = "input_audio"
    wav_path = "temp.wav"
    file.save(input_path)

    # Convert ANY format → WAV
    subprocess.run([
        "ffmpeg", "-y",
        "-i", input_path,
        "-ar", "16000",
        "-ac", "1",
        wav_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    features = extract_features(wav_path)

    auth_pred = auth_model.predict([features])[0]
    auth_prob = auth_model.predict_proba([features])[0].max()
    classification = "AI_GENERATED" if auth_pred == 1 else "HUMAN"

    lang_pred = lang_model.predict([features])[0]
    language = LANG_MAP_REV.get(lang_pred, "unknown")

    explanation = generate_explanation(classification, auth_prob)

    os.remove(input_path)
    os.remove(wav_path)

    return jsonify({
        "classification": classification,
        "confidence_score": float(auth_prob),
        "detected_language": language,
        "explanation": explanation
    })

if __name__ == "__main__":
    app.run()

