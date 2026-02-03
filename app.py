from flask import Flask, request, jsonify
import base64, os, joblib, subprocess
from features import extract_features

app = Flask(__name__)

auth_model = joblib.load("model/voiceguard_model.pkl")
lang_model = joblib.load("model/language_model.pkl")

LANG_MAP_REV = {
    0: "tamil",
    1: "english",
    2: "hindi",
    3: "malayalam",
    4: "telugu"
}

FFMPEG_PATH = "ffmpeg"  # works if ffmpeg in PATH

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

@app.route("/detect", methods=["POST"])
def detect():
    data = request.json
    audio_b64 = data["audio"]

    # Save input
    with open("temp_input", "wb") as f:
        f.write(base64.b64decode(audio_b64))

    # Convert to wav (10 seconds)
    subprocess.run([
        FFMPEG_PATH,
        "-y",
        "-i", "temp_input",
        "-t", "10",
        "-ar", "16000",
        "-ac", "1",
        "temp.wav"
    ])

    # Extract features
    features = extract_features("temp.wav")

    # AI vs Human
    auth_pred = auth_model.predict([features])[0]
    auth_prob = auth_model.predict_proba([features])[0].max()
    classification = "AI_GENERATED" if auth_pred == 1 else "HUMAN"

    # Language
    lang_pred = lang_model.predict([features])[0]
    language = LANG_MAP_REV[lang_pred]

    # Explanation
    explanation = generate_explanation(classification, auth_prob)

    os.remove("temp_input")
    os.remove("temp.wav")

    return jsonify({
        "classification": classification,
        "confidence_score": float(auth_prob),
        "detected_language": language,
        "explanation": explanation
    })

if __name__ == "__main__":
    app.run(debug=True)
