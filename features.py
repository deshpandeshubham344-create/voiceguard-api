# features.py
import numpy as np
import wave

def extract_features(audio_path):
    with wave.open(audio_path, "rb") as wf:
        frames = wf.readframes(wf.getnframes())
        signal = np.frombuffer(frames, dtype=np.int16)

    # Simple lightweight features
    mean = float(np.mean(signal))
    std = float(np.std(signal))
    max_val = float(np.max(signal))
    min_val = float(np.min(signal))

    return [mean, std, max_val, min_val]

