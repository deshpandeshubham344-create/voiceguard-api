import numpy as np
from scipy.io import wavfile

EXPECTED_FEATURES = 384  # must match training

def extract_features(audio_path):
    sr, signal = wavfile.read(audio_path)

    # Convert stereo to mono
    if signal.ndim > 1:
        signal = signal.mean(axis=1)

    signal = signal.astype(np.float32)

    # Normalize
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        signal /= max_val

    # Force fixed-length feature vector
    x_old = np.linspace(0, 1, num=len(signal))
    x_new = np.linspace(0, 1, num=EXPECTED_FEATURES)

    features = np.interp(x_new, x_old, signal)
    return features.tolist()
