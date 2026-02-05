import numpy as np
import soundfile as sf

EXPECTED_FEATURES = 384

def extract_features(audio_path):
    signal, sr = sf.read(audio_path)

    if signal.ndim > 1:
        signal = signal.mean(axis=1)

    # Normalize
    signal = signal.astype(np.float32)
    signal = signal / (np.max(np.abs(signal)) + 1e-9)

    # Chunk into fixed-size features
    features = np.interp(
        np.linspace(0, len(signal), EXPECTED_FEATURES),
        np.arange(len(signal)),
        signal
    )

    return features.tolist()

