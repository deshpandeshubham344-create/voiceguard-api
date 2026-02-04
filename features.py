import librosa
import numpy as np

def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=192)

    mean = np.mean(mfcc, axis=1)
    std = np.std(mfcc, axis=1)

    features = np.concatenate([mean, std])
    return features.tolist()
