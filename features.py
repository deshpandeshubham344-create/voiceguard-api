# features.py
import numpy as np
import librosa

def extract_features(audio_path):
    # Load audio (force 16kHz mono)
    y, sr = librosa.load(audio_path, sr=16000, mono=True)

    # Extract 24 MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=24)

    # Take first 16 frames (24 x 16 = 384)
    if mfcc.shape[1] < 16:
        pad_width = 16 - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0,0),(0,pad_width)), mode='constant')

    mfcc = mfcc[:, :16]

    # Flatten to 384 features
    features = mfcc.flatten()

    return features.tolist()


