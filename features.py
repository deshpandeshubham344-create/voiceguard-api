# features.py
import whisper
import torch
import numpy as np

model = whisper.load_model("tiny")   # << IMPORTANT: use tiny

def extract_features(path, max_seconds=5):
    audio = whisper.load_audio(path)

    # Hard cut
    max_samples = int(max_seconds * 16000)
    audio = audio[:max_samples]

    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio)

    with torch.no_grad():
        emb = model.encoder(mel.unsqueeze(0))

    # Mean pooling
    features = emb.mean(dim=1).squeeze().numpy()
    return features
