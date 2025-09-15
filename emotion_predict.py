import sys
import numpy as np
import librosa
from tensorflow.keras.models import load_model # type:ignore

# Config (adjust if needed)
SR = 22050
MAX_LEN = 217
N_MFCC = 13
MODEL_PATH = "models/cnn_emotion_model.keras"

LABEL_MAP = {
    0: "neutral",
    1: "calm",
    2: "happy",
    3: "sad",
    4: "angry",
    5: "fearful",
    6: "disgust",
    7: "surprised",
    8: "unknown"
}

def preprocess_audio(audio, sr=SR):
    audio_trimmed, _ = librosa.effects.trim(audio, top_db=60)
    mfcc = librosa.feature.mfcc(y=audio_trimmed, sr=sr, n_mfcc=N_MFCC).T
    if mfcc.shape[0] < MAX_LEN:
        mfcc = np.pad(mfcc, ((0, MAX_LEN - mfcc.shape[0]), (0, 0)), mode='constant')
    else:
        mfcc = mfcc[:MAX_LEN, :]
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-10)
    return mfcc

def predict_emotion(audio_path, model):
    # Load audio file
    audio, sr = librosa.load(audio_path, sr=SR)

    # Preprocess
    mfcc = preprocess_audio(audio, sr)

    # Prepare input for model
    X_input = mfcc[np.newaxis, ..., np.newaxis]

    # Predict
    pred = model.predict(X_input)
    pred_prob = pred[0]
    pred_class = np.argmax(pred_prob)
    max_prob = pred_prob[pred_class]

    emotion = LABEL_MAP.get(pred_class, "unknown")

    print(f"Predicted emotion: {emotion} (confidence: {max_prob:.2f})")
    return emotion

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python emotion_predict.py path_to_audio.wav")
        sys.exit(1)
    audio_path = sys.argv[1]
    predict_emotion(audio_path)
