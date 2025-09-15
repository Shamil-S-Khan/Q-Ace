import cv2
import torch
import torch.nn as nn
import numpy as np

# ------------------- Model Definition (same as training) -------------------
class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 6 * 6, 256)
        self.fc2 = nn.Linear(256, 7)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ------------------- Emotion Labels -------------------
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ------------------- Load Face Detector -------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def analyze_video(video_path, model):
    cap = cv2.VideoCapture(video_path)
    emotion_counts = {emotion: 0 for emotion in EMOTIONS}
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))
            face = face.astype("float32") / 255.0
            face = torch.tensor(face).unsqueeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(face)
                _, predicted = torch.max(output, 1)
                emotion = EMOTIONS[predicted.item()]
                emotion_counts[emotion] += 1

    cap.release()

    if frame_count == 0:
        return "No_Face_Detected"
        
    # Determine dominant emotion
    dominant_emotion = max(emotion_counts, key=emotion_counts.get)
    return dominant_emotion

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python realtime_emotion.py <video_path>")
        sys.exit(1)
    video_path = sys.argv[1]
    emotion = analyze_video(video_path)
    print(f"Dominant emotion: {emotion}")
