from flask import Flask, render_template, request, jsonify
import os
import tensorflow as tf
print(tf.__version__)
from tensorflow.keras.models import load_model
import torch
import joblib

from analysis import (
    analyze_facial_emotion_wrapper,
    analyze_vocal_emotion_wrapper,
    transcribe_audio_wrapper,
    analyze_answer_quality_wrapper
)
from facial_emotion_analyzer import EmotionCNN # Import the EmotionCNN class

app = Flask(__name__)

# --- Load Models ---
VOCAL_EMOTION_MODEL = load_model("models/cnn_emotion_model.keras")

# Load Facial Emotion Model
FACIAL_EMOTION_MODEL = EmotionCNN() # Instantiate the model
FACIAL_EMOTION_MODEL.load_state_dict(torch.load("models/emotion_cnn_improved.pth", map_location=torch.device('cpu')))
FACIAL_EMOTION_MODEL.eval() # Set model to evaluation mode

NLP_MODEL = joblib.load("models/nlp_model_pipeline.pkl")

# Dummy interview questions
interview_questions = [
    "Tell me about yourself.",
    "What are your strengths and weaknesses?",
    "Describe a challenge you faced at work and how you dealt with it."
]

def calculate_final_rating(facial_emotion, vocal_emotion, answer_quality):
    score = 0

    # Facial Emotion Scoring
    if facial_emotion == "Happy":
        score += 2
    elif facial_emotion == "Neutral":
        score += 1
    elif facial_emotion in ["Sad", "Angry", "Fearful"]:
        score -= 1

    # Vocal Emotion Scoring
    if vocal_emotion in ["happy", "calm"]:
        score += 2
    elif vocal_emotion == "neutral":
        score += 1
    elif vocal_emotion in ["sad", "angry", "fearful"]:
        score -= 1

    # Answer Quality Scoring
    if answer_quality == "Good":
        score += 3
    elif answer_quality == "Average":
        score += 1
    elif answer_quality == "Poor":
        score -= 2

    # Final Rating
    if score >= 5:
        return "Excellent"
    elif score >= 3:
        return "Good"
    elif score >= 1:
        return "Average"
    elif score >= -1:
        return "Needs Improvement"
    else:
        return "Poor"

@app.route('/')
def index():
    return render_template('index.html', questions=interview_questions)

@app.route('/record', methods=['POST'])
def record():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file found'}), 400

    video_file = request.files['video']
    upload_folder = 'uploads'
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)

    video_path = os.path.join(upload_folder, 'recording.webm')
    video_file.save(video_path)

    question = interview_questions[0]

    facial_emotion = analyze_facial_emotion_wrapper(video_path, FACIAL_EMOTION_MODEL)
    vocal_emotion = analyze_vocal_emotion_wrapper(video_path, VOCAL_EMOTION_MODEL)
    answer_text = transcribe_audio_wrapper(video_path)
    answer_quality = analyze_answer_quality_wrapper(question, answer_text, NLP_MODEL)

    final_rating = calculate_final_rating(facial_emotion, vocal_emotion, answer_quality)

    return jsonify({
        'facial_emotion': facial_emotion,
        'vocal_emotion': vocal_emotion,
        'answer_quality': answer_quality,
        'answer_text': answer_text, # Added this line
        'final_rating': final_rating
    })

if __name__ == '__main__':
    app.run(debug=True)
