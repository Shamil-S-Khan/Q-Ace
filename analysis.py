import os
import subprocess
import speech_recognition as sr

from facial_emotion_analyzer import analyze_video
from emotion_predict import predict_emotion
from nlp_model import assess_answer

def analyze_facial_emotion_wrapper(video_path, model):
    return analyze_video(video_path, model)

def analyze_vocal_emotion_wrapper(video_path, model):
    if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
        return "Error: Invalid or empty video file."
    try:
        audio_path = os.path.join("uploads", "recording.wav")
        command = ["ffmpeg", "-i", video_path, "-y", audio_path]
        subprocess.run(command, check=True)

        return predict_emotion(audio_path, model)
    except Exception as e:
        print(f"Error in vocal emotion analysis: {e}")
        return "Error"

def transcribe_audio_wrapper(video_path):
    if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
        return "Error: Invalid or empty video file."
    try:
        audio_path = os.path.join("uploads", "recording.wav")
        command = ["ffmpeg", "-i", video_path, "-y", audio_path]
        subprocess.run(command, check=True)

        r = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio_data = r.record(source)
            text = r.recognize_google(audio_data)
            return text
    except Exception as e:
        print(f"Error in audio transcription: {e}")
        return "Error in transcription"

def analyze_answer_quality_wrapper(question, answer, model):
    return assess_answer(question, answer, model)