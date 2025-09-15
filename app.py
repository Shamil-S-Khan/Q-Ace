from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import torch
import joblib
from collections import Counter

from analysis import (
    analyze_facial_emotion_wrapper,
    analyze_vocal_emotion_wrapper,
    transcribe_audio_wrapper,
    analyze_answer_quality_wrapper
)
from facial_emotion_analyzer import EmotionCNN

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# --- Load Models ---
VOCAL_EMOTION_MODEL = load_model("models/cnn_emotion_model.keras")
FACIAL_EMOTION_MODEL = EmotionCNN()
FACIAL_EMOTION_MODEL.load_state_dict(torch.load("models/emotion_cnn_improved.pth", map_location=torch.device('cpu')))
FACIAL_EMOTION_MODEL.eval()
NLP_MODEL = joblib.load("models/nlp_model_pipeline.pkl")

# --- Interview Questions ---
interview_questions = [
    "Tell me about yourself.",
    "What are your strengths?",
    "What are your weaknesses?",
    "Tell me about a time you failed.",
    "Where do you see yourself in 5 years?",
    "Why do you want to work for this company?",
    "Describe a challenge you faced at work and how you dealt with it.",
    "What is your greatest professional achievement?",
    "How do you handle pressure or stressful situations?",
    "What are your salary expectations?",
    "Why should we hire you?",
    "What do you know about our company?",
    "How do you handle conflict with a coworker?",
    "What is your leadership style?",
    "What are your long-term career goals?"
]

def calculate_question_rating(facial_emotion, vocal_emotion, answer_quality):
    score = 0
    if facial_emotion == "Happy": score += 2
    elif facial_emotion == "Neutral": score += 1
    elif facial_emotion in ["Sad", "Angry", "Fearful"]: score -= 1
    if vocal_emotion in ["happy", "calm"]: score += 2
    elif vocal_emotion == "neutral": score += 1
    elif vocal_emotion in ["sad", "angry", "fearful"]: score -= 1
    if answer_quality == "Good": score += 3
    elif answer_quality == "Average": score += 1
    elif answer_quality == "Poor": score -= 2
    return score

def calculate_final_rating(scores):
    total_score = sum(scores)
    if total_score >= 45: return "Excellent"
    elif total_score >= 30: return "Good"
    elif total_score >= 15: return "Average"
    elif total_score >= 0: return "Needs Improvement"
    else: return "Poor"

def generate_improvement_feedback(results):
    feedback = []
    
    # Facial Emotion Feedback
    facial_emotions = [res['facial_emotion'] for res in results]
    facial_emotion_counts = Counter(facial_emotions)
    negative_facial_emotions = facial_emotion_counts.get("Sad", 0) + facial_emotion_counts.get("Angry", 0) + facial_emotion_counts.get("Fearful", 0)
    if negative_facial_emotions > 5: # If more than a third of questions have negative facial emotion
        feedback.append("Consider practicing a more neutral or positive facial expression. You appeared sad, angry, or fearful in several responses.")

    # Vocal Emotion Feedback
    vocal_emotions = [res['vocal_emotion'] for res in results]
    vocal_emotion_counts = Counter(vocal_emotions)
    negative_vocal_emotions = vocal_emotion_counts.get("sad", 0) + vocal_emotion_counts.get("angry", 0) + vocal_emotion_counts.get("fearful", 0)
    if negative_vocal_emotions > 5: # If more than a third of questions have negative vocal emotion
        feedback.append("Your tone of voice came across as sad, angry, or fearful at times. Try to speak with a calmer and more confident tone.")

    # Answer Quality Feedback
    answer_qualities = [res['answer_quality'] for res in results]
    answer_quality_counts = Counter(answer_qualities)
    poor_answers = answer_quality_counts.get("Poor", 0)
    if poor_answers > 3: # If more than 3 answers are poor
        feedback.append("Some of your answers were not very detailed. A great way to improve this is to use the STAR method (Situation, Task, Action, Result) to structure your answers about past experiences.")

    # Low score questions
    low_scoring_questions = [res['question'] for res in results if res['score'] <= 0]
    if len(low_scoring_questions) > 0:
        feedback.append(f"You particularly struggled with the following questions: {', '.join(low_scoring_questions)}. It would be beneficial to practice your responses to these.")

    if not feedback:
        feedback.append("Great job! No major areas for improvement were detected. Keep practicing to stay sharp!")

    return feedback

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start')
def start():
    session['current_question'] = 0
    session['scores'] = []
    session['results'] = []
    return redirect(url_for('question'))

@app.route('/question')
def question():
    if 'current_question' not in session or session['current_question'] >= len(interview_questions):
        return redirect(url_for('results'))
    
    question_index = session['current_question']
    question_text = interview_questions[question_index]
    return render_template('question.html', question_number=question_index + 1, question_text=question_text)

@app.route('/record', methods=['POST'])
def record():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file found'}), 400

    video_file = request.files['video']
    upload_folder = 'uploads'
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)

    video_path = os.path.join(upload_folder, f'recording_{session["current_question"]}.webm')
    video_file.save(video_path)

    question_index = session['current_question']
    question = interview_questions[question_index]

    facial_emotion = analyze_facial_emotion_wrapper(video_path, FACIAL_EMOTION_MODEL)
    vocal_emotion = analyze_vocal_emotion_wrapper(video_path, VOCAL_EMOTION_MODEL)
    answer_text = transcribe_audio_wrapper(video_path)
    answer_quality = analyze_answer_quality_wrapper(question, answer_text, NLP_MODEL)
    
    score = calculate_question_rating(facial_emotion, vocal_emotion, answer_quality)
    
    scores = session.get('scores', [])
    scores.append(score)
    session['scores'] = scores

    results = session.get('results', [])
    results.append({
        'question': question,
        'facial_emotion': facial_emotion,
        'vocal_emotion': vocal_emotion,
        'answer_quality': answer_quality,
        'answer_text': answer_text,
        'score': score
    })
    session['results'] = results
    
    session['current_question'] += 1
    
    return jsonify({'status': 'success', 'next_url': url_for('question')})

@app.route('/results')
def results():
    results_data = session.get('results', [])
    final_rating = calculate_final_rating(session.get('scores', []))
    improvement_feedback = generate_improvement_feedback(results_data)
    return render_template('results.html', results=results_data, final_rating=final_rating, improvement_feedback=improvement_feedback)

if __name__ == '__main__':
    app.run(debug=True)
