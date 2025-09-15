import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib

def train_nlp_model():
    # Load the dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'interview_qa.csv')
    data = pd.read_csv(csv_path)

    # Split the data into training and testing sets
    X = data['Answer']
    y = data['Quality']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', SVC()),
    ])

    # Define the parameter grid for GridSearchCV
    parameters = {
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'tfidf__use_idf': (True, False),
        'clf__C': [1, 10, 100],
        'clf__kernel': ['linear', 'rbf']
    }

    # Create and run the grid search
    grid_search = GridSearchCV(pipeline, parameters, cv=2, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    # Print the best parameters and score
    print("Best parameters set found on development set:")
    print(grid_search.best_params_)

    # Evaluate the best model on the test set
    y_pred = grid_search.predict(X_test)
    print(f'Accuracy on test set: {accuracy_score(y_test, y_pred)}')

    # Create the models directory if it doesn't exist
    models_dir = os.path.join(script_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    # Save the best model (the entire pipeline)
    model_path = os.path.join(models_dir, 'nlp_model_pipeline.pkl')
    joblib.dump(grid_search.best_estimator_, model_path)

def assess_answer(question, answer, model):
    # Predict the quality of the answer
    prediction = model.predict([answer])
    return prediction[0]

if __name__ == '__main__':
    train_nlp_model()
