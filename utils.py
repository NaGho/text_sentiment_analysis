import kagglehub
import pandas as pd
import numpy as np
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
nltk.download('all')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
analyzer = SentimentIntensityAnalyzer()

# create preprocess_text function
def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text.lower())
    # Remove stop words
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # Lemmatize the tokens
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    # Join the tokens back into a string
    processed_text = ' '.join(lemmatized_tokens)
    return processed_text

def nltk_get_sentiment(text):
    scores = analyzer.polarity_scores(text)
    sentiment = 1 if scores['pos'] > 0 else 0
    return sentiment


def train(X_train, y_train, classifier):
    """Train the sentiment analysis model"""
    print("Training model...")
    classifier.fit(X_train, y_train)

def evaluate(y_pred, y_test):
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\n F1-Score = ", f1_score(y_test, y_pred))
    return



def predict(text, vectorizer, classifier):
    """Predict sentiment for new text"""
    processed_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([processed_text])
    prediction = classifier.predict(vectorized_text)
    probability = classifier.predict_proba(vectorized_text)
    
    return {
        'sentiment': 'positive' if prediction[0] == 1 else 'negative',
        'confidence': max(probability[0])
    }

def tfid_predict(X_train, y_train, X_test):
    tfid_vectorizer = TfidfVectorizer(max_features=5000)
    classifier = LogisticRegression(max_iter=1000)
    # Vectorize text
    print("Vectorizing text...")
    X_train_vec = tfid_vectorizer.fit_transform(X_train)
    X_test_vec = tfid_vectorizer.transform(X_test)

    train(X_train_vec, y_train, classifier)

    return classifier.predict(X_test_vec)