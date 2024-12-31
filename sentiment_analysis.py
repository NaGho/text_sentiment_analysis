import kagglehub
import pandas as pd
import numpy as np
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
from utils import (
    preprocess_text, train, evaluate, predict, nltk_get_sentiment
)
nltk.download('all')


def main():
    # Download latest version
    path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")

    print("Path to dataset files:", path)

    df = pd.read_csv(path + "/IMDB Dataset.csv").iloc[:20]
    df['sentiment'] = df['sentiment'].map({'negative': 0, 'positive': 1})

    
    tfid_vectorizer = TfidfVectorizer(max_features=5000)
    classifier = LogisticRegression(max_iter=1000)
    
    tqdm.pandas()
    # clean/preprocess the review text
    df['text_cleaned'] = df['review'].progress_apply(preprocess_text)

    # df['nltk_sentiment'] = df['review_cleaned'].progress_apply(nltk_get_sentiment)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['text_cleaned'],
        df['sentiment'],
        test_size=0.2,
        random_state=42
    )

    # Vectorize text
    print("Vectorizing text...")
    X_train_vec = tfid_vectorizer.fit_transform(X_train)
    X_test_vec = tfid_vectorizer.transform(X_test)

    train(X_train_vec, y_train, classifier)
    
    tfid_y_pred = classifier.predict(X_test_vec)
    
    print("\TFID Evaluation:")
    evaluate(tfid_y_pred, y_test)
    
    print("\TFID Evaluation:")
    nltk_sentiment_pred = nltk_get_sentiment()
    # Example prediction
    sample_text = "This product is fantastic!"
    result = predict(sample_text, tfid_vectorizer, classifier)
    print(f"\nSample prediction for '{sample_text}':")
    print(f"Sentiment: {result['sentiment']}")
    print(f"Confidence: {result['confidence']:.2f}")

    

if __name__ == "__main__":
    main()