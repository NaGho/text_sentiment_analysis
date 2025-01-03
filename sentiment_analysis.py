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
    preprocess_text, train, evaluate, predict, nltk_get_sentiment,
    tfid_predict
)
from llm_utils import LLM_predict
nltk.download('all')


def main():
    # Download latest version
    path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")

    print("Path to dataset files:", path)

    df = pd.read_csv(path + "/IMDB Dataset.csv").iloc[:20]
    df['sentiment'] = df['sentiment'].map({'negative': 0, 'positive': 1})


    
    tqdm.pandas()
    # clean/preprocess the review text
    df['text_cleaned'] = df['review'].progress_apply(preprocess_text)

    # df['nltk_sentiment'] = df['review_cleaned'].progress_apply(nltk_get_sentiment)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['text_cleaned'], df['sentiment'], test_size=0.2, random_state=42
    )

    print("---------> LLM evaluation")
    llm_y_pred = LLM_predict(X_train, y_train, X_test)
    evaluate(llm_y_pred, y_test)

    
    print("------------> TFID evaluation:")
    tfid_y_pred = tfid_predict(X_train, y_train, X_test) 
    evaluate(tfid_y_pred, y_test)
    
    print("------------> nltk sentiment evaluation:")
    nltk_sentiment_y_pred = [nltk_get_sentiment(txt) for txt in X_test]
    evaluate(nltk_sentiment_y_pred, y_test)

    print("------------> nltk sentiment & TFID evaluation:")
    mixed_y_pred = tfid_y_pred * nltk_sentiment_y_pred
    evaluate(mixed_y_pred, y_test)
    
    # # Example prediction
    # sample_text = "This product is fantastic!"
    # result = predict(sample_text, tfid_vectorizer, classifier)
    # print(f"\nSample prediction for '{sample_text}':")
    # print(f"Sentiment: {result['sentiment']}")
    # print(f"Confidence: {result['confidence']:.2f}")

    

if __name__ == "__main__":
    main()