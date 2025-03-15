from bs4 import BeautifulSoup
import re
import torch
import pandas as pd
import numpy as np
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from wordcloud import WordCloud
from collections import Counter


tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')


def extract_reviews(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    regex = re.compile('.*comment.*')
    results = soup.find_all('p', {'class':regex})
    reviews = [result.text for result in results]
    return reviews


def sentiment_score(reviews):
    tokens = tokenizer.encode(reviews, return_tensors='pt')
    result = model(tokens)
    return int(torch.argmax(result.logits))+1


def make_review_df(reviews):
    df = pd.DataFrame(np.array(reviews), columns=['review'])
    df['sentiment'] = df['review'].apply(lambda x: sentiment_score(x[:600]))
    return df


def get_emotions(texts):
    """
    Analyze emotions in a list of texts using a pre-trained emotion detection model.
    
    Args:
        texts (list): List of text strings to analyze
        
    Returns:
        pandas.DataFrame: DataFrame with emotion scores for each text (without the text column)
    """
    # Initialize the emotion analysis pipeline
    emotion_analyzer = pipeline(
        "text-classification", 
        model="j-hartmann/emotion-english-distilroberta-base", 
        return_all_scores=True
    )
    
    # Process each text and extract emotion scores
    all_emotions = []
    for text in texts:
        # Truncate text if needed (model has a token limit)
        truncated_text = text[:512]
        
        # Get emotion scores
        result = emotion_analyzer(truncated_text)[0]
        
        # Convert to dictionary format
        emotions_dict = {item['label']: item['score'] for item in result}
        all_emotions.append(emotions_dict)
    
    # Create DataFrame from results
    emotions_df = pd.DataFrame(all_emotions)
    return emotions_df


def calculate_sentiment_percentages(df):
    """
    Calculate the percentage of positive, negative, and neutral sentiments.
    
    Args:
        df (pandas.DataFrame): DataFrame containing sentiment scores (1-5)
        
    Returns:
        dict: Dictionary with percentages for positive, negative, and neutral sentiments
    """
    # Count the number of reviews in each category
    total_reviews = len(df)
    
    # Positive: sentiment scores 4-5
    positive_count = len(df[df['sentiment'].isin([4, 5])])
    
    # Negative: sentiment scores 1-2
    negative_count = len(df[df['sentiment'].isin([1, 2])])
    
    # Neutral: sentiment score 3
    neutral_count = len(df[df['sentiment'] == 3])
    
    # Calculate percentages
    percentages = {
        'positive': (positive_count / total_reviews) * 100,
        'neutral': (neutral_count / total_reviews) * 100,
        'negative': (negative_count / total_reviews) * 100
    }
    
    return percentages

    
def remove_filler_words(reviews):
    with open('stop_hinglish.txt', 'r', encoding='utf-8') as f:
        stop_words = set(f.read().split())

    cleaned_reviews = []
    for review in reviews:
        words = [word for word in review.lower().split() if word not in stop_words]
        cleaned_reviews.append(" ".join(words))

    return cleaned_reviews

def create_wordcloud(reviews):
    reviews = remove_filler_words(reviews)
    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
    df_wc = wc.generate(" ".join(reviews))
    return df_wc
