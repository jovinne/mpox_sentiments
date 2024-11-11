from flask import Flask, jsonify
from flask_cors import CORS 
from flask import render_template
import csv
import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import requests
import json
from newsapi import NewsApiClient
import re
from collections import Counter
import os

# nltk.download('stopwords', download_dir='./nltk_data')
# nltk.download('punkt', download_dir='./nltk_data')
# nltk.download('vader_lexicon', download_dir='./nltk_data')

nltk.data.path.append(os.path.join(os.path.dirname(__file__), 'nltk_data'))

from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer

app = Flask(__name__)
CORS(app, origins="*")

def get_articles():
    newsapi = NewsApiClient(api_key="bc95681e56644c11912d47c3dfef490a") 
    results = newsapi.get_everything(q="mpox", language='en')
    
    # Debug: Check the API response
    print(json.dumps(results, indent=2))

    if "articles" not in results or not results["articles"]:
        print("No articles found.")
        return  # Exit if no articles are found

    mpox_results = results["articles"]
    print(f"Found {len(mpox_results)} articles.")

    aggregated_dict = {}
    for result in mpox_results:
        for key, value in result.items():
            if key in aggregated_dict:
                aggregated_dict[key].append(value)
            else:
                aggregated_dict[key] = [value]

    aggregated_dict["source"] = [item["name"] for item in aggregated_dict["source"]]
    aggregated_dict["publishedAt"] = [item[:10] for item in aggregated_dict["publishedAt"]]

    df = pd.DataFrame(aggregated_dict)
    df.dropna()  # Ensure NaN values are dropped
    print(df.head())  # Print the DataFrame for debugging

    # Translate titles and preprocess
    # df["translated_title"] = df["title"].apply(lambda x: GoogleTranslator(source="auto", target="en").translate(x))
    # print("Success")
    stop_words = set(stopwords.words("english"))


    def preprocess(text):
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only alphabetic characters and spaces
    
        # Split the text into words
        words = text.lower().split()  # Split by spaces and convert to lowercase
        
        # Remove stopwords from the text
        cleaned_words = [word for word in words if word not in stop_words]
        
        # Join the words back into a string
        cleaned_text = " ".join(cleaned_words)
        
        return cleaned_text
    
    df["title_clean"] = df.apply(lambda x: preprocess(x["title"]), axis = 1)
    return df


def get_news_sentiments(df):
    analyzer = SentimentIntensityAnalyzer()

    def get_sentiment(text):
        scores = analyzer.polarity_scores(text)
        if scores["compound"] >= 0.05:
            sentiment = "Positive"
        elif scores["compound"] <= -0.05:
            sentiment = "Negative"
        else: 
            sentiment = "Neutral"
        return sentiment

    df["sentiment"] = df["title_clean"].apply(get_sentiment)
    sentiment_distribution = {"Positive": 0, "Negative": 0, "Neutral": 0}
    for sentiment in df["sentiment"]:
        if sentiment == "Positive":
            sentiment_distribution["Positive"] += 1
        elif sentiment == "Negative":
            sentiment_distribution["Negative"] += 1
        elif sentiment == "Neutral":
            sentiment_distribution["Neutral"] += 1
    print(sentiment_distribution)
    return sentiment_distribution

def get_word_counts(df):
    all_words =[]
    for index, row in df.iterrows():  # Using iterrows to iterate over DataFrame rows
        text = re.sub(r'[^a-zA-Z\s]', '', row['title_clean'].lower())  # Remove non-alphabetical characters
        words = text.split()  # Split by spaces into words

        # Add the words to the all_words list
        all_words.extend(words)
    stop_words = set(stopwords.words("english"))
    filtered_words = [word for word in all_words if word.isalnum() and word not in stop_words]

    # Count word frequencies
    word_counts = Counter(filtered_words)
    del word_counts["mpox"]
    del word_counts["monkeypox"]
    word_size = []
    for word, count in word_counts.items():
        # if count >= 5:
        word_size.append({"word": word, "size": count})
    print(word_size)
    return word_size

@app.route('/get_news_sentiments', methods=['GET'])
def generate_and_get_sentiments():
    articles = get_articles()
    sentiment_data = get_news_sentiments(articles)  # Get the sentiment data
    return jsonify(sentiment_data)

@app.route('/get_word_counts', methods=['GET'])
def generate_and_get_word_counts():
    articles = get_articles()
    word_counts = get_word_counts(articles)
    return jsonify(word_counts)

@app.route('/')
def index():
    return "Hello, World!"

@app.route('/wordcloud')
def wordcloud():
    return render_template('wordcloud.html')

@app.route('/piechart')
def piechart():
    return render_template('piechart.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
