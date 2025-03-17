# Importing all necessary libraries
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
from gtts import gTTS
from collections import Counter
import re

# Defining a function to scrape news
def scrape_news(company_name):
    # Construct the Google News RSS feed URL
    search_url = f"https://news.google.com/rss/search?q={company_name}"
    headers = {'User-Agent': 'Mozilla/5.0'}  # Add a user-agent to avoid being blocked
    
    try:
        # Send a GET request to fetch the RSS feed
        response = requests.get(search_url, headers=headers)
        response.raise_for_status()  # Raise an exception for HTTP errors
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to fetch news: {e}"}
    
    # Parse the RSS feed using BeautifulSoup
    soup = BeautifulSoup(response.content, 'lxml')  # Use 'xml' parser for RSS feeds
    items = soup.find_all('item')
    
    articles = []
    for item in items[:10]:
        # Extract the title
        title = item.title.text.strip() if item.title else "No title available"
                
        # Extract the summary/description
        description = item.description.text.strip() if item.description else "No summary available"
        
        # Append the article details to the list
        articles.append({
            "title": title,
            "summary": " ".join(re.findall('target="_blank">(.+?)<', description )),
        })
    
    return articles

# Function to perform sentiment analysis on articles
def analyze_sentiment(text):
    sentiment_pipeline = pipeline("sentiment-analysis")
    result = sentiment_pipeline(text)[0]
    return result['label']

# Function to compare sentiments across multiple articles
def compare_sentiments(articles):
    sentiments = [article['sentiment'] for article in articles]
    sentiment_distribution = Counter(sentiments)
    return sentiment_distribution

# Function to generate Hindi Text-To_Speech
def generate_hindi_tts(text):
    tts = gTTS(text, lang = 'hi')
    tts.save("output.mp3")
    return "output.mp3"