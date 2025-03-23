# Importing necessary libraries
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
from gtts import gTTS
from collections import Counter
import os
from dotenv import load_dotenv
import json
from huggingface_hub import InferenceClient
import re

# Load API keys from environment variables
load_dotenv()
classification_api_key = os.getenv("CLASSIFICATION_API_KEY")
summarization_api_key = os.getenv("SUMMARIZATION_API_KEY")

# Hugging Face API URL and headers for classification
API_URL_CLASS = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
HEADERS_CLASS = {"Authorization": f"Bearer {classification_api_key}"}


def read_labels(file_path):
    """
    Reads candidate labels from a text file.

    Args:
        file_path (str): Path to the text file containing labels.

    Returns:
        list: A list of labels, stripped of leading/trailing whitespace.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        labels = [line.strip() for line in f.readlines() if line.strip()]
    return labels


def classify_summary(summary, labels):
    """
    Uses Hugging Face API for zero-shot classification in batches of 10 labels.

    Args:
        summary (str): The text to classify.
        labels (list): List of candidate labels for classification.

    Returns:
        list: Top 5 labels with the highest classification scores.
    """
    results = {}

    # Process labels in batches of 10
    for i in range(0, len(labels), 10):
        batch = labels[i:i + 10]  # Take a batch of up to 10 labels

        payload = {
            "inputs": summary,
            "parameters": {"candidate_labels": batch}
        }

        response = requests.post(API_URL_CLASS, headers=HEADERS_CLASS, json=payload)

        if response.status_code == 200:
            classification = response.json()
            for label, score in zip(classification["labels"], classification["scores"]):
                results[label] = score  # Store scores for each label
        else:
            print(f"Error: {response.status_code}, {response.text}")

    # Sort results by highest probability
    sorted_results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))

    return list(sorted_results.keys())[:5]


def scrape_news(company_name):
    """
    Scrapes news articles related to a given company from Bing News RSS feed.

    Args:
        company_name (str): Name of the company to search for.

    Returns:
        list: A list of dictionaries containing article details (title, summary, topics).
              Returns an error dictionary if the request fails.
    """
    # Construct the Bing News RSS feed URL
    search_url = f"https://www.bing.com/news/search?q={company_name}&format=rss"
    headers = {'User-Agent': 'Mozilla/5.0'}  # Add a user-agent to avoid being blocked

    try:
        # Send a GET request to fetch the RSS feed
        response = requests.get(search_url, headers=headers)
        response.raise_for_status()  # Raise an exception for HTTP errors
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to fetch news: {e}"}

    # Parse the RSS feed using BeautifulSoup
    soup = BeautifulSoup(response.content, 'xml')  # Use 'xml' parser for RSS feeds
    items = soup.find_all('item')

    articles = []
    for item in items[:10]:  # Limit to the first 10 articles
        # Extract the title
        title = item.title.text.strip() if item.title else "No title available"

        # Extract the description
        description = item.description.text.strip() if item.description else "No summary available"

        # Extract topics using zero-shot classification
        labels = read_labels('labels.txt')
        topics = classify_summary(description, labels)

        # Append the article details to the list
        articles.append({
            "title": title,
            "summary": description,
            "topics": topics
        })

    return articles


def analyze_sentiment(text):
    """
    Performs sentiment analysis on the given text.

    Args:
        text (str): The text to analyze.

    Returns:
        str: The sentiment label (POSITIVE, NEGATIVE, or NEUTRAL).
    """
    sentiment_pipeline = pipeline("sentiment-analysis")
    result = sentiment_pipeline(text)[0]
    return result['label']


def compare_sentiments(articles):
    """
    Compares sentiment distribution across multiple articles.

    Args:
        articles (list): List of dictionaries, each containing article details.

    Returns:
        Counter: A Counter object with sentiment distribution.
    """
    sentiments = [article['sentiment'] for article in articles]
    sentiment_distribution = Counter(sentiments)
    return sentiment_distribution


def extract_common_topics(articles):
    """
    Extracts common topics present in all articles.

    Args:
        articles (list): List of dictionaries, each containing article details.

    Returns:
        set: A set of common topics. Returns None if no common topics exist.
    """
    if not articles:
        return None

    # Extract the topics from each article
    topics_lists = [article["topics"] for article in articles]

    # Find the common topics
    common_topics = set(topics_lists[0]).intersection(*topics_lists[1:])

    if len(common_topics) == 0:
        return None
    return common_topics


def extract_unique_topics(articles):
    """
    Extracts unique topics for each article.

    Args:
        articles (list): List of dictionaries, each containing article details.

    Returns:
        list: A list where each element is a set of unique topics for the corresponding article.
              If no unique topics exist for an article, None is included in the list.
    """
    unique_topics_list = []

    for article in articles:
        title = article["title"]
        topics = article["topics"]

        # Find unique topics for this article by comparing with all other articles
        other_topics = set()
        for other_article in articles:
            if other_article["title"] != title:
                other_topics.update(other_article["topics"])

        # Unique topics are those in the current article but not in others
        unique_topics = set(topics) - other_topics

        # If no unique topics, append None; otherwise, append the set of unique topics
        if not unique_topics:
            unique_topics_list.append(None)
        else:
            unique_topics_list.append(unique_topics)

    return unique_topics_list


# Load API Key for text summarization
load_dotenv()
summarization_api_key = os.getenv("SUMMARIZATION_API_KEY")

client = InferenceClient(api_key=f"{summarization_api_key}")


def comparisons_and_impact(articles):
    """
    Generates structured comparison and impact analysis from article summaries.

    Args:
        articles (list): List of dictionaries, each containing article details.

    Returns:
        str: A structured comparison and impact analysis.
    """
    summaries = [article["summary"] for article in articles]
    combined_summary = "\n".join(summaries)

    prompt = (
        f"Compare the following news articles and summarize their impact.\n"
        f"Extract key comparisons and overall impact in a structured format.\n\n"
        f"Articles:\n{combined_summary}\n\n"
        f"Format:\
        Impact: [Overall impact derived from the articles]\
        Key Comparisons:\
        [Comparison 1 between the two articles]\
        [Comparison 2 between the two articles]\n"
    )

    response = client.post(
        model="mistralai/Mistral-7B-Instruct-v0.1",
        json={
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 300,
                "temperature": 0.5,
                "top_k": 10,
                "top_p": 0.9
            }
        }
    )

    # Ensure response is valid JSON
    try:
        if isinstance(response, bytes):
            response = response.decode("utf-8")
        response_json = json.loads(response) if isinstance(response, str) else response
        if isinstance(response_json, list) and "generated_text" in response_json[0]:
            return response_json[0]["generated_text"]
        else:
            return "Unexpected API response format."
    except (json.JSONDecodeError, IndexError, KeyError) as e:
        return f"Error processing API response: {str(e)}"


def extract_impact_and_comparisons(text):
    """
    Extracts impact and comparisons from the generated text.

    Args:
        text (str): The text containing impact and comparisons.

    Returns:
        list: A list of impact and comparison statements.
    """
    comparisons_pattern = re.search(r'(\nImpact:\s*.*)', text, re.DOTALL)
    comparisons = comparisons_pattern.group(1).strip() if comparisons_pattern else "Comparisons not found"

    return list(filter(None, comparisons.split('\n')))  # Remove empty strings


# Load API Key from Environment
load_dotenv()
final_sum_api_key = os.getenv("FINAL_SUM_KEY")

# Hugging Face API details
API_URL_FINALSUM = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
HEADERS_FINALSUM = {"Authorization": f"Bearer {final_sum_api_key}"}  # Replace with actual API key


def get_detailed_sentiment_summary(articles):
    """
    Generates a detailed sentiment summary using Mixtral-8x7B.

    Args:
        articles (list): List of dictionaries, each containing article details.

    Returns:
        str: A natural-sounding sentiment summary.
    """
    total_articles = len(articles)
    sentiment_distribution = compare_sentiments(articles)
    positive_count = sentiment_distribution.get('POSITIVE', 0)
    negative_count = sentiment_distribution.get('NEGATIVE', 0)
    neutral_count = sentiment_distribution.get('NEUTRAL', 0)

    summaries = [article["summary"] for article in articles]
    combined_summary = "\n".join(summaries)

    # Instruction prompt for Mixtral
    input_text = (
        f"The dataset consists of {total_articles} articles\n"
        f"Sentiment Breakdown:\n"
        f"- {positive_count} articles express a positive sentiment.\n"
        f"- {negative_count} articles express a negative sentiment.\n"
        f"- {neutral_count} articles are neutral.\n\n"
        f"[Provide a detailed, **natural-sounding** one line statement of the overall sentiment from this input text. Explain the trends and implications in a way that's easy to understand.]"
    )

    payload = {
        "inputs": input_text,
        "parameters": {"max_length": 40, "temperature": 0.7, "do_sample": False}
    }

    response = requests.post(API_URL_FINALSUM, headers=HEADERS_FINALSUM, json=payload)

    if response.status_code == 200:
        response_json = response.json()
        if isinstance(response_json, list) and "generated_text" in response_json[0]:
            sentiment_summary = response_json[0]["generated_text"]
        elif isinstance(response_json, dict) and "generated_text" in response_json:
            sentiment_summary = response_json["generated_text"]
        else:
            sentiment_summary = "Unexpected API response format."
    else:
        sentiment_summary = f"API request failed with status {response.status_code}: {response.text}"

    return sentiment_summary.split('\n\n')[-1].split(']')[-1]


# Load API Key from Environment
load_dotenv()
trans_key = os.getenv("TRANSLATION_KEY")

# Hugging Face API details
API_URL_TRANS = "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-en-hi"
HEADERS_TRANS = {"Authorization": f"Bearer {trans_key}"}


def translate_en_to_hi(sentence):
    """
    Translates an English sentence to Hindi using Hugging Face API.

    Args:
        sentence (str): The English sentence to translate.

    Returns:
        str: The translated Hindi sentence.
    """
    payload = {"inputs": sentence}

    response = requests.post(API_URL_TRANS, headers=HEADERS_TRANS, json=payload)

    if response.status_code == 200:
        translated_text = response.json()[0]['translation_text']
        return translated_text
    else:
        return f"API request failed with status {response.status_code}: {response.text}"


def generate_hindi_tts(text):
    """
    Generates Hindi audio from the given text using gTTS.

    Args:
        text (str): The text to convert to speech.

    Returns:
        str: The file path of the generated audio file.
    """
    tts = gTTS(text, lang='hi')
    tts.save("output.mp3")
    return "output.mp3"