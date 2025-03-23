# Importing necessary libraries
from fastapi import FastAPI
from pydantic import BaseModel
from utils import *

# Initialising FastAPI
app = FastAPI()


# Input model for the API
class CompanyInput(BaseModel):
    """
    Pydantic model for input validation.
    
    Attributes:
        name (str): The name of the company to analyze.
    """
    name: str


# API endpoint to analyze news
@app.post("/analyze")
def analyze_news(company: CompanyInput):
    """
    API endpoint to fetch and analyze news articles for a given company.

    Args:
        company (CompanyInput): Input model containing the company name.

    Returns:
        dict: A structured response containing:
            - Company name
            - List of articles with titles, summaries, sentiments, and topics
            - Sentiment distribution across articles
            - Common topics across articles
            - Unique topics for each article
    """
    # Fetch news articles related to the company
    articles = scrape_news(company.name)

    # Perform sentiment analysis on each article
    for article in articles:
        article['sentiment'] = analyze_sentiment(article['summary'])

    # Perform comparative analysis of sentiments across articles
    sentiment_distribution = compare_sentiments(articles)

    # Extract common topics across all articles
    common_topics = extract_common_topics(articles)

    # Extract unique topics for each article
    unique_topics = extract_unique_topics(articles)

    # Return the results in a structured format
    return {
        "company": company.name,
        "articles": articles,
        "sentiment_distribution": sentiment_distribution,
        "common_topics": common_topics,
        "unique_topics": unique_topics,
    }


# Run the API
if __name__ == "__main__":
    """
    Entry point to run the FastAPI application using Uvicorn.
    """
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)