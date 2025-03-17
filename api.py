# Importing libraries
from fastapi import FastAPI
from pydantic import BaseModel
from utils import *

app = FastAPI

# Input model for the API
class CompanyInput(BaseModel):
    name: str

# API endpoint to analyze news
@app.post("/analyze")
def analyze_news(company: CompanyInput):
    # Fetch news articles
    articles = scrape_news(company.name)

    # Perform sentiment analysis on each article
    for article in articles:
        article['sentiment'] = analyze_sentiment(article['summary'])

    # Comparative analysis of sentiments
    sentiment_distribution = compare_sentiments(articles)

    # Return results
    return {
        "company": company.name,
        "articles": articles,
        "sentiment_distribution": sentiment_distribution
    }

# Run the API
if __name__=="__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port = 8000)