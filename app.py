# Importing libraries
import streamlit as st
from utils import *

# Application Title
st.title("News Summarisation and Text-To-Speech Application")

# Input box for company name
company_name = st.text_input("## Enter the name of the company: ")

# Button to generate news
if st.button("Search"):
    if company_name:
        # Scrape news articles
        articles = scrape_news(company_name)
        
        # Perform sentiment analysis on each article
        for article in articles:
            article['sentiment'] = analyze_sentiment(article['summary'])

        # Display results
        st.write(f'# {company_name.upper()}')
        st.write(f"## News Articles")
        for article in articles:
            st.write(f"**Title:** {article['title']}")
            st.write(f"**Summary:** {article['summary']}")
            st.write(f"**Sentiment:** {article['sentiment']}")
            st.write("-----------------")

        # Comparative Analysis
        sentiment_distribution = compare_sentiments(articles)
        st.write(f"## Comparative Sentiment Score")
        st.write(f"### Sentiment Distribution")
        st.write(f"Positive: {sentiment_distribution['POSITIVE']}")
        st.write(f"Negative: {sentiment_distribution['NEGATIVE']}")
        st.write(f"Neutral: {sentiment_distribution['NEUTRAL']}")


        # THE FOLLOWING ARE YET TO BE DONE
        st.write(f"### Coverage Differences")
        st.write(f"### Topic Overlap")
        st.write("-----------------")

        # Sentiment Analysis Summary
        st.write(f"## Final Sentiment Analysis")

        # text to speech audio
        st.write(f"## Audio")
        summary_text = f"{company_name} के बारे में {len(articles)} खबरें मिलीं। इनमें से {sentiment_distribution['POSITIVE']} खबरें सकारात्मक, {sentiment_distribution['NEGATIVE']} खबरें नकारात्मक, और {sentiment_distribution['NEUTRAL']} खबरें तटस्थ हैं।"
        generate_hindi_tts(summary_text)
        st.audio("output.mp3", format="audio/mp3")
    else:
        st.write("Please enter a company name.")
