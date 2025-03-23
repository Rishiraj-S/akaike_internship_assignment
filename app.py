# Importing necessary libraries
import streamlit as st
from utils import *

# Application Title
st.title("News Summarisation and Text-To-Speech Application")

# Input box for company name
company_name = st.text_input("## Enter the name of the company: ")

# Button to generate news
if st.button("Search"):
    if company_name:
        # Scrape news articles related to the company
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
            st.write(f"**Topics:** {article['topics']}")
            st.write(f"**Sentiment:** {article['sentiment']}")
            st.write("-----------------")

        # Comparative Analysis of Sentiments
        sentiment_distribution = compare_sentiments(articles)
        st.write(f"## Comparative Sentiment Score")
        st.write(f"### Sentiment Distribution")
        st.write(f"Positive: {sentiment_distribution['POSITIVE']}")
        st.write(f"Negative: {sentiment_distribution['NEGATIVE']}")
        st.write(f"Neutral: {sentiment_distribution['NEUTRAL']}")

        # Coverage Differences
        comparison = extract_impact_and_comparisons(comparisons_and_impact(articles))
        st.write(f'### Coverage Differences')
        for item in comparison:
            st.write(f'{item}')

        # Topic Overlap
        st.write(f"### Topic Overlap")
        st.write(f"Common Topics: {extract_common_topics(articles)}")
        unique_topics = extract_unique_topics(articles)
        for i, topics in enumerate(unique_topics):
            st.write(f"Unique Topics in Article {i + 1}: {topics}")
        st.write("-----------------")

        # Final Sentiment Analysis Summary
        st.write(f"## Final Sentiment Analysis")
        final_text = get_detailed_sentiment_summary(articles)
        st.write(f'{final_text}')

        # Text-to-Speech Audio
        st.write(f"## Audio")
        generate_hindi_tts(translate_en_to_hi(final_text))
        st.audio("output.mp3", format="audio/mp3")
    else:
        st.write("Please enter a company name.")