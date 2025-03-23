# News Summarization and Text-to-Speech Application

## Overview
This project is a web-based application that extracts key details from multiple news articles related to a given company, performs sentiment analysis, conducts a comparative analysis, and generates a text-to-speech (TTS) output in Hindi. The tool allows users to input a company name and receive a structured sentiment report along with an audio output.

## Features
- **News Extraction**: Fetches and displays the title, summary, and topics from at least 10 unique news articles related to the given company.
- **Sentiment Analysis**: Performs sentiment analysis on the article content (positive, negative, neutral).
- **Comparative Analysis**: Conducts a comparative sentiment analysis across the articles to derive insights on how the company's news coverage varies.
- **Text-to-Speech**: Converts the summarized content into Hindi speech using an open-source TTS model.
- **User Interface**: Provides a simple web-based interface using Streamlit.
- **API Development**: Communication between the frontend and backend happens via APIs.
- **Deployment**: Deployed on Hugging Face Spaces for testing.

---

## Project Setup

### Prerequisites
- Python 3.8 or higher
- Git (optional, for cloning the repository)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
3. Set up environment variables:
   - Create a .env file in the root directory
   - Add the follwing API keys (replace with your actual keys):
     ```CLASSIFICATION_API_KEY=your_classification_api_key
SUMMARIZATION_API_KEY=your_summarization_api_key
FINAL_SUM_KEY=your_final_sum_key
TRANSLATION_KEY=your_translation_key```

