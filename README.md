title: Akaike Internship Assignment
emoji: 🐢
colorFrom: pink
colorTo: purple
sdk: streamlit
sdk_version: 1.43.2
app_file: app.py
pinned: false
license: mit
short_description: akaike internship assignment

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
   - Create a ```.env``` file in the root directory
   - Add the follwing API keys (replace with your actual keys):
     ```plaintext
     CLASSIFICATION_API_KEY=your_classification_api_key
     SUMMARIZATION_API_KEY=your_summarization_api_key
     FINAL_SUM_KEY=your_final_sum_key
     TRANSLATION_KEY=your_translation_key
     
### Running the Application
1. Streamlit Application
   - Run the Streamlit App:
     ```bash
     streamlit run app.py
   - Open your browser and navigate to ```http://localhost:8501```

## Model Details

### Sentiment Analysis
- **Model**: `facebook/bart-large-mnli` (via Hugging Face Transformers)
- **Purpose**: Classifies the sentiment of news articles as Positive, Negative, or Neutral.
- **Integration**: Used in `utils.py` for sentiment analysis.

### Summarization
- **Model**: `mistralai/Mistral-7B-Instruct-v0.1` (via Hugging Face Inference API)
- **Purpose**: Generates summaries and comparative analysis of news articles.
- **Integration**: Used in `utils.py` for summarization and impact analysis.

### Text-to-Speech (TTS)
- **Model**: `gTTS` (Google Text-to-Speech)
- **Purpose**: Converts summarized content into Hindi speech.
- **Integration**: Used in `utils.py` for generating Hindi audio files.

### Translation
- **Model**: `Helsinki-NLP/opus-mt-en-hi` (via Hugging Face Inference API)
- **Purpose**: Translates English text to Hindi for TTS.
- **Integration**: Used in `utils.py` for translation.

## API Development

### FastAPI Backend
- **Purpose**: Handles communication between the frontend (Streamlit) and backend (news scraping, sentiment analysis, etc.).
- **Endpoints**:
  - `POST /analyze`: Accepts a company name and returns analyzed news data.
    - **Input**: JSON object with `name` field (e.g., `{"name": "Tesla"}`).
    - **Output**: JSON object containing articles, sentiment distribution, common topics, and unique topics.

### Accessing the API
1. Start the FastAPI server:
   ```bash
   uvicorn api:app --reload
## API Usage

### Third-Party APIs
1. **Hugging Face Inference API**:
   - **Purpose**: Used for zero-shot classification, summarization, and translation.
   - **Integration**: Accessed via API keys stored in `.env`.

2. **Bing News RSS Feed**:
   - **Purpose**: Used for scraping news articles related to the given company.
   - **Integration**: Accessed via `requests` and `BeautifulSoup` in `utils.py`.

---

## Assumptions & Limitations

### Assumptions
1. **News Source**: The application assumes that Bing News RSS feed provides sufficient and relevant news articles for the given company.
2. **Language**: The application assumes that news articles are in English for sentiment analysis and summarization.
3. **API Keys**: The application assumes that valid Hugging Face API keys are provided in the `.env` file.

### Limitations
1. **News Availability**: The application may not fetch news if the company name is too generic or if no relevant articles are available.
2. **Sentiment Accuracy**: Sentiment analysis may not always be accurate due to the complexity of natural language.
3. **Translation Quality**: The quality of Hindi translation and TTS depends on the underlying models and may not always be perfect.
4. **Scalability**: The application is designed for small-scale use and may not handle large volumes of data efficiently.

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
