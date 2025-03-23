# Akaike Internship Assignment
## News Summarization and Text-To-Speech Application

\documentclass{article}
\usepackage{geometry}
\geometry{a4paper, margin=1in}
\usepackage{enumitem}
\usepackage{hyperref}
\hypersetup{
    colorlinks=true,
    linkcolor=blue,
    filecolor=magenta,      
    urlcolor=cyan,
}
\urlstyle{same}

\title{News Summarization and Text-to-Speech Application Documentation}
\author{Your Name}
\date{\today}

\begin{document}

\maketitle

\section*{Project Setup}
To set up and run the News Summarization and Text-to-Speech Application, follow these steps:

\begin{enumerate}
    \item \textbf{Clone the Repository:}
    \begin{verbatim}
    git clone <repository-url>
    cd <repository-folder>
    \end{verbatim}

    \item \textbf{Install Dependencies:}
    Ensure you have Python 3.8 or higher installed. Then, install the required libraries using:
    \begin{verbatim}
    pip install -r requirements.txt
    \end{verbatim}

    \item \textbf{Set Up Environment Variables:}
    Create a `.env` file in the root directory and add the following API keys:
    \begin{verbatim}
    CLASSIFICATION_API_KEY=<your-huggingface-api-key>
    SUMMARIZATION_API_KEY=<your-huggingface-api-key>
    TRANSLATION_KEY=<your-huggingface-api-key>
    FINAL_SUM_KEY=<your-huggingface-api-key>
    \end{verbatim}

    \item \textbf{Run the Application:}
    \begin{itemize}
        \item To run the Streamlit web application:
        \begin{verbatim}
        streamlit run app.py
        \end{verbatim}
        \item To run the FastAPI backend:
        \begin{verbatim}
        uvicorn api:app --host 0.0.0.0 --port 8000
        \end{verbatim}
    \end{itemize}
\end{enumerate}

\section*{Model Details}
The application uses the following models for various tasks:

\begin{itemize}
    \item \textbf{Summarization:}
    The Hugging Face model \texttt{mistralai/Mistral-7B-Instruct-v0.1} is used for generating summaries and performing comparative analysis of news articles.

    \item \textbf{Sentiment Analysis:}
    The Hugging Face model \texttt{facebook/bart-large-mnli} is used for zero-shot classification to determine the sentiment (positive, negative, or neutral) of news articles.

    \item \textbf{Text-to-Speech (TTS):}
    The \texttt{gTTS} (Google Text-to-Speech) library is used to convert summarized content into Hindi speech. The translation from English to Hindi is performed using the Hugging Face model \texttt{Helsinki-NLP/opus-mt-en-hi}.
\end{itemize}

\section*{API Development}
The application uses FastAPI to create a backend API for communication between the frontend and backend. The API is defined in \texttt{api.py} and provides the following endpoint:

\begin{itemize}
    \item \textbf{Endpoint:} \texttt{POST /analyze}
    \begin{itemize}
        \item \textbf{Input:} A JSON object containing the company name.
        \begin{verbatim}
        {
            "name": "Tesla"
        }
        \end{verbatim}
        \item \textbf{Output:} A structured JSON response containing:
        \begin{itemize}
            \item Company name
            \item List of articles with titles, summaries, sentiments, and topics
            \item Sentiment distribution across articles
            \item Common topics across articles
            \item Unique topics for each article
        \end{itemize}
    \end{itemize}
\end{itemize}

\section*{API Usage}
The application uses the following third-party APIs:

\begin{itemize}
    \item \textbf{Hugging Face Inference API:}
    \begin{itemize}
        \item \textbf{Purpose:} Used for zero-shot classification, summarization, and translation tasks.
        \item \textbf{Integration:} API keys are required and must be stored in the `.env` file. The application interacts with the Hugging Face API using the `requests` library.
    \end{itemize}

    \item \textbf{Accessing the API via Postman:}
    \begin{itemize}
        \item Set the request type to \texttt{POST}.
        \item Use the URL: \texttt{http://localhost:8000/analyze}.
        \item Add the input JSON in the request body.
        \item Send the request to receive the structured output.
    \end{itemize}
\end{itemize}

\section*{Assumptions \& Limitations}
\begin{itemize}
    \item \textbf{Assumptions:}
    \begin{itemize}
        \item The Bing News RSS feed provides at least 10 articles for the given company name.
        \item The Hugging Face API keys are valid and have sufficient quota for inference tasks.
        \item The user has a stable internet connection to access external APIs.
    \end{itemize}

    \item \textbf{Limitations:}
    \begin{itemize}
        \item The application relies on external APIs, which may introduce latency or downtime.
        \item The sentiment analysis model may not always accurately classify the sentiment of complex or ambiguous text.
        \item The translation and TTS quality depend on the performance of the Hugging Face and gTTS models.
        \item The application is designed for single-company analysis and may not handle multiple companies simultaneously.
    \end{itemize}
\end{itemize}

\end{document}
