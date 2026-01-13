# D2W10 – Agentic RAG Streamlit App

This project is a beginner-friendly Agentic RAG (Retrieval-Augmented Generation) application built as part of the D2W10 Daily Challenge.

The application retrieves information from a small local knowledge base, optionally uses a web search tool, and generates grounded answers with sources using a language model.

## Project Overview

The app demonstrates a simple agentic loop:
retrieve relevant documents → optionally use a web search tool → synthesize an answer with sources.

## Project Structure

agentic-rag-streamlit/
- app.py (Streamlit user interface)
- rag_agent.py (RAG pipeline logic)
- agentic_rag.ipynb (Notebook to test the RAG pipeline)
- requirements.txt (Python dependencies)
- .env.example (Environment variables template)
- data/kb.txt (Mini knowledge base)

## Setup

1. Install dependencies  
py -m pip install -r requirements.txt

2. Create environment file  
copy .env.example .env

Add your API keys to the .env file:
- GROQ_API_KEY (required)
- TAVILY_API_KEY (optional)

The .env file is not committed to GitHub.

3. Run the application  
py -m streamlit run app.py

Open in your browser at:  
http://localhost:8501

## Usage

Example questions:
- What is RAG?
- What is Agentic RAG?

The application returns an answer and the sources used.

## Notebook

The file agentic_rag.ipynb runs the same RAG pipeline used by the Streamlit app and is included for reproducibility.

## Notes

This project focuses on a simple and beginner-friendly implementation of Agentic RAG with source attribution.

## Author

Bootcamp project – D2W10 Daily Challenge

