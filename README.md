---
title: PDF QA Chatbot
emoji: 📄
colorFrom: purple
colorTo: blue
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
---

# PDF Q&A Chatbot (RAG-based)

Upload any PDF and ask questions about it.
Powered by LangChain, FAISS, HuggingFace embeddings
and Google Gemini 1.5 Flash.

## Tech stack
- LangChain — RAG pipeline
- FAISS — vector similarity search
- HuggingFace sentence-transformers — free embeddings
- Google Gemini 1.5 Flash — free LLM
- Streamlit — web interface

## How to run locally
pip install -r requirements.txt
streamlit run app.py

## Live demo
[View on Hugging Face Spaces](PASTE_YOUR_LINK_HERE)