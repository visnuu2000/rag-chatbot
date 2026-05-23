# 📄 PDF Q&A Chatbot

A RAG (Retrieval-Augmented Generation) powered chatbot that lets you upload any PDF and ask questions about it in natural language. Built with Streamlit, LangChain, FAISS, HuggingFace Embeddings, and Groq's ultra-fast LLaMA 3.3 70B model.

---

## 🚀 Demo

> Upload a PDF → Ask questions → Get instant, context-aware answers.

![PDF Q&A Chatbot](screenshot.png)

---

## ✨ Features

- 📤 Upload any PDF document
- 🔍 Semantic search using FAISS vector store
- 🤖 Powered by **LLaMA 3.3 70B** via Groq API (blazing fast inference)
- 🧠 HuggingFace `all-MiniLM-L6-v2` embeddings (runs locally, no API key needed)
- 💬 Chat-style interface with message history
- 🔗 LangChain RAG pipeline for accurate, context-grounded answers

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit |
| LLM | LLaMA 3.3 70B (via Groq) |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector Store | FAISS |
| PDF Parsing | PyPDF2 |
| RAG Framework | LangChain |

---

## 📁 Project Structure

```
rag-chatbot(HF)/
│
├── app.py              # Streamlit frontend & chat UI
├── rag_engine.py       # Core RAG pipeline (PDF parsing, chunking, retrieval, LLM)
├── requirements.txt    # Python dependencies
├── .env                # API keys (not committed)
└── .gitignore
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/rag-chatbot-HF.git
cd rag-chatbot-HF
```

### 2. Create a virtual environment
```bash
python -m venv venv

# Activate
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
```

> Get your free Groq API key at [console.groq.com](https://console.groq.com)

### 5. Run the app
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 🧠 How It Works

```
PDF Upload
    ↓
Extract Text (PyPDF2)
    ↓
Split into Chunks (RecursiveCharacterTextSplitter)
    ↓
Generate Embeddings (HuggingFace all-MiniLM-L6-v2)
    ↓
Store in FAISS Vector DB
    ↓
User asks a question
    ↓
Retrieve top-5 relevant chunks (semantic search)
    ↓
Send context + question to LLaMA 3.3 70B (Groq)
    ↓
Display answer in chat
```

---

## 📦 Requirements

```
streamlit
langchain
langchain-community
langchain-groq
langchain-core
langchain-text-splitters
faiss-cpu
PyPDF2
sentence-transformers
python-dotenv
```

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
