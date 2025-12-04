# 📚 RAG PDF Assistant

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-green.svg)](https://github.com/facebookresearch/faiss)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An intelligent PDF Question & Answer system powered by Retrieval Augmented Generation (RAG). Upload multiple PDFs, ask questions, and get accurate answers grounded in your documents with source citations.

## ✨ Features

### 🔐 User Authentication
- **Secure Login/Signup**: Built-in user authentication system
- **Per-User Chat History**: Each user maintains their own conversation history
- **Persistent Sessions**: Chat history saved to disk and restorable

### 📄 Multi-PDF Management
- **Batch Upload**: Upload and ingest multiple PDFs simultaneously
- **Selective Search**: Choose which PDFs to search for each query
- **Source Tracking**: Every chunk is tracked with filename and page number

### 🤖 Intelligent RAG System
- **Vector Search**: FAISS-based semantic similarity search
- **Smart Retrieval**: Balanced chunk selection across multiple documents
- **LLM Integration**: Groq/OpenAI powered answer generation
- **Source Citations**: Answers include references to source documents and pages

### 📊 Advanced Controls
- **Adjustable Retrieval**: Configure number of chunks (k) to retrieve
- **Chat Export**: Download conversation history as TXT file
- **History Management**: Save, reload, or clear chat history

## 🛠️ Tech Stack

- **[Streamlit](https://streamlit.io/)**: Interactive web interface
- **[FAISS](https://github.com/facebookresearch/faiss)**: Facebook's efficient similarity search library
- **[Sentence-Transformers](https://www.sbert.net/)**: Local embedding generation (all-MiniLM-L6-v2)
- **[PyPDF](https://pypdf.readthedocs.io/)**: PDF text extraction
- **[Groq API](https://groq.com/)**: Fast LLM inference (primary)
- **[OpenAI API](https://openai.com/)**: Alternative LLM backend
- **[TQDM](https://tqdm.github.io/)**: Progress bars for ingestion

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- Groq API key (or OpenAI API key)
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/UDAYKIRANHARI/rag-pdf-assistant.git
cd rag-pdf-assistant
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**

Create a `.env` file in the project root:
```env
# Required: Groq API Key (recommended for fast inference)
GROQ_API_KEY=your_groq_api_key_here

# Optional: OpenAI API Key (fallback)
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Model selection (default: llama-3.1-8b-instant)
OPENAI_CHAT_MODEL=llama-3.1-8b-instant

# Optional: Default user credentials
APP_USERNAME=uday
APP_PASSWORD=login123
```

**Get API Keys:**
- Groq: [https://console.groq.com/keys](https://console.groq.com/keys)
- OpenAI: [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)

4. **Run the application**
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📝 Usage Guide

### 1. Login / Sign Up
- Use default credentials (`uday`/`login123`) or create a new account
- Each user has their own isolated chat history

### 2. Upload PDFs
1. Click "Browse files" in the sidebar
2. Select one or more PDF files
3. Click "Ingest all uploaded PDFs" to process them
4. Wait for the ingestion progress bar to complete

### 3. Ask Questions
1. Select which PDFs to search (checkboxes in sidebar)
2. Adjust retrieval settings (k value) if needed
3. Type your question in the input box
4. Click "Send" to get an answer

### 4. View Results
- **Answer**: AI-generated response grounded in your documents
- **Sources Used**: List of PDFs and pages referenced
- **Retrieved Snippets**: Expand to see exact text chunks used

### 5. Manage History
- **Download Chat**: Export conversation as TXT file
- **Save to Disk**: Persist current session
- **Reload from Disk**: Restore previous session
- **Clear History**: Start fresh

## 🗂️ Project Structure

```
rag-pdf-assistant/
├── app.py              # Main Streamlit application
├── ingest.py           # PDF processing and indexing
├── retriever.py        # FAISS retrieval and answer generation
├── requirements.txt    # Python dependencies
├── .env                # Environment variables (create this)
├── data/               # Auto-created for PDFs and indices
│   ├── *.pdf           # Uploaded PDF files
│   ├── faiss_index.*   # Vector index files
│   ├── users.json      # User credentials
│   └── history_*.json  # Per-user chat history
└── README.md           # This file
```

## 🧑‍💻 How It Works

### RAG Pipeline

1. **PDF Ingestion** (`ingest.py`)
   - Extract text from PDFs using PyPDF
   - Split text into manageable chunks (500 chars, 50 char overlap)
   - Generate embeddings using Sentence-Transformers
   - Store in FAISS vector index with metadata

2. **Query Processing** (`retriever.py`)
   - Embed user query using same model
   - Search FAISS index for top-k similar chunks
   - Filter by selected PDFs
   - Balance chunks across multiple sources

3. **Answer Generation**
   - Send retrieved chunks + query to LLM
   - LLM generates answer grounded in context
   - Include source citations (filename + page)
   - Return with snippets for verification

### Architecture

```
┌─────────────┐
│   User Query  │
└─────┬───────┘
     │
     │  Embedding
     │
     v
┌─────────────┐
│ FAISS Index  │ ────> Retrieve Top-K
└─────┬───────┘       Chunks
     │                 │
     │                 v
     │            ┌───────────────┐
     │            │  LLM (Groq/   │
     │            │   OpenAI)     │
     │            └─────┬─────────┘
     │                 │
     v                 v
┌───────────────────────────┐
│  Answer + Citations      │
└───────────────────────────┘
```

## 🔧 Configuration

### Retrieval Settings

- **k (Number of chunks)**: Control how many text chunks to retrieve (1-8)
  - Lower k: Faster, more focused answers
  - Higher k: More context, potentially more comprehensive

### LLM Models

Groq supported models:
- `llama-3.1-8b-instant` (default, fastest)
- `llama-3.1-70b-versatile` (more capable)
- `mixtral-8x7b-32768` (long context)

## 🛡️ Security Notes

⚠️ **Important**: This is a demo application. For production:
- Use proper password hashing (bcrypt, argon2)
- Implement HTTPS
- Add rate limiting
- Use secure session management
- Store API keys in secure vaults

## 📚 Learning Resources

This project demonstrates:
- **RAG (Retrieval Augmented Generation)** architecture
- **Vector databases** with FAISS
- **Semantic search** using embeddings
- **LLM integration** for answer generation
- **Streamlit** for rapid web app development
- **User authentication** and session management

## 🔮 Roadmap

- [ ] Add support for more document formats (DOCX, TXT, etc.)
- [ ] Implement conversation memory (multi-turn chat)
- [ ] Add support for local LLMs (Ollama integration)
- [ ] Improve chunking strategy (semantic splitting)
- [ ] Add citation highlighting in source PDFs
- [ ] Deploy to Streamlit Cloud

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👤 Author

**Uday Kiran Hari**
- GitHub: [@UDAYKIRANHARI](https://github.com/UDAYKIRANHARI)

## 🚀 Deployment

### Deploy to Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Add secrets in "Advanced settings":
   ```toml
   GROQ_API_KEY = "your_key_here"
   APP_USERNAME = "your_username"
   APP_PASSWORD = "your_password"
   ```
5. Deploy!

## 🐛 Troubleshooting

**Issue**: "No index found" error
- **Solution**: Upload and ingest at least one PDF first

**Issue**: "API key not configured"
- **Solution**: Check your `.env` file has `GROQ_API_KEY` or `OPENAI_API_KEY`

**Issue**: Slow retrieval
- **Solution**: Reduce k value or use fewer PDFs

**Issue**: Out of memory
- **Solution**: Process PDFs in smaller batches

## 🌟 Acknowledgments

- FAISS team at Facebook Research
- Sentence-Transformers by UKP Lab
- Streamlit team for the amazing framework
- Groq for fast LLM inference
- The open-source community

---

⭐ If you find this project helpful, please consider giving it a star!
