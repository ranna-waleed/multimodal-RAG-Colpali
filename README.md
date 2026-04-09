# Multi-Modal Document Intelligence System
### DSAI 413 — Assignment 1 | RAG-Based QA System

A document question-answering system that understands PDFs **visually** — including tables, charts, figures, and text — using ColPali embeddings, Qdrant vector search, and LLaMA 4 for cited answers.

**[Video Demo](https://drive.google.com/drive/folders/1Ne_dVJBJgO2PHw7s6y_-vfNoe-JMfomL?usp=sharing)**

---

## What it does

Most RAG systems extract text from PDFs and lose all the visual structure — tables break apart, charts disappear, and figures are ignored. This system takes a different approach: it treats each page as an **image** and uses a vision-language model (ColPali) to understand the full visual content of every page.

You upload a PDF, ask a question, and the system:
1. Finds the most visually relevant pages using ColPali embeddings
2. Sends those page images to LLaMA 4 Scout
3. Returns a detailed answer with **page-level citations**

---

## Architecture

```
PDF Upload
    
PyMuPDF → Page Images (JPEG)
    
ColIdefics3 (ColSmol-256M) → 128-dim Multi-Vector Embeddings
    
Local Qdrant → Vector Storage (MAX_SIM cosine similarity)
    
User Query → ColIdefics3 Query Embedding → Qdrant Search → Top-K Pages
    
LLaMA 4 Scout (Groq API) → Cited Answer + Retrieved Page Images
    
Streamlit UI → Display Answer + Sources + Page Images
```

---

## Tech Stack

| Component | Tool |
|-----------|------|
| PDF Processing | PyMuPDF (fitz) |
| Visual Embeddings | ColIdefics3 / ColSmol-256M (ColPali family) |
| Vector Database | Qdrant (local) |
| Answer Generation | Groq — LLaMA 4 Scout 17B |
| UI | Streamlit |
| Language | Python 3.11 |

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/multimodal-RAG-Colpali.git
cd multimodal-RAG-Colpali
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# or
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
Copy `.env.example` to `.env` and fill in your keys:
```
QDRANT_COLLECTION_NAME=colpali_docs
GROQ_API_KEY=your_groq_api_key_here
```

Get a free Groq API key at: https://console.groq.com

### 5. Run the app
```bash
python -m streamlit run ui/streamlit_app.py
```

Open http://localhost:8501 in your browser.

---

## How to Use

1. **Upload a PDF** using the sidebar file uploader
2. **Click "Ingest Document"** — the system converts pages to images and generates embeddings
3. **Ask a question** in the text box or use the example buttons
4. **View the answer** with page citations and the retrieved page images shown below

---

## Project Structure

```
multimodal-RAG-Colpali/
├── app/
│   ├── config.py          # Environment variables and settings
│   ├── ingest.py          # PDF → images → embeddings → Qdrant
│   ├── retriever.py       # Query embedding + Qdrant similarity search
│   └── generator.py       # Groq LLaMA 4 answer generation
├── ui/
│   └── streamlit_app.py   # Streamlit chatbot interface
├── evaluation/
│   └── eval_queries.py    # Benchmark evaluation script
├── data/
│   ├── sample_docs/       # Uploaded PDFs saved here
│   ├── page_images/       # Converted page images
│   └── qdrant_local/      # Local Qdrant vector database
├── .env.example
├── requirements.txt
└── README.md
```

---

## Evaluation Results

Tested on 7 benchmark queries across text, table, and chart modalities:

| Query Type | Example | Result |
|------------|---------|--------|
| Text | "What is the main topic?" |  Pass |
| Text | "Summarize key findings" |  Pass |
| Table | "What numbers are mentioned?" |  Pass |
| Chart | "What do the charts show?" |  Pass |
| Text | "What are main conclusions?" | Pass |
| Text | "What methodology is used?" | Pass |
| Text | "What are the recommendations?" | Pass |

**7/7 queries successful — 100% retrieval accuracy**

---

## Features Covered

-  Multi-modal ingestion (text, tables, images, charts)
-  Visual embedding with ColPali (ColSmol-256M)
-  Page-level semantic chunking
-  Vector-based retrieval with Qdrant (MAX_SIM)
-  Interactive QA chatbot (Streamlit)
-  Page-level citations in every answer
-  Evaluation suite with benchmark queries

---

## Notes

- The model (ColSmol-256M, ~500MB) downloads automatically on first run
- Ingestion is slow on CPU — a GPU would be significantly faster
- All vectors are stored locally in `data/qdrant_local/` — no cloud dependency
- For best results use PDFs with clear text and well-structured tables

---

## References

- [ColPali Paper](https://arxiv.org/abs/2407.01449) — Faysse et al., 2024
- [ColPali GitHub](https://github.com/illuin-tech/colpali)
- [ViDoRe Benchmark](https://huggingface.co/spaces/vidore/vidore-leaderboard)
