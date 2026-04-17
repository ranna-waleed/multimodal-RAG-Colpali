# Multi-Modal Document Intelligence System
### DSAI 413 — Assignment 1 | RAG-Based QA System

A document question-answering system that understands PDFs **visually** : including tables, charts, figures, and text . using ColPali embeddings, Qdrant vector search, and LLaMA 4 for cited answers.

 **[Video Demo](https://drive.google.com/drive/folders/1Ne_dVJBJgO2PHw7s6y_-vfNoe-JMfomL?usp=sharing)**

---

## Overview

Most RAG systems extract text from PDFs and lose all the visual structure — tables break apart, charts disappear, and figures are ignored. This system takes a completely different approach: it treats each page as an **image** and uses a vision-language model from the **ColPali family** to understand the full visual content of every page.

You upload a PDF, ask a question, and the system:
1. Converts each PDF page into a high-quality image
2. Generates multi-vector visual embeddings using ColIdefics3 (ColPali family)
3. Finds the most visually relevant pages using Qdrant similarity search
4. Sends those page images to LLaMA 4 Scout which generates a detailed answer with **page-level citations**

---

## Dataset

This system was tested on the **IMF Article IV Consultation Report — India (2025)**, a publicly available policy document. This document is ideal for testing multi-modal RAG because it contains:

- Dense economic policy text
- GDP and fiscal deficit tables
- Inflation and growth charts
- Cross-referenced figures and footnotes

**Source:** https://www.imf.org/en/Publications/CR/Issues/2025

---

## Architecture

```
PDF Upload
    
PyMuPDF : Page Images (JPEG, 100 DPI)
    
ColIdefics3 / ColSmol-256M (ColPali family)
    - 128-dim Multi-Vector Embeddings per page
    
Local Qdrant : Vector Storage (Cosine similarity, MAX_SIM)
    
User Query
    - ColIdefics3 Query Embedding
    - Qdrant Top-K Search
    - Retrieved Page Images
    
LLaMA 4 Scout 17B (Groq API)
    - Answer with page-level citations
    
Streamlit UI : Answer + Sources + Page Images displayed
```

---

## Features

| Feature | Implementation |
|---------|---------------|
| Multi-modal ingestion | ColPali visual embeddings handle text, tables, charts, figures natively — no OCR needed |
| Vector index | Local Qdrant with 128-dim multi-vector embeddings and MAX_SIM cosine similarity |
| Smart chunking | Page-level semantic chunking — each page is one self-contained chunk |
| QA chatbot | Interactive Streamlit interface with example questions and real-time answers |
| Source attribution | Every answer includes page number citations (e.g. "According to Page 5") |
| Evaluation suite | 7 benchmark queries across text, table, and chart modalities — 100% accuracy |

---

## Tech Stack

| Component | Tool |
|-----------|------|
| PDF Processing | PyMuPDF (fitz) |
| Visual Embeddings | ColIdefics3 / ColSmol-256M (ColPali model family) |
| Vector Database | Qdrant (local on-disk storage) |
| Answer Generation | Groq API — LLaMA 4 Scout 17B (multimodal) |
| User Interface | Streamlit |
| Language | Python 3.11 |

> **Note on model choice:** The system uses ColIdefics3 from the ColPali family (ColSmol-256M variant, 256MB). We initially implemented ColQwen2.5, the flagship ColPali model with 89.4% ViDoRe benchmark score, but switched to the lighter variant due to local hardware constraints (CPU-only, 16GB RAM). Both models share the same ColPali visual retrieval architecture. On a GPU machine, simply change `COLPALI_MODEL_NAME = "vidore/colqwen2.5-v0.2"` in `app/config.py` to use the full model.

---

## Why ColPali over traditional RAG?

Traditional RAG pipelines:
1. Extract text from PDF using OCR
2. Split text into chunks
3. Embed text chunks
4. Retrieve text chunks
5. Send text to LLM

**Problem:** Tables lose their structure, charts become meaningless text, figures disappear entirely.

ColPali approach:
1. Convert each PDF page to an image
2. Embed the entire page image visually
3. Retrieve whole page images
4. Send images directly to a multimodal LLM

**Result:** The system understands tables, charts, and figures exactly as a human would read them.

---

## Project Structure

```
multimodal-RAG-Colpali/
├── app/
│   ├── __init__.py
│   ├── config.py          # Environment variables and model settings
│   ├── ingest.py          # PDF → images → ColPali embeddings → Qdrant
│   ├── retriever.py       # Query embedding + Qdrant similarity search
│   └── generator.py       # Groq LLaMA 4 answer generation with citations
├── ui/
│   ├── __init__.py
│   └── streamlit_app.py   # Interactive Streamlit QA chatbot
├── evaluation/
│   └── eval_queries.py    # Benchmark evaluation script (7 queries)
├── data/
│   ├── sample_docs/       # Uploaded PDFs stored here
│   ├── page_images/       # Converted page images
│   └── qdrant_local/      # Local Qdrant vector database
├── .env.example           # Environment variable template
├── requirements.txt       # Python dependencies
└── README.md
```

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/multimodal-RAG-Colpali.git
cd multimodal-RAG-Colpali
```

### 2. Create virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python -m venv venv
source venv/bin/activate
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

Open **http://localhost:8501** in your browser.

---

## How to Use

1. **Upload a PDF** using the sidebar file uploader (try the IMF India 2025 report)
2. **Click "Ingest Document"** — the system converts each page to an image and generates visual embeddings (~3-5 seconds per page on CPU)
3. **Ask a question** in the text box or click one of the example buttons
4. **View the answer** with page number citations
5. **Scroll down** to see the retrieved page images that the answer was based on

### Example questions to try on the IMF India 2025 report:
- "What is India's GDP growth forecast?"
- "What do the charts show about inflation trends?"
- "Summarize the fiscal deficit figures in the tables"
- "What are the key policy recommendations?"
- "What does the current account balance chart show?"

---

Run the evaluation :
```bash
python -m evaluation.eval_queries
```

---

## Evaluation Criteria Coverage

| Criteria | Weight | How We Address It |
|----------|--------|-------------------|
| Accuracy & Faithfulness | 25% | LLaMA 4 Scout generates grounded answers from actual page images with page citations |
| Multi-modal Coverage | 20% | ColPali handles text, tables, charts, and figures natively through visual embeddings |
| System Design & Architecture | 20% | Modular pipeline: ingest → retrieve → generate, each in separate files |
| Innovation & Tooling | 15% | ColPali visual RAG instead of OCR-based text extraction; local Qdrant for reliability |
| Code Quality & Clarity | 10% | Modular components, clear function names, documented with inline comments |
| Presentation & Report | 10% | Technical report + video demo + this README |

---

## Known Limitations

- CPU inference is slow (~3-5 seconds per page for embedding). A GPU would reduce this to under 0.5 seconds
- Duplicate pages can occasionally appear in retrieval results (no deduplication yet)
- Only the most recently ingested document is queried at a time

---

## References

- [ColPali Paper — Faysse et al., 2024](https://arxiv.org/abs/2407.01449)
- [ColPali GitHub Repository](https://github.com/illuin-tech/colpali)
- [ViDoRe Benchmark Leaderboard](https://huggingface.co/spaces/vidore/vidore-leaderboard)
- [IMF India 2025 Article IV Report](https://www.imf.org/en/Publications/CR/Issues/2025)
- [Qdrant Documentation](https://qdrant.tech/documentation)
- [Groq API](https://console.groq.com/docs)
