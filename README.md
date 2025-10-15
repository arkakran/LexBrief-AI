# Legal Document Analyzer
> AI-powered tool for extracting and analyzing legal arguments from PDF documents using RAG (Retrieval-Augmented Generation)

## Overview

Legal Document Analyzer automates the extraction of key legal arguments from PDF briefs, identifying statutory violations, regulatory failures, constitutional claims, and case law references. Built with Flask and powered by Groq LLaMA 3.3 70B, it uses a 10-stage RAG pipeline to ensure comprehensive analysis.

## Features

- **Multi-Category Extraction**: Automatically categorizes arguments (statutory, regulatory, constitutional, case law, procedural)
- **Importance Scoring**: Assigns CRITICAL/HIGH/MEDIUM/LOW ratings with numerical scores
- **Quote Attribution**: Links arguments to exact page numbers with supporting quotes
- **Diversity Ranking**: Ensures balanced representation across all legal categories
- **Modern UI**: Responsive interface with export options (print, JSON)

## Technology Stack

- **Backend**: Flask, LangChain, Pydantic
- **AI Models**: Groq LLaMA 3.3 70B, sentence-transformers/all-mpnet-base-v2
- **Vector DB**: Qdrant Cloud
- **Document Processing**: PDFPlumber, RecursiveCharacterTextSplitter
- **Utilities**: RapidFuzz (quote matching), StructLog

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/legal-document-analyzer.git
cd legal-document-analyzer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env with your API keys

# Run application
python app.py
```

Visit `http://localhost:5000`

## Configuration

Create `.env` file with the following:

```bash
# Required API Keys
GROQ_API_KEY=your_groq_api_key
QDRANT_URL=https://your-cluster.qdrant.io:6333
QDRANT_API_KEY=your_qdrant_api_key
HUGGINGFACE_API_KEY=your_huggingface_api_key

# Model Settings
LLM_MODEL=llama-3.3-70b-versatile
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2

# Processing Parameters
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RETRIEVAL=60
TOP_K_RERANKED=25
FINAL_OUTPUT_COUNT=10
```

### Getting API Keys

- **Groq**: [console.groq.com](https://console.groq.com/)
- **Qdrant**: [cloud.qdrant.io](https://cloud.qdrant.io/)
- **HuggingFace**: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

## Usage

1. Open application at `http://localhost:5000`
2. Upload PDF document (max 50MB)
3. Optionally customize analysis query
4. Click "Analyze Document"
5. Review extracted arguments with importance scores, quotes, and page numbers
6. Export as JSON or print report

## Project Structure

```
legal-document-analyzer/
├── app.py                      # Main Flask application
├── config.py                   # Configuration management
├── requirements.txt            # Dependencies
├── .env                        # Environment variables
├── utils/
│   ├── document_processor.py  # PDF processing & chunking
│   ├── vector_operations.py   # Embeddings & Qdrant
│   ├── llm_services.py        # LLM analysis
│   └── post_processor.py      # Quote mapping & ranking
├── models/
│   └── schemas.py             # Pydantic models
├── templates/
│   ├── index.html             # Upload page
│   └── results.html           # Results display
└── static/
    └── style.css              # Styling
```

## Pipeline Workflow

1. **Document Processing**: Load PDF, split into chunks, extract metadata
2. **Embedding Generation**: Create vector embeddings using sentence-transformers
3. **Vector Storage**: Store in document-specific Qdrant collection
4. **Multi-Query Retrieval**: Generate 4 query variations targeting different argument types
5. **Reranking**: Sort and select top 25 chunks by relevance
6. **LLM Analysis**: Extract 12-15 diverse arguments with Groq LLaMA 3.3 70B
7. **Quote Mapping**: Match quotes to page numbers using fuzzy matching
8. **LLM Refinement**: Add precise statutory citations
9. **Diversity Ranking**: Deduplicate and ensure category balance
10. **Output**: Return top 10 arguments with full metadata

## API Endpoints

- `GET /` - Upload page
- `POST /analyze` - Analyze document (params: `file`, `query`)
- `GET /health` - Health check

## Deployment

### Render

```yaml
# render.yaml
services:
  - type: web
    name: legal-document-analyzer
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn app:app
```



## Performance

- Average processing time: 30-60 seconds
- Max file size: 50MB
- Tested up to 200 pages
- Embedding dimension: 768
- LLM context: 32K tokens



---

Built with Flask, LangChain, and Groq for legal professionals
