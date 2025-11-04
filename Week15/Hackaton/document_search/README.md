# AI-Powered Document Search and Summarization System

An intelligent document ingestion and retrieval system that uses sentence transformers for semantic search and BART for automatic summarization. Upload PDF, Word, or text documents, then search and summarize content using natural language queries.

## Features

- **Multi-format document ingestion**: PDF, DOCX, DOC, TXT, Markdown
- **Semantic search**: Uses sentence-transformers (all-MiniLM-L6-v2) for meaning-based search
- **Vector similarity**: FAISS IndexFlatL2 for CPU-efficient similarity search
- **Automatic summarization**: BART-base model generates concise summaries of search results
- **Interactive web interface**: Streamlit-based UI with document upload and real-time search
- **Evaluation metrics**: Precision@k, Recall@k, and ROUGE-L for quality assessment
- **CPU-optimized**: Batch processing with configurable sizes for CPU-only environments

## Project Structure

```
document_search/
├── config.py               # Configuration constants and paths
├── logging_utils.py        # Logging configuration
├── document_io.py          # Document ingestion and text extraction
├── text_processing.py      # Text cleaning and chunking
├── embedding_engine.py     # Sentence-transformers embedding generation
├── vector_store.py         # FAISS vector store management
├── summarization.py        # BART-based summarization
├── evaluation.py           # Precision@k, Recall@k, ROUGE metrics
├── pipeline.py             # Orchestration logic
├── app.py                  # Streamlit web interface
├── requirements.txt        # Python dependencies
├── uploads/                # Place documents here for batch processing
└── outputs/                # All generated artifacts
    ├── raw_texts/          # Extracted text from documents
    ├── chunks/             # Chunked text with metadata (JSON)
    ├── embeddings/         # NumPy embeddings and metadata
    ├── vector_store/       # FAISS index and metadata
    ├── summaries/          # Generated summaries (JSON)
    ├── reports/            # Evaluation results
    └── logs/               # Pipeline execution logs
```

## Installation

### 1. Install Dependencies

```bash
cd Week15/Hackaton/document_search
pip install -r requirements.txt
```

**Core Dependencies:**

- `sentence-transformers` - Embedding generation
- `faiss-cpu` - Vector similarity search
- `transformers` - BART summarization
- `torch` - PyTorch backend
- `PyPDF2` - PDF text extraction
- `python-docx` - Word document extraction
- `streamlit` - Web interface
- `rouge-score` - Summarization evaluation
- `tqdm` - Progress bars

### 2. Download Models (First Run)

Models are automatically downloaded on first use:

- `all-MiniLM-L6-v2` (~80MB) - Embedding model
- `facebook/bart-base` (~560MB) - Summarization model

## Usage

### Option 1: Streamlit Web Interface (Recommended)

**Start the web app:**

```bash
streamlit run app.py
```

**Features:**

1. **Upload documents** via sidebar file uploader
2. **Process documents** to extract text and build index
3. **Search** using natural language queries
4. **View results** with similarity scores and metadata
5. **Summarize** top results with one click

**Access:** Opens automatically in your browser at `http://localhost:8501`

### Option 2: Batch Processing Pipeline

**Prepare documents:**

```bash
# Place your documents in the uploads/ folder
mkdir -p uploads
cp /path/to/your/documents/*.pdf uploads/
```

**Run the pipeline:**

```python
from pipeline import run_pipeline

# Process all documents in uploads/
results = run_pipeline(
    documents_dir=None,        # Uses uploads/ by default
    run_evaluation=True,       # Generate evaluation metrics
    rebuild_index=True         # Rebuild from scratch
)

print(f"Processed: {results['num_documents']} documents")
print(f"Generated: {results['num_chunks']} chunks")
```

### Option 3: Interactive Search (Python)

```python
from embedding_engine import EmbeddingEngine
from vector_store import load_vector_store
from summarization import Summarizer

# Load components
engine = EmbeddingEngine()
engine.load_model()
store = load_vector_store()
summarizer = Summarizer()

# Search
query = "How does machine learning work?"
query_embedding = engine.embed_texts([query])
results = store.search_similar(query_embedding[0], top_k=5)

# Display results
for idx, result in enumerate(results, 1):
    print(f"{idx}. {result['doc_id']} - {result['text'][:100]}...")
    print(f"   Similarity: {result['similarity_score']:.2%}\n")

# Summarize
summary = summarizer.summarize_chunks(results)
print(f"Summary: {summary['summary']}")
```

## Configuration

Edit `config.py` to customize behavior:

```python
# Processing parameters
CHUNK_SIZE = 400              # Tokens per chunk
CHUNK_OVERLAP = 50            # Token overlap between chunks
BATCH_SIZE = 4                # CPU-optimized batch size
TOP_K = 5                     # Default search results
MAX_DOCS = 50                 # Document limit for CPU constraints

# Model configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SUMMARIZATION_MODEL = "facebook/bart-base"

# Summarization
SUMMARY_MIN_LENGTH = 40
SUMMARY_MAX_LENGTH = 150
SUMMARY_INPUT_MAX_TOKENS = 1024
```

## Performance Optimization

**CPU Constraints:**

- Small batch sizes (4) for embedding/summarization
- IndexFlatL2 (no IVF clustering) for FAISS
- Document limit (`MAX_DOCS = 50`)
- Chunk size limit for summarization

**Memory Management:**

- Embeddings cached to disk (NumPy arrays)
- FAISS index persisted between sessions
- Streamlit caching for models

**Speedup Tips:**

- Process documents in batches
- Use smaller chunk sizes for faster indexing
- Reduce `TOP_K` for faster searches
- Skip evaluation if not needed

## Evaluation Metrics

The pipeline generates synthetic test queries and computes:

**Search Quality:**

- **Precision@k**: Fraction of retrieved chunks that are relevant
- **Recall@k**: Fraction of relevant chunks that are retrieved
- Computed for k = 1, 3, 5, 10

**Summarization Quality:**

- **ROUGE-L**: Longest common subsequence overlap with reference
- Precision, Recall, F1 variants

**Results saved to:** `outputs/reports/evaluation.json`

## Output Artifacts

All pipeline outputs are organized in `outputs/`:

| Directory       | Contents                                            |
| --------------- | --------------------------------------------------- |
| `raw_texts/`    | Extracted text from documents (`{doc_id}.txt`)      |
| `chunks/`       | Chunked text with metadata (`{doc_id}_chunks.json`) |
| `embeddings/`   | NumPy embeddings (`embeddings.npy`) and metadata    |
| `vector_store/` | FAISS index (`index.faiss`) and chunk mapping       |
| `summaries/`    | Generated summaries with query context              |
| `reports/`      | Evaluation metrics and analysis                     |
| `logs/`         | Pipeline execution logs (`pipeline.log`)            |

## Troubleshooting

**Issue:** "No module named 'sentence_transformers'"

- **Fix:** `pip install sentence-transformers`

**Issue:** "FAISS index not found"

- **Fix:** Process documents first using Streamlit or `run_pipeline()`

**Issue:** "Out of memory during embedding"

- **Fix:** Reduce `BATCH_SIZE` in `config.py` or limit documents with `MAX_DOCS`

**Issue:** "Summarization too slow"

- **Fix:** Reduce `SUMMARY_MAX_LENGTH` or use smaller model (t5-small)

**Issue:** PDF extraction fails

- **Fix:** Ensure PDF is text-based (not scanned images). Try OCR preprocessing if needed.

## Extending the System

**Add new document formats:**

```python
# In document_io.py
def extract_text_from_html(html_path: Path) -> str:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html_path.read_text(), 'html.parser')
    return soup.get_text()
```

**Use different embedding model:**

```python
# In config.py
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"  # Higher quality
EMBEDDING_DIM = 768  # Update dimension
```

**Add query expansion:**

```python
# Generate related queries before search
from transformers import pipeline
query_expander = pipeline("text-generation", model="gpt2")
expanded_queries = query_expander(query, max_length=50, num_return_sequences=3)
```

## Architecture Patterns

This project follows DI-Bootcamp Week 10+ conventions:

- **Modular pipeline**: Separate modules for each stage (IO, processing, embedding, search)
- **Config-driven**: Constants in `config.py` (paths, model names, hyperparameters)
- **Structured logging**: Console + file logging with UTF-8 encoding
- **Graceful degradation**: Optional dependencies with `HAS_*` flags
- **Output organization**: Categorized subdirectories in `outputs/`
- **Type hints**: Python 3.10+ type annotations throughout
- **Docstrings**: Google-style documentation for all functions
- **Error handling**: Try/except with descriptive messages

## Example Workflow

```bash
# 1. Start with batch processing
python -c "from pipeline import run_pipeline; run_pipeline()"

# 2. Check outputs
ls outputs/raw_texts/        # Extracted text
ls outputs/vector_store/     # FAISS index

# 3. Launch web interface
streamlit run app.py

# 4. Upload more documents via UI
# 5. Search and summarize interactively
```

## Citations

**Models:**

- Sentence-Transformers: [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- BART: [facebook/bart-base](https://huggingface.co/facebook/bart-base)

**Libraries:**

- FAISS: Facebook AI Similarity Search
- Hugging Face Transformers
- Streamlit

## License

This is an educational project for DI-Bootcamp Week 15 Hackathon.
