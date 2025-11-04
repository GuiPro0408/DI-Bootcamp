# Quick Start Guide

Get the AI Document Search system running in 5 minutes!

## 1. Install Dependencies

```bash
cd Week15/Hackaton/document_search
pip install -r requirements.txt
```

This will install:

- sentence-transformers (embeddings)
- faiss-cpu (vector search)
- transformers (summarization)
- streamlit (web interface)
- PyPDF2, python-docx (document parsing)
- Additional utilities

**Note:** First run will download models (~640MB total):

- all-MiniLM-L6-v2 (~80MB)
- facebook/bart-base (~560MB)

## 2. Try Sample Documents

We've included sample documents in the `data/` folder:

- `sample_ml_overview.txt` - Machine Learning overview
- `sample_nlp_overview.txt` - NLP overview

Copy them to the uploads folder:

```bash
mkdir -p uploads
cp data/*.txt uploads/
```

## 3. Option A: Quick Test with CLI

Run the pipeline to process documents:

```bash
python run_pipeline.py
```

This will:

- Extract text from documents in `uploads/`
- Split into chunks
- Generate embeddings
- Build FAISS index
- Run evaluation metrics
- Save artifacts to `outputs/`

## 4. Option B: Launch Web Interface

Start Streamlit (recommended):

```bash
streamlit run app.py
```

Opens in your browser at `http://localhost:8501`

**Try these queries:**

- "What is supervised learning?"
- "How does machine translation work?"
- "Explain reinforcement learning"
- "What are transformers in NLP?"

## 5. Upload Your Own Documents

Via **Streamlit UI:**

1. Use sidebar file uploader
2. Select PDF, DOCX, or TXT files
3. Click "Process Uploaded Documents"
4. Wait for processing to complete
5. Start searching!

Via **CLI:**

1. Copy documents to `uploads/` folder
2. Run `python run_pipeline.py`
3. Launch Streamlit: `streamlit run app.py`

## 6. Enable Automation (Bonus)

Auto-process new documents:

```bash
python automation.py
```

Now any file added to `uploads/` is automatically indexed!

## Example Workflow

```bash
# 1. Install
pip install -r requirements.txt

# 2. Add sample documents
cp data/*.txt uploads/

# 3. Process
python run_pipeline.py

# 4. Launch UI
streamlit run app.py

# 5. Search and explore!
```

## Troubleshooting

**"No module named 'sentence_transformers'"**

```bash
pip install sentence-transformers
```

**"FAISS index not found"**

- Process documents first with CLI or Streamlit

**"Out of memory"**

- Reduce `BATCH_SIZE` in `config.py` to 2
- Limit documents with `MAX_DOCS` setting

**Slow summarization**

- Reduce `SUMMARY_MAX_LENGTH` in `config.py`
- Or skip summarization (disable in UI checkbox)

## What's Next?

- Check `outputs/` for generated artifacts
- Review `outputs/logs/pipeline.log` for details
- Read `README.md` for full documentation
- Customize `config.py` for your needs

## Quick Python API Example

```python
from embedding_engine import EmbeddingEngine
from vector_store import load_vector_store

# Load components
engine = EmbeddingEngine()
engine.load_model()
store = load_vector_store()

# Search
query = "What is machine learning?"
embedding = engine.embed_texts([query])
results = store.search_similar(embedding[0], top_k=3)

# Display
for i, result in enumerate(results, 1):
    print(f"{i}. {result['text'][:100]}...")
```

Enjoy exploring your documents! 🔍
