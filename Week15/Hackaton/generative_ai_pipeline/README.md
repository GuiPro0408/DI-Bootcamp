# Generative AI Pipeline

CPU-friendly generative AI workflow combining LLMs, quality control, ethical filtering, VAE image synthesis, and automation. Built with modular architecture following Week15 Hackathon conventions.

## Features

**Text Generation**
- `distilgpt2` and `t5-small` models via Hugging Face
- IMDB dataset for prompt bootstrapping
- Batch processing with configurable size

**Quality & Ethics**
- Automated validation: embeddings, perplexity, length checks
- Ethical filtering: rule-based + optional `toxic-bert` classifier
- Comprehensive evaluation: BLEU, ROUGE, perplexity, robustness

**Automation & UI**
- Streamlit dashboard (generate, batch, schedule, evaluate)
- CLI scheduler (hourly/daily/on-demand)
- Lightweight Conv-VAE for CIFAR-10 images (optional)

## Quick Start

```bash
# 1. Setup environment
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt')"  # Optional for BLEU

# 2. Run Streamlit UI
streamlit run app.py

# 3. Or use CLI pipeline
python -c "from pipeline import run_pipeline; run_pipeline()"
```

Models download automatically on first use. VAE requires `torch` and `torchvision`.

## Pipeline Architecture

```
IMDB data → text_generation → quality_control → ethical_filter → evaluation → outputs
                            ↘ [vae_image_gen] ↗              ↘ automation
```

**Modules**: `config.py` (constants) • `logging_utils.py` (logging) • `data_io.py` (I/O) • `pipeline.py` (orchestration)

## Usage Examples

### Streamlit Dashboard
```bash
streamlit run app.py
```
Navigate sidebar: **Generate** | **Batch** | **Schedule** | **Evaluate**

### Python API
```python
from pipeline import run_pipeline

# Basic usage
results = run_pipeline()

# Custom models
results = run_pipeline(
    models=["distilgpt2", "t5-small"],
    enable_vae=False
)
print(results.metrics)
```

### Automated Scheduling
```bash
python -m run_automation \
  --interval daily \
  --time 09:00 \
  --prompts-file prompts.txt \
  --models distilgpt2,t5-small
```
Logs: `outputs/logs/automation.log`

## Configuration

Key settings in `config.py`:

| Setting | Default | Purpose |
| --- | --- | --- |
| `TEXT_MODEL` | `distilgpt2` | Primary generation model |
| `ALT_TEXT_MODEL` | `t5-small` | Secondary model |
| `QUALITY_MODEL` | `distilbert-base-uncased` | QC backbone |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Similarity checks |
| `TOXICITY_MODEL` | `unitary/toxic-bert` | Ethical filter |
| `QUALITY_SIMILARITY_THRESHOLD` | `0.5` | Cosine similarity cutoff |
| `ETHICAL_SEVERITY_THRESHOLD` | `0.6` | Flagging threshold |
| `BATCH_SIZE` | `4` | Generation batch size |

## Output Structure

```
outputs/
├── generated_texts/     # Raw JSON generations
├── quality_reports/     # Per-sample QC results
├── ethical_flags/       # Flagged content + reasons
├── images/              # VAE-generated PNGs
├── evaluation_reports/  # Aggregated metrics
├── models/              # VAE checkpoints
└── logs/                # Pipeline & automation logs
```

## Evaluation Metrics

| Metric | Description | Implementation |
| --- | --- | --- |
| **BLEU** | N-gram overlap with references | NLTK with smoothing |
| **ROUGE** | Recall-oriented metrics (1/2/L) | `rouge-score` with stemming |
| **Perplexity** | LM quality score | Via quality checker model |
| **Robustness** | Avg length under adversarial prompts | Word swaps, typos, bias triggers |

## Ethical Considerations

⚠️ **Important Limitations**
- Filter uses regex rules + optional toxicity classifier
- False positives possible in nuanced language
- Manual review recommended for production use
- Models may amplify training data biases

**Best Practices**
- Review flagged content before deployment
- Adjust thresholds for your domain (`ETHICAL_SEVERITY_THRESHOLD`)
- Document model limitations for end users
- Use generated content responsibly

## Troubleshooting

| Issue | Solution |
| --- | --- |
| Missing dependencies | `pip install -r requirements.txt` |
| Slow generation | Reduce `BATCH_SIZE`, shorten max length, or use single model |
| Memory errors | Limit IMDB sample percentage or batch prompts |
| Perplexity unavailable | Install `torch` and model weights |
| VAE not working | Install `torch` and `torchvision` (optional feature) |
| Streamlit scheduler | Use CLI `run_automation.py` for persistent jobs |

---

**Built with**: Hugging Face Transformers • Streamlit • NLTK • scikit-learn • PyTorch (optional)  
**Architecture**: Follows DI-Bootcamp Week15 modular pipeline conventions
