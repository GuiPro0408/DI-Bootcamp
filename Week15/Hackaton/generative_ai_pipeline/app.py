"""Streamlit interface for the generative AI pipeline."""

from __future__ import annotations

import json
import os
import threading
from typing import Dict, List, Optional

# Suppress tokenizers parallelism warning before imports
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st

from automation import ScheduledPipeline
from config import ALT_TEXT_MODEL, TEXT_MODEL
from data_io import load_imdb_subset
from ethical_filter import EthicalFilter, FilterResult
from logging_utils import configure_logging
from pipeline import PipelineResults, run_pipeline
from quality_control import QualityChecker, QualityResult
from text_generation import GenerationResult, TextGenerator

LOGGER = configure_logging("generative_ai_pipeline.app")

st.set_page_config(page_title="Generative AI Pipeline", page_icon="🤖", layout="wide")


@st.cache_resource
def load_text_generator() -> TextGenerator:
    """Cached loader for text generator."""
    return TextGenerator()


@st.cache_resource
def load_quality_checker() -> Optional[QualityChecker]:
    """Cached loader for quality checker."""
    try:
        return QualityChecker()
    except ImportError as exc:
        LOGGER.warning(
            "QualityChecker unavailable in Streamlit app", extra={"reason": str(exc)}
        )
        return None


@st.cache_resource
def load_ethical_filter() -> Optional[EthicalFilter]:
    """Cached loader for ethical filter."""
    try:
        return EthicalFilter()
    except ImportError as exc:
        LOGGER.warning(
            "EthicalFilter unavailable in Streamlit app", extra={"reason": str(exc)}
        )
        return None


def display_quality_badge(result: Optional[QualityResult]) -> str:
    """Return emoji badge for quality outcome."""
    if result is None:
        return "ℹ️"
    return "✅" if result.passed else "⚠️"


def render_generation(
    result: GenerationResult,
    quality: Optional[QualityResult],
    ethics: Optional[FilterResult],
) -> None:
    """Render a single generation with quality and ethics details."""
    badge = display_quality_badge(quality)
    st.markdown(
        f"**{badge} Model:** `{result.model_name}` • *{result.generation_time:.2f}s*"
    )
    st.write(result.generated_text)

    if quality:
        with st.expander("Quality details"):
            st.json(quality.to_dict())
    if ethics:
        with st.expander("Ethical scan"):
            st.json(ethics.to_dict())


def handle_generate_tab(models: List[str]) -> None:
    """Interactive prompt-based generation."""
    # Help box
    with st.expander("ℹ️ How to use Generate", expanded=False):
        st.markdown("""
        **Purpose**: Generate text from a single custom prompt using LLM models.
        
        **How it works**:
        1. Enter your prompt (e.g., "Write a review about a sci-fi movie")
        2. Select a model (`distilgpt2` or `t5-small`)
        3. Adjust temperature (creativity) and max length
        4. Click Generate to create text
        
        **Example**:
        ```
        Prompt: "Write a positive movie review about a space adventure"
        Temperature: 0.7 (balanced creativity)
        Max length: 100 tokens
        
        Result: AI-generated review with quality score and ethical scan
        ```
        
        **Quality indicators**: ✅ Pass | ⚠️ Warning | ❌ Fail  
        **Ethical scan**: Flags potentially toxic or biased content
        """)

    try:
        generator = load_text_generator()
    except ImportError as exc:
        st.error(f"Text generation unavailable: {exc}")
        return

    quality_checker = load_quality_checker()
    ethical_filter = load_ethical_filter()

    prompt = st.text_area(
        "Enter a prompt", height=150, placeholder="Write a movie review about..."
    )
    selected_model = st.selectbox("Choose model", options=models, index=0)
    temperature = st.slider(
        "Temperature", min_value=0.1, max_value=1.5, value=0.7, step=0.1
    )
    max_length = st.slider(
        "Max length", min_value=30, max_value=200, value=100, step=10
    )

    if st.button("Generate", type="primary", disabled=not prompt.strip()):
        with st.spinner("Generating..."):
            results = generator.generate_from_prompt(
                prompt=prompt,
                temperature=temperature,
                max_length=max_length,
                model_name=selected_model,
            )

        for result in results:
            quality_result = (
                quality_checker.validate_output(result.prompt, result.generated_text)
                if quality_checker
                else None
            )
            ethics_result = (
                ethical_filter.scan_text(result.generated_text)
                if ethical_filter
                else None
            )
            render_generation(result, quality_result, ethics_result)


def handle_batch_tab(models: List[str]) -> None:
    """Batch generation workflow with file upload."""
    # Help box
    with st.expander("ℹ️ How to use Batch", expanded=False):
        st.markdown("""
        **Purpose**: Generate text for multiple prompts at once from a file or text input.
        
        **How it works**:
        1. Upload a `.txt` file with one prompt per line, OR paste prompts directly
        2. Select one or more models to use
        3. Click Generate Batch
        4. Download results as JSON
        
        **Example file** (`prompts.txt`):
        ```
        Write a thriller movie review
        Summarize a romantic comedy plot
        Describe a horror film scene
        Review an action adventure movie
        ```
        
        **Output**: JSON file with all generations, quality scores, and ethical flags.
        
        **Use case**: Test multiple prompts, compare models, or generate training data.
        """)

    try:
        generator = load_text_generator()
    except ImportError as exc:
        st.error(f"Text generation unavailable: {exc}")
        return

    uploaded = st.file_uploader("Upload prompts (.txt)", type=["txt"])
    selected_models = st.multiselect("Models", options=models, default=models)
    batch_size = st.slider("Batch size", min_value=1, max_value=10, value=4)

    if st.button(
        "Run batch generation", disabled=uploaded is None or not selected_models
    ):
        prompts = uploaded.read().decode("utf-8").splitlines()
        prompts = [line.strip() for line in prompts if line.strip()]
        if not prompts:
            st.warning("No prompts found in uploaded file.")
            return

        quality_checker = load_quality_checker()
        ethical_filter = load_ethical_filter()

        progress = st.progress(0, text="Starting batch generation...")
        aggregated: List[Dict[str, object]] = []
        total = len(prompts) * len(selected_models)
        completed = 0

        for model_name in selected_models:
            results = generator.batch_generate(
                prompts, batch_size=batch_size, model_name=model_name
            )
            for result in results:
                quality_result = (
                    quality_checker.validate_output(
                        result.prompt, result.generated_text
                    )
                    if quality_checker
                    else None
                )
                ethics_result = (
                    ethical_filter.scan_text(result.generated_text)
                    if ethical_filter
                    else None
                )

                aggregated.append(
                    {
                        "prompt": result.prompt,
                        "generated_text": result.generated_text,
                        "model": result.model_name,
                        "quality": quality_result.to_dict() if quality_result else None,
                        "ethics": ethics_result.to_dict() if ethics_result else None,
                    }
                )
                completed += 1
                progress.progress(
                    completed / total, text=f"Processed {completed}/{total}"
                )

        st.success("Batch generation complete.")
        st.download_button(
            "Download results",
            data=json.dumps(aggregated, ensure_ascii=False, indent=2),
            file_name="batch_results.json",
            mime="application/json",
        )


def _init_scheduler_state() -> None:
    """Ensure scheduler-related session state keys exist."""
    if "scheduler" not in st.session_state:
        st.session_state.scheduler = None
        st.session_state.scheduler_thread = None
        st.session_state.scheduler_running = False


def handle_schedule_tab(default_models: List[str]) -> None:
    """Render scheduling controls."""
    # Help box
    with st.expander("ℹ️ How to use Schedule", expanded=False):
        st.markdown("""
        **Purpose**: Automate text generation at scheduled intervals.
        
        **How it works**:
        1. Choose interval: `hourly`, `daily`, or `on-demand` (run once)
        2. For daily: select start time (e.g., 09:00)
        3. Enter prompt templates (one per line)
        4. Select models to use
        5. Click Start Scheduler
        
        **Example - Daily content generation**:
        ```
        Interval: daily
        Time: 09:00 AM
        Prompts:
          - Write today's movie recommendation
          - Generate a film industry news summary
          - Create a classic movie review
        Models: distilgpt2, t5-small
        ```
        
        **Use case**: Automated content creation, daily reports, or periodic data generation.
        
        **Logs**: Check `outputs/logs/automation.log` for scheduler activity.
        """)

    _init_scheduler_state()
    interval = st.radio("Interval", options=["hourly", "daily", "on-demand"])
    time_of_day = None
    if interval == "daily":
        selected_time = st.time_input("Start time", value=None)
        if selected_time:
            time_of_day = selected_time.strftime("%H:%M")

    prompt_template = st.text_area(
        "Prompt templates (one per line)",
        placeholder="Write a positive review about...\nSummarize the plot of...",
    )
    models = st.multiselect("Models", options=default_models, default=default_models)
    enable_vae = st.toggle("Enable VAE image generation", value=False)

    col1, col2 = st.columns(2)
    status_placeholder = st.empty()

    def start_scheduler() -> None:
        prompts = [
            line.strip() for line in prompt_template.splitlines() if line.strip()
        ] or None
        scheduler = ScheduledPipeline(
            prompts=prompts, models=models, enable_vae=enable_vae
        )
        scheduler.setup_schedule(
            interval=interval,
            time_of_day=time_of_day,
            prompt_templates=prompts,
            models=models,
            enable_vae=enable_vae,
        )

        if interval != "on-demand":
            thread = threading.Thread(
                target=scheduler.run_scheduler,
                kwargs={"poll_interval": 60},
                daemon=True,
            )
            thread.start()
            st.session_state.scheduler = scheduler
            st.session_state.scheduler_thread = thread
            st.session_state.scheduler_running = True
        else:
            st.session_state.scheduler = None
            st.session_state.scheduler_thread = None
            st.session_state.scheduler_running = False

    def stop_scheduler() -> None:
        scheduler = st.session_state.scheduler
        if scheduler:
            scheduler.stop()
        st.session_state.scheduler_running = False

    with col1:
        if st.button("Start scheduler", type="primary"):
            start_scheduler()
            status_placeholder.success(
                "Scheduler started."
                if interval != "on-demand"
                else "Pipeline executed once."
            )
    with col2:
        if st.button("Stop scheduler", disabled=not st.session_state.scheduler_running):
            stop_scheduler()
            status_placeholder.info("Scheduler stopped.")

    if st.session_state.scheduler_running:
        status_placeholder.info("Scheduler running...")


def handle_evaluate_tab(models: List[str]) -> None:
    """Render evaluation dashboard and quick pipeline trigger."""
    # Help box
    with st.expander("ℹ️ How to use Evaluate", expanded=False):
        st.markdown("""
        **Purpose**: Run the full pipeline with IMDB prompts and evaluate generation quality.
        
        **How it works**:
        1. Set number of IMDB prompts to use (5-200)
        2. Select models to compare
        3. Enable/disable VAE image generation (optional)
        4. Click "Run pipeline now"
        5. View aggregated metrics and artifacts
        
        **Evaluation metrics**:
        - **BLEU**: N-gram overlap with references (0-1, higher = better)
        - **ROUGE**: Recall-oriented metrics for summaries
        - **Perplexity**: Language model quality score (lower = better)
        - **Robustness**: Performance under adversarial prompts
        
        **Example**:
        ```
        IMDB prompts: 20
        Models: distilgpt2, t5-small
        
        Results:
          Generated: 40 texts (20 per model)
          Passed Quality: 35
          Flagged: 2 (ethical issues)
          BLEU: 0.3452
          Perplexity: 45.23
        ```
        
        **Artifacts**: Check `outputs/` for detailed reports and generated texts.
        """)

    if "last_pipeline_results" not in st.session_state:
        st.session_state.last_pipeline_results = None

    prompts_count = st.number_input(
        "Number of IMDB prompts", min_value=5, max_value=200, value=20, step=5
    )
    selected_models = st.multiselect("Models", options=models, default=models)
    enable_vae = st.toggle("Enable VAE", value=False, key="eval_vae")

    if st.button("Run pipeline now"):
        with st.spinner("Running pipeline..."):
            dataset = load_imdb_subset(split="train", sample_pct=0.05, stratify=True)
            prompts = dataset.prompts[: int(prompts_count)]
            results = run_pipeline(
                prompts=prompts,
                models=selected_models or models,
                enable_vae=enable_vae,
                run_evaluation=True,
            )
        st.session_state.last_pipeline_results = results
        st.success("Pipeline finished. Metrics updated.")

    results: Optional[PipelineResults] = st.session_state.last_pipeline_results
    if results:
        st.subheader("Metrics")
        metric_cols = st.columns(3)
        metric_cols[0].metric("Generated", results.num_generated)
        metric_cols[1].metric("Passed Quality", results.num_passed_quality)
        metric_cols[2].metric("Flagged", results.num_flagged)

        if results.metrics:
            st.subheader("Evaluation scores")
            for key, value in results.metrics.items():
                st.metric(
                    key.upper(), f"{value:.4f}" if isinstance(value, float) else value
                )
        if results.artifacts.evaluation_reports:
            st.write(
                "Latest evaluation artifact:", results.artifacts.evaluation_reports[-1]
            )

        if results.artifacts.generated_texts:
            st.write(
                "Generated outputs saved at:", results.artifacts.generated_texts[-1]
            )
    else:
        st.info("Run the pipeline to populate evaluation metrics.")


def main() -> None:
    """Main entry point for Streamlit UI."""
    default_models = [TEXT_MODEL, ALT_TEXT_MODEL]
    sidebar_selection = st.sidebar.radio(
        "Go to", options=["Generate", "Batch", "Schedule", "Evaluate"]
    )

    if sidebar_selection == "Generate":
        handle_generate_tab(default_models)
    elif sidebar_selection == "Batch":
        handle_batch_tab(default_models)
    elif sidebar_selection == "Schedule":
        handle_schedule_tab(default_models)
    elif sidebar_selection == "Evaluate":
        handle_evaluate_tab(default_models)


if __name__ == "__main__":
    main()
