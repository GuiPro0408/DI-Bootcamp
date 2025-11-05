"""CLI entry point for scheduling the generative AI pipeline."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional

# Suppress tokenizers parallelism warning before imports
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from automation import ScheduledPipeline


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for the automation runner."""
    parser = argparse.ArgumentParser(description="Schedule the generative AI pipeline.")
    parser.add_argument(
        "--interval",
        choices=["hourly", "daily", "on-demand"],
        default="hourly",
        help="Scheduling cadence for the pipeline.",
    )
    parser.add_argument(
        "--time",
        dest="time_of_day",
        help="Time of day in HH:MM (24h) for daily runs.",
    )
    parser.add_argument(
        "--prompts-file",
        type=Path,
        help="Path to a text file containing one prompt per line.",
    )
    parser.add_argument(
        "--models",
        type=lambda value: value.split(","),
        help="Comma-separated list of model names to use for generation.",
    )
    parser.add_argument(
        "--vae",
        action="store_true",
        help="Enable VAE image generation stage.",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=60,
        help="Seconds between scheduler polls.",
    )
    return parser.parse_args(argv)


def load_prompts_from_file(path: Optional[Path]) -> Optional[List[str]]:
    """Load prompts from a newline-separated file."""
    if not path:
        return None
    with path.open("r", encoding="utf-8") as handle:
        prompts = [line.strip() for line in handle if line.strip()]
    return prompts or None


def main(argv: Optional[List[str]] = None) -> None:
    """Entrypoint for automation CLI."""
    args = parse_args(argv)
    prompts = load_prompts_from_file(args.prompts_file)
    pipeline = ScheduledPipeline(prompts=prompts, models=args.models, enable_vae=args.vae)

    pipeline.setup_schedule(
        interval=args.interval,
        time_of_day=args.time_of_day,
        prompt_templates=prompts,
        models=args.models,
        enable_vae=args.vae,
    )

    if args.interval != "on-demand":
        pipeline.run_scheduler(poll_interval=args.poll_interval)


if __name__ == "__main__":
    main()
