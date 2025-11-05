"""Scheduling helpers for the generative AI pipeline."""

from __future__ import annotations

import json
import time
from typing import Callable, List, Optional, Sequence

import schedule

from config import LOG_DIR, TEXT_MODEL
from logging_utils import configure_logging
from pipeline import PipelineResults, run_pipeline

AUTOMATION_LOG = LOG_DIR / "automation.log"


class ScheduledPipeline:
    """Utility for configuring and running scheduled pipeline executions."""

    def __init__(
        self,
        pipeline_fn: Callable[..., PipelineResults] = run_pipeline,
        prompts: Optional[Sequence[str]] = None,
        models: Optional[Sequence[str]] = None,
        enable_vae: bool = False,
    ) -> None:
        self.pipeline_fn = pipeline_fn
        self.prompts = list(prompts) if prompts else None
        self.models = list(models) if models else [TEXT_MODEL]
        self.enable_vae = enable_vae

        self.logger = configure_logging("generative_ai_pipeline.automation")
        AUTOMATION_LOG.parent.mkdir(parents=True, exist_ok=True)

        self._jobs: List[schedule.Job] = []
        self._running = False

    def _log_execution(self, status: str, metrics: Optional[dict] = None) -> None:
        """Write a structured automation log entry."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] [SCHEDULED] {status}"
        if metrics:
            compact_metrics = json.dumps(metrics, ensure_ascii=False)
            line += f" - {compact_metrics}"
        with AUTOMATION_LOG.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        self.logger.info("Recorded automation execution", extra={"status": status, "metrics": metrics or {}})

    def _run_once(self) -> None:
        """Execute the pipeline once and log outcome."""
        try:
            results = self.pipeline_fn(
                prompts=self.prompts,
                models=self.models,
                enable_vae=self.enable_vae,
                run_evaluation=True,
            )
            metrics = {
                "num_generated": results.num_generated,
                "num_passed_quality": results.num_passed_quality,
                "num_flagged": results.num_flagged,
                **results.metrics,
            }
            self._log_execution("SUCCESS", metrics)
        except Exception as exc:  # pragma: no cover - runtime safeguard
            self._log_execution("FAILED", {"error": str(exc)})
            self.logger.exception("Scheduled pipeline run failed")

    def setup_schedule(
        self,
        interval: str = "hourly",
        time_of_day: Optional[str] = None,
        prompt_templates: Optional[Sequence[str]] = None,
        models: Optional[Sequence[str]] = None,
        enable_vae: Optional[bool] = None,
    ) -> None:
        """Configure schedule based on provided parameters."""
        if prompt_templates is not None:
            self.prompts = list(prompt_templates)
        if models is not None:
            self.models = list(models)
        if enable_vae is not None:
            self.enable_vae = enable_vae

        for job in self._jobs:
            schedule.cancel_job(job)
        self._jobs.clear()

        if interval == "hourly":
            job = schedule.every(1).hours.do(self._run_once)
        elif interval == "daily":
            if not time_of_day:
                raise ValueError("time_of_day is required for daily schedules (format HH:MM).")
            job = schedule.every().day.at(time_of_day).do(self._run_once)
        elif interval == "on-demand":
            self._run_once()
            return
        else:
            raise ValueError(f"Unsupported interval '{interval}'. Choose from hourly, daily, on-demand.")

        self._jobs.append(job)
        self.logger.info(
            "Scheduler configured",
            extra={
                "interval": interval,
                "time_of_day": time_of_day,
                "num_prompts": len(self.prompts or []),
                "models": self.models,
                "enable_vae": self.enable_vae,
            },
        )

    def run_scheduler(self, poll_interval: int = 60) -> None:
        """Blocking loop that executes scheduled jobs."""
        self._running = True
        self.logger.info("Scheduler started", extra={"poll_interval": poll_interval})
        try:
            while self._running:
                schedule.run_pending()
                time.sleep(poll_interval)
        except KeyboardInterrupt:  # pragma: no cover - interactive use
            self.logger.info("Scheduler interrupted by user.")
        finally:
            self._running = False
            self.logger.info("Scheduler stopped")

    def stop(self) -> None:
        """Stop the scheduler loop."""
        self._running = False
