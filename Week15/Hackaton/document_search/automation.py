"""Automated document processing with file system monitoring."""

from __future__ import annotations

import time
from pathlib import Path

from config import UPLOADS_DIR
from pipeline import process_new_document_incremental
from logging_utils import configure_logging

# Optional dependency with graceful degradation
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileCreatedEvent

    HAS_WATCHDOG = True
except ImportError:
    HAS_WATCHDOG = False


class DocumentWatcher(FileSystemEventHandler):
    """
    File system event handler for automatic document processing.
    """

    def __init__(self, logger=None):
        """
        Initialize the document watcher.

        Args:
            logger: Optional logger instance
        """
        super().__init__()
        self.logger = logger or configure_logging()
        self.supported_extensions = {".pdf", ".docx", ".doc", ".txt", ".md"}
        self.processing = set()  # Track files being processed

    def on_created(self, event: FileCreatedEvent) -> None:
        """
        Handle file creation events.

        Args:
            event: File system event
        """
        if event.is_directory:
            return

        file_path = Path(event.src_path)

        # Check if supported file type
        if file_path.suffix.lower() not in self.supported_extensions:
            return

        # Avoid double-processing
        if file_path in self.processing:
            return

        self.logger.info(f"New document detected: {file_path.name}")
        self.processing.add(file_path)

        # Wait a moment for file to be fully written
        time.sleep(2)

        try:
            # Process the document
            result = process_new_document_incremental(file_path, self.logger)

            if result["status"] == "success":
                self.logger.info(
                    f"Successfully processed {file_path.name}: "
                    f"{result['num_chunks']} chunks added"
                )
            else:
                self.logger.error(
                    f"Failed to process {file_path.name}: "
                    f"{result.get('message', 'Unknown error')}"
                )

        except Exception as e:
            self.logger.error(f"Error processing {file_path.name}: {str(e)}")

        finally:
            self.processing.discard(file_path)


def start_watching(directory: Path = UPLOADS_DIR) -> Observer:
    """
    Start monitoring a directory for new documents.

    Args:
        directory: Directory to monitor

    Returns:
        Observer instance (must be kept alive)

    Raises:
        ImportError: If watchdog is not installed
    """
    if not HAS_WATCHDOG:
        raise ImportError("watchdog not installed. Install with: pip install watchdog")

    logger = configure_logging()

    # Ensure directory exists
    directory.mkdir(parents=True, exist_ok=True)

    # Create observer and event handler
    event_handler = DocumentWatcher(logger)
    observer = Observer()
    observer.schedule(event_handler, str(directory), recursive=False)

    # Start monitoring
    observer.start()
    logger.info(f"Started monitoring directory: {directory}")
    logger.info("Waiting for new documents... (Press Ctrl+C to stop)")

    return observer


def run_automation(directory: Path = UPLOADS_DIR) -> None:
    """
    Run the automation daemon indefinitely.

    Args:
        directory: Directory to monitor
    """
    logger = configure_logging()

    try:
        observer = start_watching(directory)

        # Keep running until interrupted
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Stopping automation...")
        observer.stop()
        observer.join()
        logger.info("Automation stopped")

    except Exception as e:
        logger.error(f"Automation error: {str(e)}")
        raise


if __name__ == "__main__":
    """
    Run the automation daemon from command line.
    
    Usage:
        python automation.py
    """
    run_automation()
