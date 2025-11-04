"""Command-line entry point for the document search pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from pipeline import run_pipeline


def main() -> None:
    """
    Run the document search pipeline from command line.
    """
    parser = argparse.ArgumentParser(
        description="AI-Powered Document Search and Summarization Pipeline"
    )
    parser.add_argument(
        "--documents-dir",
        type=Path,
        default=None,
        help="Directory containing documents to process (default: uploads/)",
    )
    parser.add_argument(
        "--no-eval", action="store_true", help="Skip evaluation metrics"
    )
    parser.add_argument(
        "--no-rebuild",
        action="store_true",
        help="Don't rebuild index from scratch (add to existing)",
    )

    args = parser.parse_args()

    # Run pipeline
    results = run_pipeline(
        documents_dir=args.documents_dir,
        run_evaluation=not args.no_eval,
        rebuild_index=not args.no_rebuild,
    )

    # Print summary
    if results["status"] == "success":
        print("\n✅ Pipeline completed successfully!")
        print(f"   Documents processed: {results['num_documents']}")
        print(f"   Chunks generated: {results['num_chunks']}")
        if "evaluation" in results:
            prec = results["evaluation"].get("precision@5", 0)
            rec = results["evaluation"].get("recall@5", 0)
            print(f"   Search quality: Precision@5={prec:.3f}, Recall@5={rec:.3f}")
        print("\nNext steps:")
        print("   - Run: streamlit run app.py")
        print("   - Or use the Python API to search documents")
    else:
        print(f"\n❌ Pipeline failed: {results.get('message', 'Unknown error')}")


if __name__ == "__main__":
    main()
