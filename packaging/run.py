"""Entry point for the PyInstaller build (the pip install uses the
`gear360-stitch` console script instead)."""
import multiprocessing
import sys

from gear360_stitcher.cli import main

if __name__ == "__main__":
    # Needed so --workers > 1 does not make a frozen Windows build re-run itself
    # in every worker process.
    multiprocessing.freeze_support()
    sys.exit(main())
