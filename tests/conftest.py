"""Pytest configuration for Claude Conversation Extractor tests."""

import sys
from pathlib import Path

# Add src directory to Python path for testing
src_dir = Path(__file__).parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Add tests directory to Python path so test modules can import helpers.py
tests_dir = Path(__file__).parent
if str(tests_dir) not in sys.path:
    sys.path.insert(0, str(tests_dir))