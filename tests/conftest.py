"""Pytest configuration — ensures correct import paths."""

import sys
from pathlib import Path

# Add src/ to Python path so uncertainty_rag is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))
