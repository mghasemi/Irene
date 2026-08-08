"""Pytest configuration for IreneRewrite."""
import sys
from pathlib import Path

# Ensure the project root is on sys.path so `Irene` is importable
sys.path.insert(0, str(Path(__file__).parent))
