"""
Scoring strategies for VERITAS.

This module provides scorers that convert feature vectors into
AI probability scores.
"""

from src.scorers.heuristic import HeuristicScorer

__all__ = [
    "HeuristicScorer",
]
