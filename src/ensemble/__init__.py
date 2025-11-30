"""
Ensemble voting strategies for VERITAS.

This module provides voters that combine scores from multiple analyzers
into a single ensemble result.
"""

from src.ensemble.weighted_voter import WeightedVoter

__all__ = [
    "WeightedVoter",
]
