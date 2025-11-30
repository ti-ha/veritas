"""
Core abstractions and interfaces for VERITAS.

This module defines the core protocols and types used throughout the system.
"""

from src.core.interfaces import (
    AnalyzerProtocol,
    ScorerProtocol,
    VoterProtocol,
    ExplainerProtocol,
)
from src.core.types import (
    FeatureVector,
    Score,
    EnsembleResult,
    Classification,
    ClassificationLevel,
)
from src.core.exceptions import (
    VERITASError,
    FeatureExtractionError,
    ScoringError,
    ClassificationError,
)

__all__ = [
    # Protocols
    "AnalyzerProtocol",
    "ScorerProtocol",
    "VoterProtocol",
    "ExplainerProtocol",
    # Types
    "FeatureVector",
    "Score",
    "EnsembleResult",
    "Classification",
    "ClassificationLevel",
    # Exceptions
    "VERITASError",
    "FeatureExtractionError",
    "ScoringError",
    "ClassificationError",
]
