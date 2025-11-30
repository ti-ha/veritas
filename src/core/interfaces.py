"""
Protocol definitions for VERITAS components.

These protocols define the interfaces that all implementations must follow,
enabling dependency inversion and making the system highly extensible.
"""

from typing import Protocol, Dict, List
import numpy as np
from src.core.types import FeatureVector, Score, EnsembleResult, Classification


class AnalyzerProtocol(Protocol):
    """
    Protocol for feature extraction analyzers.

    All analyzers (KCDA, TDA, Fractal, Ergodic) must implement this interface.
    This allows easy addition of new analyzers without modifying existing code.
    """

    @property
    def name(self) -> str:
        """Unique identifier for this analyzer"""
        ...

    @property
    def feature_count(self) -> int:
        """Number of features this analyzer extracts"""
        ...

    def extract_features(self, text: str) -> np.ndarray:
        """
        Extract features from text.

        Args:
            text: Input text to analyze

        Returns:
            Feature vector as numpy array

        Raises:
            FeatureExtractionError: If extraction fails
        """
        ...

    def validate_features(self, features: np.ndarray) -> bool:
        """
        Validate extracted features.

        Args:
            features: Feature vector to validate

        Returns:
            True if features are valid, False otherwise
        """
        ...


class ScorerProtocol(Protocol):
    """
    Protocol for feature scoring strategies.

    Scorers convert feature vectors into AI probability scores.
    This enables swapping between heuristic and ML-based scoring.
    """

    def score(self, features: np.ndarray, analyzer_name: str) -> Score:
        """
        Score features for AI likelihood.

        Args:
            features: Feature vector from an analyzer
            analyzer_name: Name of the analyzer that produced features

        Returns:
            Score object with probability and confidence

        Raises:
            ScoringError: If scoring fails
        """
        ...


class VoterProtocol(Protocol):
    """
    Protocol for ensemble voting strategies.

    Voters combine scores from multiple analyzers into a single result.
    This enables different voting strategies (weighted, majority, etc.)
    """

    def vote(self, scores: Dict[str, Score]) -> EnsembleResult:
        """
        Combine multiple scores into ensemble result.

        Args:
            scores: Dictionary mapping analyzer names to scores

        Returns:
            EnsembleResult with combined score and metadata

        Raises:
            ClassificationError: If voting fails
        """
        ...


class ExplainerProtocol(Protocol):
    """
    Protocol for explanation generation.

    Explainers generate human-readable explanations of classifications.
    This enables different explanation strategies and templates.
    """

    def explain(
        self,
        classification: Classification,
        text: str,
        scores: Dict[str, Score]
    ) -> str:
        """
        Generate explanation for classification.

        Args:
            classification: Classification result
            text: Original text that was classified
            scores: Individual analyzer scores

        Returns:
            Human-readable explanation string
        """
        ...


class ConfigProvider(Protocol):
    """
    Protocol for configuration management.

    Config providers abstract away configuration sources,
    enabling environment-based configuration and easy testing.
    """

    def get(self, key: str, default=None):
        """Get configuration value"""
        ...

    def get_int(self, key: str, default: int = 0) -> int:
        """Get integer configuration value"""
        ...

    def get_float(self, key: str, default: float = 0.0) -> float:
        """Get float configuration value"""
        ...

    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get boolean configuration value"""
        ...


class CacheProvider(Protocol):
    """
    Protocol for caching strategies.

    Cache providers abstract caching implementation,
    enabling different backends (memory, Redis, etc.)
    """

    def get(self, key: str):
        """Get cached value"""
        ...

    def set(self, key: str, value, ttl: int = None):
        """Set cached value with optional TTL"""
        ...

    def delete(self, key: str):
        """Delete cached value"""
        ...

    def clear(self):
        """Clear all cached values"""
        ...


class ModelRepository(Protocol):
    """
    Protocol for model persistence.

    Model repositories abstract storage of ML models,
    enabling different backends (filesystem, S3, database, etc.)
    """

    def load_model(self, model_name: str):
        """Load a trained model"""
        ...

    def save_model(self, model_name: str, model):
        """Save a trained model"""
        ...

    def list_models(self) -> List[str]:
        """List available models"""
        ...


class MetricsCollector(Protocol):
    """
    Protocol for metrics collection.

    Metrics collectors abstract metrics gathering,
    enabling different backends (Prometheus, StatsD, etc.)
    """

    def inc_counter(self, name: str, value: int = 1, tags: Dict[str, str] = None):
        """Increment a counter"""
        ...

    def set_gauge(self, name: str, value: float, tags: Dict[str, str] = None):
        """Set a gauge value"""
        ...

    def time_histogram(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record timing histogram"""
        ...
