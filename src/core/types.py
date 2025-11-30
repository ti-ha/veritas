"""
Domain types and data models for VERITAS.

These types define the core data structures used throughout the system,
providing type safety and clear contracts between components.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
import numpy as np


class ClassificationLevel(Enum):
    """Classification confidence levels"""
    DEFINITIVE = 1       # High confidence, strong signal
    PROBABILISTIC = 2    # Moderate confidence, majority agreement
    INCONCLUSIVE = 3     # Low confidence, significant disagreement


@dataclass(frozen=True)
class FeatureVector:
    """
    Feature vector from an analyzer.

    Immutable to prevent accidental modification.
    """
    analyzer_name: str
    features: np.ndarray
    feature_count: int

    def __post_init__(self):
        """Validate feature vector"""
        if len(self.features) != self.feature_count:
            raise ValueError(
                f"Feature count mismatch: expected {self.feature_count}, "
                f"got {len(self.features)}"
            )


@dataclass(frozen=True)
class Score:
    """
    Score from a single analyzer or scorer.

    Represents the AI probability score and associated metadata.
    """
    analyzer_name: str
    probability: float  # 0.0 to 1.0
    confidence: float   # 0.0 to 1.0
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate score values"""
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(f"Probability must be 0-1, got {self.probability}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be 0-1, got {self.confidence}")


@dataclass(frozen=True)
class EnsembleResult:
    """
    Result from ensemble voting.

    Combines multiple scores with agreement metrics and outlier detection.
    """
    weighted_probability: float
    agreement_score: float
    outlier_modules: List[str]
    individual_scores: Dict[str, Score]
    metadata: Dict = field(default_factory=dict)


@dataclass(frozen=True)
class Classification:
    """
    Final classification result.

    Complete result including level, probability, and all metadata.
    """
    level: ClassificationLevel
    ai_probability: float
    confidence: float
    explanation: str
    word_count: int
    module_scores: Dict[str, float]
    features: Dict = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for API responses"""
        return {
            "classification_level": self.level.value,
            "ai_probability": float(self.ai_probability),
            "confidence": float(self.confidence),
            "explanation": self.explanation,
            "word_count": self.word_count,
            "module_scores": {k: float(v) for k, v in self.module_scores.items()},
            "features": self.features,
            "metadata": self.metadata,
        }


@dataclass
class AnalyzerConfig:
    """Configuration for an analyzer"""
    name: str
    enabled: bool = True
    weight: float = 1.0
    params: Dict = field(default_factory=dict)


@dataclass
class EnsembleConfig:
    """Configuration for ensemble voting"""
    voting_strategy: str = "weighted"  # weighted, majority, unanimous
    outlier_detection: bool = True
    outlier_threshold: float = 1.5  # IQR multiplier
    agreement_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'high': 0.85,
        'moderate': 0.70,
        'low': 0.50
    })


@dataclass
class ClassificationConfig:
    """Configuration for classification"""
    thresholds: Dict[str, float] = field(default_factory=lambda: {
        'definitive_ai': 0.75,
        'probable_ai': 0.60,
        'ambiguous_high': 0.55,
        'ambiguous_low': 0.45,
        'probable_human': 0.40,
        'definitive_human': 0.25
    })
    min_text_length: int = 50
    enable_caching: bool = True
    cache_ttl: int = 3600  # seconds


@dataclass
class TextMetadata:
    """Metadata about input text"""
    length: int
    word_count: int
    sentence_count: int
    language: Optional[str] = None
    source: Optional[str] = None


@dataclass
class AnalysisContext:
    """
    Context for an analysis request.

    Carries metadata and configuration through the analysis pipeline.
    """
    text: str
    text_metadata: TextMetadata
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    timestamp: Optional[float] = None
    options: Dict = field(default_factory=dict)
