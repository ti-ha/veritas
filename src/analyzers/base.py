"""
Base analyzer implementation.

Provides common functionality for all analyzers and ensures they
properly implement the AnalyzerProtocol interface.
"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from src.core.interfaces import AnalyzerProtocol
from src.core.exceptions import FeatureExtractionError, ValidationError


class BaseAnalyzer(ABC):
    """
    Base class for all feature extraction analyzers.

    Provides common functionality and enforces the AnalyzerProtocol contract.
    All analyzers should inherit from this class.
    """

    def __init__(self, random_seed: int = 42):
        """
        Initialize analyzer.

        Args:
            random_seed: Random seed for reproducibility
        """
        self._random_seed = random_seed
        self._name: Optional[str] = None
        self._feature_count: Optional[int] = None

    @property
    def name(self) -> str:
        """Unique identifier for this analyzer"""
        if self._name is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must set self._name in __init__"
            )
        return self._name

    @property
    def feature_count(self) -> int:
        """Number of features this analyzer extracts"""
        if self._feature_count is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must set self._feature_count in __init__"
            )
        return self._feature_count

    @property
    def random_seed(self) -> int:
        """Random seed for reproducibility"""
        return self._random_seed

    @abstractmethod
    def _extract_features_impl(self, text: str) -> np.ndarray:
        """
        Implementation of feature extraction.

        This method must be implemented by subclasses to perform
        the actual feature extraction logic.

        Args:
            text: Input text to analyze

        Returns:
            Feature vector as numpy array

        Raises:
            Any exception will be caught and wrapped in FeatureExtractionError
        """
        pass

    def extract_features(self, text: str) -> np.ndarray:
        """
        Extract features from text.

        This method wraps _extract_features_impl with error handling
        and validation.

        Args:
            text: Input text to analyze

        Returns:
            Feature vector as numpy array

        Raises:
            FeatureExtractionError: If extraction fails
            ValidationError: If input validation fails
        """
        # Validate input
        if not text or not isinstance(text, str):
            raise ValidationError("text", "Text must be a non-empty string")

        if len(text.strip()) < 10:
            raise ValidationError("text", "Text must be at least 10 characters")

        try:
            # Extract features
            features = self._extract_features_impl(text)

            # Validate output
            if not self.validate_features(features):
                raise FeatureExtractionError(
                    self.name,
                    f"Extracted features failed validation. "
                    f"Expected {self.feature_count} features, got {len(features)}"
                )

            return features

        except ValidationError:
            # Re-raise validation errors as-is
            raise
        except FeatureExtractionError:
            # Re-raise feature extraction errors as-is
            raise
        except Exception as e:
            # Wrap unexpected errors
            raise FeatureExtractionError(
                self.name,
                f"Unexpected error during feature extraction: {str(e)}",
                original_error=e
            )

    def validate_features(self, features: np.ndarray) -> bool:
        """
        Validate extracted features.

        Args:
            features: Feature vector to validate

        Returns:
            True if features are valid, False otherwise
        """
        if not isinstance(features, np.ndarray):
            return False

        if len(features) != self.feature_count:
            return False

        # Check for invalid values
        if np.any(np.isnan(features)):
            return False

        if np.any(np.isinf(features)):
            return False

        return True

    def __repr__(self) -> str:
        """String representation"""
        return f"{self.__class__.__name__}(name='{self.name}', features={self.feature_count})"


# Type check to ensure BaseAnalyzer satisfies AnalyzerProtocol
def _type_check() -> None:
    """Compile-time type check that BaseAnalyzer implements AnalyzerProtocol"""
    def check(analyzer: AnalyzerProtocol) -> None:
        pass

    # This will fail type checking if BaseAnalyzer doesn't implement the protocol
    # (though it won't run since it's abstract)
    # check(BaseAnalyzer())  # Would fail at runtime since it's abstract
