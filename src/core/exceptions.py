"""
Custom exceptions for VERITAS.

Provides specific exception types for different error scenarios,
enabling better error handling and debugging.
"""


class VERITASError(Exception):
    """Base exception for all VERITAS errors"""
    pass


class FeatureExtractionError(VERITASError):
    """Raised when feature extraction fails"""

    def __init__(self, analyzer_name: str, message: str, original_error: Exception = None):
        self.analyzer_name = analyzer_name
        self.original_error = original_error
        super().__init__(f"Feature extraction failed in {analyzer_name}: {message}")


class ScoringError(VERITASError):
    """Raised when scoring fails"""

    def __init__(self, scorer_name: str, message: str, original_error: Exception = None):
        self.scorer_name = scorer_name
        self.original_error = original_error
        super().__init__(f"Scoring failed in {scorer_name}: {message}")


class ClassificationError(VERITASError):
    """Raised when classification fails"""

    def __init__(self, message: str, original_error: Exception = None):
        self.original_error = original_error
        super().__init__(f"Classification failed: {message}")


class ValidationError(VERITASError):
    """Raised when input validation fails"""

    def __init__(self, field: str, message: str):
        self.field = field
        super().__init__(f"Validation error for '{field}': {message}")


class ConfigurationError(VERITASError):
    """Raised when configuration is invalid"""

    def __init__(self, key: str, message: str):
        self.key = key
        super().__init__(f"Configuration error for '{key}': {message}")


class ModelNotFoundError(VERITASError):
    """Raised when a required model is not found"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        super().__init__(f"Model not found: {model_name}")


class CacheError(VERITASError):
    """Raised when cache operations fail"""

    def __init__(self, operation: str, message: str):
        self.operation = operation
        super().__init__(f"Cache {operation} failed: {message}")
