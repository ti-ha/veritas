"""
Heuristic-based scoring strategy.

Implements hand-crafted scoring rules based on domain knowledge
and theoretical understanding of each analyzer's features.
"""

import numpy as np
from src.core.types import Score
from src.core.exceptions import ScoringError


class HeuristicScorer:
    """
    Heuristic scorer that converts feature vectors to AI probability scores.

    Uses domain knowledge and theoretical understanding to score features
    from KCDA, TDA, Fractal, and Ergodic analyzers.
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
        try:
            # Dispatch to appropriate scoring method
            if analyzer_name == "kcda":
                probability, confidence = self._score_kcda(features)
            elif analyzer_name == "tda":
                probability, confidence = self._score_tda(features)
            elif analyzer_name == "fractal":
                probability, confidence = self._score_fractal(features)
            elif analyzer_name == "ergodic":
                probability, confidence = self._score_ergodic(features)
            else:
                raise ScoringError(
                    "HeuristicScorer",
                    f"Unknown analyzer: {analyzer_name}"
                )

            # Validate output
            if not 0.0 <= probability <= 1.0:
                raise ScoringError(
                    "HeuristicScorer",
                    f"Invalid probability {probability} for {analyzer_name}"
                )

            if not 0.0 <= confidence <= 1.0:
                raise ScoringError(
                    "HeuristicScorer",
                    f"Invalid confidence {confidence} for {analyzer_name}"
                )

            return Score(
                analyzer_name=analyzer_name,
                probability=probability,
                confidence=confidence,
                metadata={
                    'scoring_method': 'heuristic',
                    'feature_count': len(features)
                }
            )

        except ScoringError:
            raise
        except Exception as e:
            raise ScoringError(
                "HeuristicScorer",
                f"Unexpected error scoring {analyzer_name}: {str(e)}",
                original_error=e
            )

    def _score_kcda(self, features: np.ndarray) -> tuple[float, float]:
        """
        Score KCDA features for AI likelihood.

        AI text tends to be more compressible (lower compression ratios)
        and shows less cross-complexity variation.

        Args:
            features: 48-dimensional KCDA feature vector

        Returns:
            (probability, confidence) tuple
        """
        # Features breakdown:
        # 0-8: Compression ratios (3 algorithms × 3 scales)
        # 9-20: Cross-complexity statistics
        # 21-35: Divergence features
        # 36-47: Behavior features

        compression_ratios = features[:9]
        cross_complexity_stats = features[9:21]

        # Lower compression ratio suggests AI (more predictable)
        compression_score = 1.0 - np.mean(compression_ratios)

        # Lower cross-complexity variance suggests AI (more consistent)
        if np.std(cross_complexity_stats) > 0:
            consistency_score = 1.0 / (1.0 + np.std(cross_complexity_stats))
        else:
            consistency_score = 0.8

        probability = (compression_score + consistency_score) / 2

        # Confidence based on how clear the signal is
        confidence = 0.7 + 0.3 * abs(probability - 0.5) * 2

        return float(probability), float(confidence)

    def _score_tda(self, features: np.ndarray) -> tuple[float, float]:
        """
        Score TDA features for AI likelihood.

        RECALIBRATED: Return neutral score to reduce TDA bias.
        Original scoring showed systematic overestimation (0.76 avg on mixed data).

        Args:
            features: 64-dimensional TDA feature vector

        Returns:
            (probability, confidence) tuple
        """
        # Features breakdown:
        # 0-19: Distance-based topology
        # 20-39: Connectivity features
        # 40-51: Clustering coefficients
        # 52-63: Dimensional features

        all_features = features[:64]
        feature_variance = np.std(all_features) if len(all_features) > 0 else 0.5

        # Moderate scoring: higher variance = more human-like irregularity
        # Lower variance = more AI-like regularity
        probability = 0.5 + (0.3 * (0.5 - min(feature_variance, 1.0)))

        # Lower confidence due to calibration issues
        confidence = 0.5

        return float(np.clip(probability, 0.0, 1.0)), float(confidence)

    def _score_fractal(self, features: np.ndarray) -> tuple[float, float]:
        """
        Score fractal features for AI likelihood.

        AI text tends to have more uniform fractal properties.

        Args:
            features: 32-dimensional Fractal feature vector

        Returns:
            (probability, confidence) tuple
        """
        # Features breakdown:
        # 0-11: Box-counting dimensions
        # 12-21: Fractality degrees
        # 22-31: Multi-scale profile

        box_counting_features = features[:12]
        fractality_degrees = features[12:22]

        # Lower variance in fractality suggests AI (more uniform)
        fractality_variance = np.std(fractality_degrees) if len(fractality_degrees) > 0 else 0.5
        uniformity_score = 1.0 / (1.0 + fractality_variance * 2.0)

        # Use mean box-counting value directly (normalized)
        box_mean = np.mean(box_counting_features) if len(box_counting_features) > 0 else 0.5
        dimension_score = np.clip(box_mean, 0.0, 1.0)

        probability = uniformity_score * 0.6 + dimension_score * 0.4

        # Confidence based on signal strength
        confidence = 0.65 + 0.25 * abs(probability - 0.5) * 2

        return float(probability), float(confidence)

    def _score_ergodic(self, features: np.ndarray) -> tuple[float, float]:
        """
        Score ergodic features for AI likelihood.

        AI text shows faster mixing (shorter decorrelation time).

        Args:
            features: 24-dimensional Ergodic feature vector

        Returns:
            (probability, confidence) tuple
        """
        # Features breakdown:
        # 0-7: Autocorrelation decay
        # 8-15: Mixing time estimates
        # 16-23: Spectral analysis

        autocorr_features = features[:8]
        mixing_features = features[8:16]

        # Faster decay in autocorrelation suggests AI
        decay_rate = autocorr_features[5] if len(autocorr_features) > 5 else 0.5
        mixing_score = min(decay_rate, 1.0)

        # Higher entropy suggests AI (more uniform distribution)
        if len(mixing_features) > 1:
            entropy = mixing_features[1]
            entropy_score = min(entropy / 10.0, 1.0)  # Normalize
        else:
            entropy_score = 0.5

        probability = (mixing_score + entropy_score) / 2

        # Moderate confidence
        confidence = 0.6 + 0.3 * abs(probability - 0.5) * 2

        return float(probability), float(confidence)
