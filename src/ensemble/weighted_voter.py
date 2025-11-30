"""
Weighted voting strategy for ensemble classification.

Combines scores from multiple analyzers using weighted averaging
with outlier detection and agreement computation.
"""

import numpy as np
from typing import Dict
from src.core.types import Score, EnsembleResult
from src.core.exceptions import ClassificationError


class WeightedVoter:
    """
    Weighted voter that combines scores using weighted averaging.

    Features:
    - Configurable weights per analyzer
    - Outlier detection using IQR method
    - Agreement computation (pairwise + similarity)
    - Robust to missing scores
    """

    def __init__(
        self,
        weights: Dict[str, float] = None,
        outlier_threshold: float = 1.5
    ):
        """
        Initialize weighted voter.

        Args:
            weights: Dictionary mapping analyzer names to weights.
                    If None, uses equal weights.
            outlier_threshold: IQR multiplier for outlier detection (default 1.5)
        """
        # Default weights based on theoretical strength
        self.weights = weights or {
            'kcda': 0.30,    # Strong theoretical foundation
            'tda': 0.25,     # Novel topological approach
            'fractal': 0.25, # Good at detecting uniformity
            'ergodic': 0.20  # Complementary statistical analysis
        }

        # Normalize weights to sum to 1.0
        total_weight = sum(self.weights.values())
        self.weights = {k: v / total_weight for k, v in self.weights.items()}

        self.outlier_threshold = outlier_threshold

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
        try:
            if not scores:
                raise ClassificationError("No scores provided for voting")

            # Extract probabilities
            probabilities = {name: score.probability for name, score in scores.items()}

            # Compute weighted average
            weighted_prob = self._compute_weighted_average(probabilities)

            # Detect outliers
            outliers = self._detect_outliers(probabilities)

            # Compute agreement
            agreement = self._compute_agreement(probabilities)

            return EnsembleResult(
                weighted_probability=weighted_prob,
                agreement_score=agreement,
                outlier_modules=outliers,
                individual_scores=scores,
                metadata={
                    'voting_method': 'weighted_average',
                    'weights_used': self.weights,
                    'num_voters': len(scores),
                    'num_outliers': len(outliers)
                }
            )

        except ClassificationError:
            raise
        except Exception as e:
            raise ClassificationError(
                f"Voting failed: {str(e)}",
                original_error=e
            )

    def _compute_weighted_average(self, probabilities: Dict[str, float]) -> float:
        """
        Compute weighted average of probabilities.

        Uses configured weights, falling back to equal weights for
        unknown analyzers.

        Args:
            probabilities: Dictionary mapping analyzer names to probabilities

        Returns:
            Weighted average probability
        """
        total_weight = 0.0
        weighted_sum = 0.0

        for name, prob in probabilities.items():
            weight = self.weights.get(name, 1.0 / len(probabilities))
            weighted_sum += prob * weight
            total_weight += weight

        if total_weight == 0:
            return 0.5  # Neutral if no weights

        return weighted_sum / total_weight

    def _detect_outliers(self, probabilities: Dict[str, float]) -> list[str]:
        """
        Detect outlier scores using IQR method.

        Args:
            probabilities: Dictionary mapping analyzer names to probabilities

        Returns:
            List of analyzer names that are statistical outliers
        """
        if len(probabilities) < 3:
            return []  # Need at least 3 scores for outlier detection

        scores_array = np.array(list(probabilities.values()))

        # Compute IQR
        q1 = np.percentile(scores_array, 25)
        q3 = np.percentile(scores_array, 75)
        iqr = q3 - q1

        # Compute bounds
        lower_bound = q1 - self.outlier_threshold * iqr
        upper_bound = q3 + self.outlier_threshold * iqr

        # Find outliers
        outliers = []
        for name, prob in probabilities.items():
            if prob < lower_bound or prob > upper_bound:
                outliers.append(name)

        return outliers

    def _compute_agreement(self, probabilities: Dict[str, float]) -> float:
        """
        Compute inter-module agreement.

        Combines two measures:
        1. Binary classification agreement (AI vs Human)
        2. Score similarity (low standard deviation)

        Args:
            probabilities: Dictionary mapping analyzer names to probabilities

        Returns:
            Agreement score between 0.0 and 1.0
        """
        if len(probabilities) < 2:
            return 1.0  # Perfect agreement if only one score

        scores_array = np.array(list(probabilities.values()))

        # 1. Binary classification agreement
        binary = (scores_array > 0.5).astype(int)
        n = len(binary)

        agreements = 0
        total_pairs = 0

        for i in range(n):
            for j in range(i + 1, n):
                if binary[i] == binary[j]:
                    agreements += 1
                total_pairs += 1

        binary_agreement = agreements / max(total_pairs, 1)

        # 2. Score similarity (inverse of std dev)
        score_std = np.std(scores_array)
        similarity_score = 1.0 / (1.0 + score_std)

        # Combine both measures (60% binary, 40% similarity)
        agreement = 0.6 * binary_agreement + 0.4 * similarity_score

        return float(agreement)
