"""
Fractal Dimension Analysis of Textual Structure

Analyzes self-similar patterns in text through fractal geometry,
measuring how word distributions cluster at multiple scales.
"""

import numpy as np
from typing import List, Dict
from collections import Counter


class FractalAnalyzer:
    """
    Fractal dimension analyzer for AI text detection.

    Extracts 32-dimensional feature vectors capturing fractal
    properties of word distributions.
    """

    def __init__(self, random_seed: int = 42):
        """Initialize Fractal analyzer"""
        self.random_seed = random_seed
        np.random.seed(random_seed)

    def extract_features(self, text: str) -> np.ndarray:
        """
        Extract 32-dimensional fractal feature vector.

        Args:
            text: Input text

        Returns:
            32-dimensional numpy array
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        features = []

        # Get word positions
        words = text.lower().split()
        if len(words) < 10:
            return np.zeros(32)

        # 1. Box-counting dimensions for different word types (12 features)
        features.extend(self._compute_box_counting_features(words))

        # 2. Degree of fractality features (10 features)
        features.extend(self._compute_fractality_degree(words))

        # 3. Multi-scale fractal profile (10 features)
        features.extend(self._compute_multiscale_profile(words))

        return np.array(features[:32], dtype=np.float64)

    def _compute_box_counting_features(self, words: List[str]) -> List[float]:
        """Compute box-counting fractal dimensions"""
        features = []

        # Analyze different word classes
        word_classes = {
            'content': self._get_content_words(words),
            'function': self._get_function_words(words),
            'all': list(range(len(words)))
        }

        for class_name, positions in word_classes.items():
            if len(positions) < 2:
                features.extend([0.0, 0.0])
                continue

            # Compute fractal dimension
            dimension = self._box_counting_dimension(positions, len(words))
            features.append(dimension)

            # Compare to shuffled
            shuffled_positions = sorted(positions)
            shuffled_dim = self._box_counting_dimension(shuffled_positions, len(words))
            degree_of_fractality = abs(dimension - shuffled_dim)
            features.append(degree_of_fractality)

        # Padding
        while len(features) < 12:
            features.append(0.0)

        return features[:12]

    def _compute_fractality_degree(self, words: List[str]) -> List[float]:
        """Compute degree of fractality for important words"""
        features = []

        # Find frequent words (proxy for "important")
        word_counts = Counter(words)
        frequent_words = [word for word, count in word_counts.most_common(10) if count > 1]

        if not frequent_words:
            return [0.0] * 10

        dof_values = []
        for word in frequent_words[:5]:  # Top 5 words
            positions = [i for i, w in enumerate(words) if w == word]
            if len(positions) >= 2:
                original_dim = self._box_counting_dimension(positions, len(words))

                # Create uniform distribution
                uniform_positions = np.linspace(0, len(words) - 1, len(positions)).astype(int)
                uniform_dim = self._box_counting_dimension(uniform_positions.tolist(), len(words))

                dof = abs(original_dim - uniform_dim)
                dof_values.append(dof)

        if dof_values:
            features.append(np.mean(dof_values))
            features.append(np.std(dof_values))
            features.append(np.min(dof_values))
            features.append(np.max(dof_values))
            features.append(np.median(dof_values))
        else:
            features.extend([0.0] * 5)

        # Distribution statistics
        all_positions = []
        for word in frequent_words:
            all_positions.extend([i for i, w in enumerate(words) if w == word])

        if len(all_positions) >= 2:
            # Measure clustering
            positions_array = np.array(sorted(all_positions))
            gaps = np.diff(positions_array)
            features.append(np.mean(gaps))
            features.append(np.std(gaps))
            features.append(np.std(gaps) / max(np.mean(gaps), 1))  # Coefficient of variation
        else:
            features.extend([0.0] * 3)

        # Padding
        while len(features) < 10:
            features.append(0.0)

        return features[:10]

    def _compute_multiscale_profile(self, words: List[str]) -> List[float]:
        """Compute fractal profile at multiple scales"""
        features = []

        # Divide text into segments and analyze each
        segment_sizes = [len(words) // 4, len(words) // 2, len(words)]

        for seg_size in segment_sizes:
            if seg_size < 10:
                features.extend([0.0, 0.0])
                continue

            segment = words[:seg_size]

            # Compute Hurst exponent (measure of self-similarity)
            hurst = self._estimate_hurst_exponent(segment)
            features.append(hurst)

            # Compute entropy at this scale
            word_counts = Counter(segment)
            total = sum(word_counts.values())
            probs = [count / total for count in word_counts.values()]
            entropy = -sum(p * np.log(p + 1e-10) for p in probs)
            features.append(entropy)

        # Padding
        while len(features) < 10:
            features.append(0.0)

        return features[:10]

    def _box_counting_dimension(self, positions: List[int], total_length: int) -> float:
        """
        Compute fractal dimension using box-counting method.

        Args:
            positions: List of word positions
            total_length: Total text length

        Returns:
            Estimated fractal dimension
        """
        if len(positions) < 2:
            return 1.0

        positions = sorted(positions)

        # Try different box sizes
        box_sizes = []
        counts = []

        max_box_size = total_length // 2
        min_box_size = max(1, total_length // 20)

        for box_size in range(min_box_size, max_box_size + 1, max(1, (max_box_size - min_box_size) // 10)):
            # Count how many boxes contain at least one position
            boxes_with_points = set()
            for pos in positions:
                box_idx = pos // box_size
                boxes_with_points.add(box_idx)

            box_sizes.append(box_size)
            counts.append(len(boxes_with_points))

        if len(box_sizes) < 2:
            return 1.0

        # Fit line to log-log plot
        log_sizes = np.log(box_sizes)
        log_counts = np.log(np.array(counts) + 1)  # Add 1 to avoid log(0)

        # Linear regression
        if np.std(log_sizes) > 0:
            slope, _ = np.polyfit(log_sizes, log_counts, 1)
            dimension = -slope  # Fractal dimension is negative slope
            return max(0.0, min(dimension, 2.0))  # Clamp to [0, 2]

        return 1.0

    def _estimate_hurst_exponent(self, words: List[str]) -> float:
        """
        Estimate Hurst exponent using R/S analysis.

        Hurst exponent measures long-range dependence and self-similarity.
        H ~ 0.5: random walk
        H > 0.5: persistent (trends continue)
        H < 0.5: anti-persistent (mean-reverting)
        """
        if len(words) < 20:
            return 0.5

        # Create time series from word lengths
        series = np.array([len(word) for word in words])

        # Remove mean
        mean = np.mean(series)
        Y = np.cumsum(series - mean)

        # Calculate R/S for different lags
        lags = range(10, min(len(words) // 2, 100), 10)
        rs_values = []

        for lag in lags:
            # Split into blocks
            n_blocks = len(Y) // lag
            if n_blocks == 0:
                continue

            rs_block = []
            for i in range(n_blocks):
                block = Y[i * lag:(i + 1) * lag]
                R = np.max(block) - np.min(block)  # Range
                S = np.std(series[i * lag:(i + 1) * lag]) + 1e-10  # Standard deviation
                rs_block.append(R / S)

            if rs_block:
                rs_values.append(np.mean(rs_block))

        if len(rs_values) < 2:
            return 0.5

        # Fit line to log-log plot
        log_lags = np.log(list(lags[:len(rs_values)]))
        log_rs = np.log(rs_values)

        if np.std(log_lags) > 0:
            hurst, _ = np.polyfit(log_lags, log_rs, 1)
            return max(0.0, min(hurst, 1.0))  # Clamp to [0, 1]

        return 0.5

    def _get_content_words(self, words: List[str]) -> List[int]:
        """Get positions of content words (non-function words)"""
        function_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
                             'to', 'for', 'of', 'with', 'by', 'from', 'is', 'was',
                             'are', 'were', 'been', 'be', 'have', 'has', 'had', 'do',
                             'does', 'did', 'will', 'would', 'should', 'could', 'may',
                             'might', 'can', 'this', 'that', 'these', 'those', 'it',
                             'its', 'he', 'she', 'they', 'we', 'you', 'i', 'me', 'my'])

        return [i for i, word in enumerate(words) if word.lower() not in function_words]

    def _get_function_words(self, words: List[str]) -> List[int]:
        """Get positions of function words"""
        function_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
                             'to', 'for', 'of', 'with', 'by', 'from', 'is', 'was',
                             'are', 'were', 'been', 'be'])

        return [i for i, word in enumerate(words) if word.lower() in function_words]
