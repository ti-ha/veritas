"""
Ergodic Mixing Analysis

Analyzes the statistical behavior of text generation as a dynamical system,
measuring mixing properties and long-range correlations.
"""

import numpy as np
from typing import List
from collections import Counter
from src.analyzers.base import BaseAnalyzer


class ErgodicAnalyzer(BaseAnalyzer):
    """
    Ergodic mixing analyzer for AI text detection.

    Extracts 24-dimensional feature vectors capturing mixing time,
    autocorrelation decay, and spectral properties.
    """

    def __init__(self, random_seed: int = 42):
        """Initialize Ergodic analyzer"""
        super().__init__(random_seed)
        self._name = "ergodic"
        self._feature_count = 24
        np.random.seed(random_seed)

    def _extract_features_impl(self, text: str) -> np.ndarray:
        """
        Extract 24-dimensional ergodic feature vector.

        Args:
            text: Input text

        Returns:
            24-dimensional numpy array
        """
        features = []

        words = text.lower().split()
        if len(words) < 10:
            return np.zeros(24)

        # 1. Autocorrelation decay features (8 features)
        features.extend(self._compute_autocorrelation_features(words))

        # 2. Mixing time estimates (8 features)
        features.extend(self._compute_mixing_features(words))

        # 3. Spectral analysis features (8 features)
        features.extend(self._compute_spectral_features(words))

        return np.array(features[:24], dtype=np.float64)

    def _compute_autocorrelation_features(self, words: List[str]) -> List[float]:
        """Compute autocorrelation function and decay characteristics"""
        features = []

        # Create numeric representation (word length sequence)
        word_lengths = np.array([len(word) for word in words])

        # Normalize
        word_lengths = (word_lengths - np.mean(word_lengths)) / (np.std(word_lengths) + 1e-10)

        # Compute autocorrelation at different lags
        max_lag = min(50, len(words) // 4)
        lags = [1, 5, 10, 20, max_lag]
        autocorrs = []

        for lag in lags:
            if lag >= len(words):
                autocorrs.append(0.0)
                continue

            corr = np.corrcoef(word_lengths[:-lag], word_lengths[lag:])[0, 1]
            if np.isnan(corr):
                corr = 0.0
            autocorrs.append(corr)

        features.extend(autocorrs)

        # Decay rate (fit exponential)
        if len(autocorrs) > 2:
            valid_autocorrs = [max(a, 1e-10) for a in autocorrs if a > 0]
            if len(valid_autocorrs) >= 2:
                log_autocorrs = np.log(valid_autocorrs)
                decay_rate, _ = np.polyfit(range(len(log_autocorrs)), log_autocorrs, 1)
                features.append(abs(decay_rate))
            else:
                features.append(0.0)
        else:
            features.append(0.0)

        # Time to decorrelation (when autocorr drops below threshold)
        threshold = 0.1
        decorr_time = max_lag
        for i, corr in enumerate(autocorrs):
            if abs(corr) < threshold:
                decorr_time = lags[i]
                break
        features.append(decorr_time / max(max_lag, 1))

        # Oscillation detection (negative correlations)
        num_negative = sum(1 for c in autocorrs if c < 0)
        features.append(num_negative / len(autocorrs))

        # Padding
        while len(features) < 8:
            features.append(0.0)

        return features[:8]

    def _compute_mixing_features(self, words: List[str]) -> List[float]:
        """Compute mixing time estimates and related features"""
        features = []

        # Analyze word transition statistics
        if len(words) < 2:
            return [0.0] * 8

        # Build bigram distribution
        bigrams = [(words[i], words[i + 1]) for i in range(len(words) - 1)]
        bigram_counts = Counter(bigrams)
        total_bigrams = len(bigrams)

        # Compute bigram entropy
        bigram_probs = [count / total_bigrams for count in bigram_counts.values()]
        bigram_entropy = -sum(p * np.log(p + 1e-10) for p in bigram_probs)
        features.append(bigram_entropy)

        # Compare to unigram entropy
        word_counts = Counter(words)
        total_words = len(words)
        word_probs = [count / total_words for count in word_counts.values()]
        unigram_entropy = -sum(p * np.log(p + 1e-10) for p in word_probs)
        features.append(unigram_entropy)

        # Mutual information (how much bigrams reduce uncertainty)
        mutual_info = unigram_entropy - (bigram_entropy / 2)
        features.append(mutual_info)

        # Analyze trigrams for longer-range dependencies
        if len(words) >= 3:
            trigrams = [(words[i], words[i + 1], words[i + 2]) for i in range(len(words) - 2)]
            trigram_counts = Counter(trigrams)
            total_trigrams = len(trigrams)

            trigram_probs = [count / total_trigrams for count in trigram_counts.values()]
            trigram_entropy = -sum(p * np.log(p + 1e-10) for p in trigram_probs)
            features.append(trigram_entropy)

            # Compare trigram vs bigram (diminishing returns suggests mixing)
            entropy_reduction = bigram_entropy - (trigram_entropy / 3)
            features.append(entropy_reduction)
        else:
            features.extend([0.0, 0.0])

        # Measure repetition patterns (non-mixing systems repeat more)
        unique_ratio = len(set(words)) / len(words)
        features.append(unique_ratio)

        # Measure word position dependency
        # Sample same words at different positions
        word_position_variance = self._compute_position_variance(words)
        features.append(word_position_variance)

        # Markov property test
        markov_score = self._test_markov_property(words)
        features.append(markov_score)

        # Padding
        while len(features) < 8:
            features.append(0.0)

        return features[:8]

    def _compute_spectral_features(self, words: List[str]) -> List[float]:
        """Compute spectral analysis features"""
        features = []

        # Create time series from word properties
        word_lengths = np.array([len(word) for word in words])

        if len(word_lengths) < 16:
            return [0.0] * 8

        # Compute power spectrum using FFT
        fft = np.fft.fft(word_lengths - np.mean(word_lengths))
        power_spectrum = np.abs(fft) ** 2

        # Use only positive frequencies
        n = len(power_spectrum) // 2
        power_spectrum = power_spectrum[:n]

        if len(power_spectrum) == 0:
            return [0.0] * 8

        # Normalize
        power_spectrum = power_spectrum / (np.sum(power_spectrum) + 1e-10)

        # Spectral features
        features.append(np.max(power_spectrum))  # Peak power
        features.append(np.argmax(power_spectrum) / len(power_spectrum))  # Peak frequency

        # Spectral entropy
        spec_entropy = -np.sum(power_spectrum * np.log(power_spectrum + 1e-10))
        features.append(spec_entropy)

        # Spectral centroid
        freqs = np.arange(len(power_spectrum))
        centroid = np.sum(freqs * power_spectrum) / np.sum(power_spectrum)
        features.append(centroid / len(power_spectrum))

        # Spectral spread
        spread = np.sqrt(np.sum(((freqs - centroid) ** 2) * power_spectrum) / np.sum(power_spectrum))
        features.append(spread / len(power_spectrum))

        # 1/f noise detection (pink noise characteristic of human writing)
        # Fit power law: P(f) ~ 1/f^α
        log_freqs = np.log(freqs[1:10] + 1)  # Low frequencies, avoid f=0
        log_power = np.log(power_spectrum[1:10] + 1e-10)

        if np.std(log_freqs) > 0:
            alpha, _ = np.polyfit(log_freqs, log_power, 1)
            features.append(-alpha)  # Negative slope indicates 1/f^α
        else:
            features.append(0.0)

        # Spectral flatness (white noise = 1, tonal = 0)
        geometric_mean = np.exp(np.mean(np.log(power_spectrum + 1e-10)))
        arithmetic_mean = np.mean(power_spectrum)
        flatness = geometric_mean / (arithmetic_mean + 1e-10)
        features.append(flatness)

        # High frequency energy
        high_freq_energy = np.sum(power_spectrum[n // 2:]) / np.sum(power_spectrum)
        features.append(high_freq_energy)

        # Padding
        while len(features) < 8:
            features.append(0.0)

        return features[:8]

    def _compute_position_variance(self, words: List[str]) -> float:
        """Measure how much word usage varies by position"""
        word_counts = Counter(words)

        # Find words that appear multiple times
        repeated_words = [word for word, count in word_counts.items() if count >= 3]

        if not repeated_words:
            return 0.0

        variances = []
        for word in repeated_words[:10]:  # Sample top 10
            positions = [i for i, w in enumerate(words) if w == word]
            # Normalize positions to [0, 1]
            norm_positions = np.array(positions) / len(words)
            variance = np.var(norm_positions)
            variances.append(variance)

        return np.mean(variances) if variances else 0.0

    def _test_markov_property(self, words: List[str]) -> float:
        """
        Test if word sequence satisfies Markov property.

        Returns score indicating how well bigram model predicts trigrams.
        High score = more Markovian (AI-like)
        """
        if len(words) < 3:
            return 0.0

        # Build conditional probability tables
        bigram_counts = Counter()
        trigram_counts = Counter()

        for i in range(len(words) - 1):
            bigram_counts[(words[i], words[i + 1])] += 1

        for i in range(len(words) - 2):
            trigram_counts[(words[i], words[i + 1], words[i + 2])] += 1

        # For each trigram, compare to Markov prediction
        markov_scores = []
        for (w1, w2, w3), count in trigram_counts.items():
            # Markov: P(w3|w1,w2) ≈ P(w3|w2)
            p_trigram = count / sum(trigram_counts.values())
            p_bigram = bigram_counts[(w2, w3)] / sum(bigram_counts.values())

            if p_bigram > 0:
                ratio = p_trigram / p_bigram
                markov_scores.append(min(ratio, 2.0))  # Cap to avoid outliers

        return np.mean(markov_scores) if markov_scores else 0.5
