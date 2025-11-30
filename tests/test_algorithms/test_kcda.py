"""
Tests for Kolmogorov Complexity Differential Analysis (KCDA) module
"""
import pytest
import numpy as np
from src.algorithms.kcda import KCDAAnalyzer


class TestKCDAAnalyzer:
    """Test suite for KCDA analyzer"""

    @pytest.fixture
    def analyzer(self):
        """Create KCDA analyzer instance"""
        return KCDAAnalyzer()

    @pytest.fixture
    def sample_text(self):
        """Sample human-like text"""
        return """
        The quick brown fox jumps over the lazy dog. This sentence contains
        every letter of the alphabet, making it useful for testing purposes.
        Natural language has interesting patterns that emerge from human cognition,
        including varied sentence structures, creative word choices, and thematic
        coherence that spans multiple paragraphs.
        """

    @pytest.fixture
    def ai_like_text(self):
        """Sample AI-like text (more uniform, predictable)"""
        return """
        Natural language processing is a field of artificial intelligence. It focuses
        on the interaction between computers and human language. The goal is to enable
        computers to understand, interpret, and generate human language. This technology
        has many applications in modern software systems.
        """

    def test_analyzer_initialization(self, analyzer):
        """Test that analyzer initializes correctly"""
        assert analyzer is not None
        assert hasattr(analyzer, 'extract_features')
        assert hasattr(analyzer, 'compute_compression_ratios')

    def test_extract_features_returns_correct_dimension(self, analyzer, sample_text):
        """Test that feature extraction returns 48-dimensional vector"""
        features = analyzer.extract_features(sample_text)
        assert isinstance(features, np.ndarray)
        assert features.shape == (48,), f"Expected shape (48,), got {features.shape}"
        assert not np.isnan(features).any(), "Features should not contain NaN values"

    def test_compression_ratios_multiple_algorithms(self, analyzer, sample_text):
        """Test that compression analysis uses multiple algorithms"""
        ratios = analyzer.compute_compression_ratios(sample_text)
        # Should have compression ratios for multiple algorithms
        assert isinstance(ratios, dict)
        assert len(ratios) >= 3, "Should use at least 3 compression algorithms"
        # All ratios should be positive and less than or equal to 1
        for algo, ratio in ratios.items():
            assert 0 < ratio <= 1, f"{algo} ratio should be between 0 and 1, got {ratio}"

    def test_cross_complexity_computation(self, analyzer, sample_text):
        """Test cross-complexity K(x|y) computation"""
        segments = sample_text.split('.')[:3]  # Get first 3 sentences
        if len(segments) >= 2:
            cross_complexity = analyzer.compute_cross_complexity(segments[0], segments[1])
            assert isinstance(cross_complexity, float)
            assert cross_complexity >= 0

    def test_multi_scale_analysis(self, analyzer, sample_text):
        """Test that analysis is performed at multiple scales"""
        features = analyzer.extract_features(sample_text)
        # Multi-scale features should capture different granularities
        # Features should be diverse (not all the same)
        assert np.std(features) > 0, "Features should show variation across scales"

    def test_handles_short_text(self, analyzer):
        """Test that analyzer handles short text gracefully"""
        short_text = "Hello world."
        features = analyzer.extract_features(short_text)
        assert features.shape == (48,)
        # Should not crash, but may have limited information
        assert not np.isnan(features).any()

    def test_handles_empty_text(self, analyzer):
        """Test that analyzer handles empty text"""
        with pytest.raises(ValueError):
            analyzer.extract_features("")

    def test_feature_consistency(self, analyzer, sample_text):
        """Test that same text produces consistent features"""
        features1 = analyzer.extract_features(sample_text)
        features2 = analyzer.extract_features(sample_text)
        np.testing.assert_array_almost_equal(features1, features2, decimal=6)

    def test_different_texts_produce_different_features(self, analyzer, sample_text, ai_like_text):
        """Test that different texts produce different feature vectors"""
        features1 = analyzer.extract_features(sample_text)
        features2 = analyzer.extract_features(ai_like_text)
        # Features should be different
        assert not np.array_equal(features1, features2)
        # Cosine similarity should be less than 1 (not identical)
        similarity = np.dot(features1, features2) / (np.linalg.norm(features1) * np.linalg.norm(features2))
        assert similarity < 0.99
