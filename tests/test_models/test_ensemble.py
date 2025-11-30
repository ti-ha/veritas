"""
Tests for the VERITAS Ensemble Classifier
"""
import pytest
from src.models.ensemble import VERITASClassifier


class TestVERITASClassifier:
    """Test suite for ensemble classifier"""

    @pytest.fixture
    def classifier(self):
        """Create classifier instance"""
        return VERITASClassifier()

    @pytest.fixture
    def human_text(self):
        """Sample human-written text"""
        return """
        I've been thinking about this problem for weeks now, and honestly, I'm still not
        entirely sure what the right answer is. My initial intuition was to approach it
        from a different angle - maybe looking at the historical context first? But then
        again, that might miss some of the more subtle contemporary implications. You know
        what I mean? It's like when you're trying to solve a puzzle and you keep rotating
        it in your mind, hoping that sudden flash of insight will hit you. Sometimes it
        does, sometimes it doesn't. Life's funny that way.
        """

    @pytest.fixture
    def ai_text(self):
        """Sample AI-generated text"""
        return """
        Artificial intelligence has transformed numerous industries in recent years.
        Machine learning algorithms can now process vast amounts of data efficiently.
        Natural language processing enables computers to understand human communication.
        These technologies have applications in healthcare, finance, and education.
        The development of neural networks has been particularly significant. Deep learning
        models can recognize patterns in complex datasets. This capability has led to
        breakthroughs in image recognition and language translation. Organizations worldwide
        are adopting these solutions to improve their operations.
        """

    def test_classifier_initialization(self, classifier):
        """Test classifier initializes properly"""
        assert classifier is not None
        assert hasattr(classifier, 'analyze_text')
        assert hasattr(classifier, 'kcda')
        assert hasattr(classifier, 'tda')
        assert hasattr(classifier, 'fractal')
        assert hasattr(classifier, 'ergodic')

    def test_analyze_text_returns_correct_structure(self, classifier, human_text):
        """Test that analysis returns expected structure"""
        result = classifier.analyze_text(human_text)

        assert isinstance(result, dict)
        assert 'classification_level' in result
        assert 'ai_probability' in result
        assert 'confidence' in result
        assert 'explanation' in result
        assert 'features' in result

        # Check classification level is valid
        assert result['classification_level'] in [1, 2, 3]

        # Check probability is in valid range
        assert 0 <= result['ai_probability'] <= 1

        # Check confidence is in valid range
        assert 0 <= result['confidence'] <= 1

    def test_short_text_handling(self, classifier):
        """Test handling of very short text"""
        short_text = "Hello world."
        result = classifier.analyze_text(short_text)

        assert result['classification_level'] == 3
        assert 'too short' in result['explanation'].lower()

    def test_empty_text_handling(self, classifier):
        """Test handling of empty text"""
        result = classifier.analyze_text("")
        assert result['classification_level'] == 3

    def test_feature_extraction(self, classifier, human_text):
        """Test that features are extracted from all modules"""
        result = classifier.analyze_text(human_text)

        assert 'features' in result
        assert 'kcda_score' in result['features']
        assert 'tda_score' in result['features']
        assert 'fractal_score' in result['features']
        assert 'ergodic_score' in result['features']
        assert 'feature_vector_size' in result['features']

        # Should be 48 + 64 + 32 + 24 = 168 dimensions
        assert result['features']['feature_vector_size'] == 168

    def test_different_texts_produce_different_results(self, classifier, human_text, ai_text):
        """Test that different texts produce different analysis results"""
        result_human = classifier.analyze_text(human_text)
        result_ai = classifier.analyze_text(ai_text)

        # Results should be different
        assert result_human['ai_probability'] != result_ai['ai_probability']

    def test_consistency(self, classifier, human_text):
        """Test that same text produces consistent results"""
        result1 = classifier.analyze_text(human_text)
        result2 = classifier.analyze_text(human_text)

        assert result1['classification_level'] == result2['classification_level']
        assert abs(result1['ai_probability'] - result2['ai_probability']) < 0.01
        assert result1['explanation'] == result2['explanation']

    def test_module_scores_in_valid_range(self, classifier, human_text):
        """Test that all module scores are in [0, 1] range"""
        result = classifier.analyze_text(human_text)

        assert 0 <= result['features']['kcda_score'] <= 1
        assert 0 <= result['features']['tda_score'] <= 1
        assert 0 <= result['features']['fractal_score'] <= 1
        assert 0 <= result['features']['ergodic_score'] <= 1
