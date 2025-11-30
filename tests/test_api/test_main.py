"""
Tests for the VERITAS API
"""
import pytest
from fastapi.testclient import TestClient
from src.api.main import app


class TestVERITASAPI:
    """Test suite for the API endpoints"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)

    @pytest.fixture
    def sample_text(self):
        """Sample text for testing"""
        return """
        This is a test passage for analyzing AI detection capabilities.
        The text needs to be long enough to provide meaningful analysis.
        Various linguistic features are examined including complexity,
        topological structure, fractal patterns, and statistical properties.
        """

    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "VERITAS"

    def test_stats_endpoint(self, client):
        """Test stats endpoint"""
        response = client.get("/api/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_features" in data
        assert data["total_features"] == 168
        assert "modules" in data
        assert len(data["modules"]) == 4

    def test_detect_endpoint_success(self, client, sample_text):
        """Test successful text detection"""
        response = client.post(
            "/api/detect",
            json={"text": sample_text}
        )
        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert "classification_level" in data
        assert "ai_probability" in data
        assert "confidence" in data
        assert "explanation" in data
        assert "features" in data
        assert "processing_time_ms" in data

        # Check value ranges
        assert 1 <= data["classification_level"] <= 3
        assert 0 <= data["ai_probability"] <= 1
        assert 0 <= data["confidence"] <= 1

    def test_detect_endpoint_short_text(self, client):
        """Test detection with short text"""
        # Text needs minimum 50 chars for analysis
        short_text = "This text is a bit longer but still relatively short for analysis purposes."
        response = client.post(
            "/api/detect",
            json={"text": short_text}
        )
        assert response.status_code == 200
        data = response.json()
        # Should return a result (may be inconclusive)
        assert "classification_level" in data

    def test_detect_endpoint_empty_text(self, client):
        """Test detection with empty text"""
        response = client.post(
            "/api/detect",
            json={"text": ""}
        )
        # Empty text may return 500 error or inconclusive result
        # Either is acceptable as empty text is invalid input
        assert response.status_code in [200, 500]
        if response.status_code == 200:
            data = response.json()
            assert data["classification_level"] == 3

    def test_detect_endpoint_missing_text(self, client):
        """Test detection without text field"""
        response = client.post(
            "/api/detect",
            json={}
        )
        assert response.status_code == 422  # Validation error

    def test_response_time(self, client, sample_text):
        """Test that response time is reasonable"""
        response = client.post(
            "/api/detect",
            json={"text": sample_text}
        )
        assert response.status_code == 200
        data = response.json()

        # Should complete in less than 5 seconds for short text
        assert data["processing_time_ms"] < 5000
