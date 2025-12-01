"""
VERITAS FastAPI Backend

REST API for AI text detection using the VERITAS ensemble classifier.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional
import time
from src.models.ensemble import VERITASClassifier

# Initialize FastAPI app
app = FastAPI(
    title="VERITAS API",
    description="Next-Generation AI Detection System",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize classifier (singleton)
classifier = VERITASClassifier()


class DetectionRequest(BaseModel):
    """Request model for text detection"""
    text: str = Field(..., max_length=100000, description="Text to analyze")


class DetectionResponse(BaseModel):
    """Response model for text detection"""
    classification_level: int = Field(..., description="1=Definitive, 2=Probabilistic, 3=Inconclusive")
    ai_probability: float = Field(..., description="Probability of AI authorship (0-1)")
    confidence: float = Field(..., description="Confidence in classification (0-1)")
    explanation: str = Field(..., description="Human-readable explanation")
    features: dict = Field(..., description="Detailed feature scores")
    word_count: int = Field(..., description="Number of words analyzed")
    module_scores: dict = Field(..., description="Individual module scores")
    processing_time_ms: float = Field(..., description="Analysis time in milliseconds")


@app.get("/")
async def root():
    """Serve the main web interface"""
    return FileResponse("static/index.html")


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "VERITAS"}


@app.post("/api/detect", response_model=DetectionResponse)
async def detect_ai_text(request: DetectionRequest):
    """
    Analyze text for AI detection.

    Performs comprehensive multi-modal analysis using:
    - Kolmogorov Complexity Differential Analysis (KCDA)
    - Topological Data Analysis (TDA)
    - Fractal Dimension Analysis
    - Ergodic Mixing Analysis

    Returns hierarchical classification with calibrated uncertainty.
    """
    start_time = time.time()

    try:
        # Perform analysis
        result = classifier.analyze_text(request.text)

        # Add processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        result['processing_time_ms'] = processing_time

        return DetectionResponse(**result)

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"[ERROR] Analysis failed:\n{error_details}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/api/stats")
async def get_stats():
    """Get system statistics"""
    return {
        "total_features": 168,
        "modules": {
            "kcda": {"features": 48, "description": "Kolmogorov Complexity Analysis"},
            "tda": {"features": 64, "description": "Topological Data Analysis"},
            "fractal": {"features": 32, "description": "Fractal Dimension Analysis"},
            "ergodic": {"features": 24, "description": "Ergodic Mixing Analysis"}
        },
        "performance_targets": {
            "latency_1000_words": "< 500ms",
            "latency_5000_words": "< 2000ms",
            "false_positive_rate": "< 5%",
            "true_positive_rate": "92%"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
