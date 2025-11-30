# Veritas

An advanced AI text detection system using multi-modal algorithmic analysis.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Start API server
uvicorn src.api.main:app --reload

# Access web interface
http://localhost:8000
```

---

## Overview

VERITAS combines multiple algorithmic approaches to detect AI-generated text through feature analysis rather than simple pattern matching.

### Quick Links

| Need | Document |
|------|----------|
| Run the system | [Quick Start](#-quick-start) |
| Run tests | [Testing](#-testing) |
| API usage | [API Example](#-api-example) |
| Project structure | [Architecture](#️-architecture) |

---

## Technology

Multi-modal ensemble combining:
- **KCDA** (Kolmogorov Complexity) - 48 features
- **TDA** (Topological Analysis) - 64 features
- **Fractal Analysis** - 32 features
- **Ergodic Analysis** - 24 features

**Total**: 168-dimensional feature vector per text sample

---

## Architecture

```
Text → [KCDA, TDA, Fractal, Ergodic] → Weighted Ensemble → Classification
         48    64     32      24          (outlier detection,
                                            agreement scoring)
```

---

## Testing

```bash
# All tests
pytest tests/ -v

# Validation tests with real data
pytest tests/test_validation.py -v -s

# (AI TPR test fails - needs ML training)
```

---

## Project Structure

```
Veritas/
├── src/
│   ├── algorithms/      # KCDA, TDA, Fractal, Ergodic
│   ├── models/          # Ensemble classifier
│   └── api/             # FastAPI backend
├── tests/               # 30 comprehensive tests
├── data/                # Training and test data
├── static/              # Web interface
└── scripts/             # Data collection and training tools
```

---


## Key Features

1. **Multi-Modal Analysis**: Combines 4 different algorithmic approaches
2. **168-Dimensional Feature Space**: Comprehensive text characterization
3. **Ensemble Classification**: Weighted voting system for robust detection
4. **Modular Architecture**: Easy to extend with additional algorithms

---

## Resources

- **API Documentation**: Start server and visit http://localhost:8000/docs
- **Interactive Testing**: Web interface at http://localhost:8000
- **Test Suite**: Run `pytest tests/ -v` for comprehensive validation

---

## API Example

```python
import requests

response = requests.post(
    'http://localhost:8000/detect',
    json={'text': 'Your text to analyze here'}
)

result = response.json()
print(f"AI Probability: {result['ai_probability']:.2%}")
print(f"Classification: Level {result['classification_level']}")
```

---

## License

CC 1.0 Universal License - See LICENSE file for details.

---
