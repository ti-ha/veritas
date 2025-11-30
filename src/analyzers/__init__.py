"""
Feature analyzers for VERITAS.

This module contains all feature extraction analyzers that implement
the AnalyzerProtocol interface.
"""

from src.analyzers.base import BaseAnalyzer

# Import new refactored analyzers
try:
    from src.analyzers.kcda import KCDAAnalyzer as NewKCDAAnalyzer
except ImportError:
    NewKCDAAnalyzer = None

try:
    from src.analyzers.tda import TDAAnalyzer as NewTDAAnalyzer
except ImportError:
    NewTDAAnalyzer = None

try:
    from src.analyzers.fractal import FractalAnalyzer as NewFractalAnalyzer
except ImportError:
    NewFractalAnalyzer = None

try:
    from src.analyzers.ergodic import ErgodicAnalyzer as NewErgodicAnalyzer
except ImportError:
    NewErgodicAnalyzer = None

# Re-export for backwards compatibility
# Use new implementation if available, otherwise fall back to old
try:
    if NewKCDAAnalyzer is not None:
        KCDAAnalyzer = NewKCDAAnalyzer
    else:
        from src.algorithms.kcda import KCDAAnalyzer
except ImportError:
    KCDAAnalyzer = None

try:
    if NewTDAAnalyzer is not None:
        TDAAnalyzer = NewTDAAnalyzer
    else:
        from src.algorithms.tda import TDAAnalyzer
except ImportError:
    TDAAnalyzer = None

try:
    if NewFractalAnalyzer is not None:
        FractalAnalyzer = NewFractalAnalyzer
    else:
        from src.algorithms.fractal import FractalAnalyzer
except ImportError:
    FractalAnalyzer = None

try:
    if NewErgodicAnalyzer is not None:
        ErgodicAnalyzer = NewErgodicAnalyzer
    else:
        from src.algorithms.ergodic import ErgodicAnalyzer
except ImportError:
    ErgodicAnalyzer = None

__all__ = [
    "BaseAnalyzer",
    "KCDAAnalyzer",
    "TDAAnalyzer",
    "FractalAnalyzer",
    "ErgodicAnalyzer",
]
