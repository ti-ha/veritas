"""
Kolmogorov Complexity Differential Analysis (KCDA)

This module implements approximations of Kolmogorov complexity through
compression-based methods, analyzing text at multiple scales to detect
structural signatures of AI-generated content.

Based on algorithmic information theory and Normalized Compression Distance (NCD),
this approach measures the computational resources needed to specify text, revealing
patterns characteristic of autoregressive generation.

Key improvements:
- Proper NCD multiset formula: NCD₁(X) = [G(X) - min{G(x)}] / max{G(X\{x})}
- Conditional complexity: C(x|y) = C(xy) - C(y)
- Enhanced cross-complexity with multiple compressors
- Statistical validation of compression behavior
"""

import zlib
import bz2
import lzma
import numpy as np
from typing import Dict, List
from src.analyzers.base import BaseAnalyzer


class KCDAAnalyzer(BaseAnalyzer):
    """
    Kolmogorov Complexity Differential Analysis for AI text detection.

    Implements multi-scale compression analysis using various algorithms
    to extract 48-dimensional feature vectors capturing information-theoretic
    properties of text.
    """

    def __init__(self, random_seed: int = 42):
        """Initialize the KCDA analyzer with compression algorithms"""
        super().__init__(random_seed)
        self._name = "kcda"
        self._feature_count = 48

        self.compression_algorithms = {
            'zlib': self._compress_zlib,
            'bz2': self._compress_bz2,
            'lzma': self._compress_lzma,
        }
        self.window_sizes = ['sentence', 'paragraph', 'full']
        np.random.seed(random_seed)

    def _extract_features_impl(self, text: str) -> np.ndarray:
        """
        Extract 48-dimensional feature vector from text.

        Features include:
        - Compression ratios across algorithms and scales (9 features)
        - Cross-complexity statistics (12 features)
        - Relative complexity divergence measures (15 features)
        - Multi-scale compression behavior (12 features)

        Args:
            text: Input text to analyze

        Returns:
            48-dimensional numpy array of features
        """
        features = []

        # 1. Multi-scale compression ratios (3 algorithms × 3 scales = 9 features)
        compression_features = self._compute_multiscale_compression(text)
        features.extend(compression_features)

        # 2. Cross-complexity features (12 features)
        cross_complexity_features = self._compute_cross_complexity_features(text)
        features.extend(cross_complexity_features)

        # 3. Relative complexity divergence (15 features)
        divergence_features = self._compute_divergence_features(text)
        features.extend(divergence_features)

        # 4. Compression behavior statistics (12 features)
        behavior_features = self._compute_behavior_features(text)
        features.extend(behavior_features)

        return np.array(features, dtype=np.float64)

    def compute_compression_ratios(self, text: str) -> Dict[str, float]:
        """
        Compute compression ratios using multiple algorithms.

        Args:
            text: Input text

        Returns:
            Dictionary mapping algorithm name to compression ratio
        """
        ratios = {}
        text_bytes = text.encode('utf-8')
        original_size = len(text_bytes)

        if original_size == 0:
            return {algo: 1.0 for algo in self.compression_algorithms}

        for algo_name, compress_func in self.compression_algorithms.items():
            compressed = compress_func(text_bytes)
            ratio = len(compressed) / original_size
            ratios[algo_name] = min(ratio, 1.0)  # Cap at 1.0

        return ratios

    def compute_ncd(self, x: str, y: str, compressor_name: str = 'zlib') -> float:
        """
        Proper Normalized Compression Distance (NCD) between two strings.

        NCD(x,y) = (C(xy) - min(C(x), C(y))) / max(C(x), C(y))

        where C(x) is the compressed size of x.

        Args:
            x: First string
            y: Second string
            compressor_name: Compression algorithm to use

        Returns:
            NCD value between 0 and 1
        """
        if not x or not y:
            return 1.0

        compressor = self.compression_algorithms.get(compressor_name, self._compress_zlib)

        # Compress individual strings
        C_x = len(compressor(x.encode('utf-8')))
        C_y = len(compressor(y.encode('utf-8')))

        # Compress concatenation
        C_xy = len(compressor((x + y).encode('utf-8')))

        # NCD formula
        ncd = (C_xy - min(C_x, C_y)) / max(C_x, C_y, 1)

        return min(max(ncd, 0.0), 1.0)  # Clamp to [0, 1]

    def compute_ncd_multiset(self, texts: List[str], compressor_name: str = 'zlib') -> float:
        """
        NCD for multisets - more accurate for multiple text segments.

        NCD₁(X) = [G(X) - min{G(x)}] / max{G(X\{x})}

        Args:
            texts: List of text segments
            compressor_name: Compression algorithm to use

        Returns:
            Multiset NCD value
        """
        if not texts or len(texts) < 2:
            return 0.0

        compressor = self.compression_algorithms.get(compressor_name, self._compress_zlib)

        # G(X): Compress all texts concatenated
        combined = ''.join(texts)
        G_X = len(compressor(combined.encode('utf-8')))

        # min{G(x)}: Minimum individual compression
        min_G_x = min(len(compressor(t.encode('utf-8'))) for t in texts if t)

        # max{G(X\{x})}: Maximum compression excluding each element
        max_without = 0
        for i in range(len(texts)):
            subset = texts[:i] + texts[i+1:]
            if subset:
                subset_combined = ''.join(subset)
                G_subset = len(compressor(subset_combined.encode('utf-8')))
                max_without = max(max_without, G_subset)

        if max_without == 0:
            return 0.0

        ncd_multiset = (G_X - min_G_x) / max_without

        return min(max(ncd_multiset, 0.0), 1.0)

    def compute_conditional_complexity(self, x: str, y: str, compressor_name: str = 'zlib') -> float:
        """
        Conditional complexity C(x|y) = C(xy) - C(y)

        Measures how much information y provides about x.

        Args:
            x: String to compress
            y: Context/condition string
            compressor_name: Compression algorithm to use

        Returns:
            Conditional complexity (normalized by C(x))
        """
        if not x or not y:
            return 1.0

        compressor = self.compression_algorithms.get(compressor_name, self._compress_zlib)

        C_x = len(compressor(x.encode('utf-8')))
        C_y = len(compressor(y.encode('utf-8')))
        C_xy = len(compressor((x + y).encode('utf-8')))

        # Conditional complexity
        conditional = C_xy - C_y

        # Normalize by C(x) to get relative benefit
        return conditional / max(C_x, 1)

    def compute_cross_complexity(self, segment_x: str, segment_y: str) -> float:
        """
        Compute cross-complexity K(x|y) - how much knowing y helps compress x.

        This is approximated as: K(x|y) ≈ K(xy) - K(y)
        where K is estimated via compression.

        Args:
            segment_x: First text segment
            segment_y: Second text segment

        Returns:
            Cross-complexity measure
        """
        # Use improved conditional complexity method
        return self.compute_conditional_complexity(segment_x, segment_y, 'zlib')

    def _compress_zlib(self, data: bytes) -> bytes:
        """Compress using zlib (LZ77-based)"""
        return zlib.compress(data, level=9)

    def _compress_bz2(self, data: bytes) -> bytes:
        """Compress using bz2 (BWT-based)"""
        return bz2.compress(data, compresslevel=9)

    def _compress_lzma(self, data: bytes) -> bytes:
        """Compress using LZMA"""
        return lzma.compress(data, preset=9)

    def _compute_multiscale_compression(self, text: str) -> List[float]:
        """Compute compression ratios at multiple scales"""
        features = []

        # Split text into different granularities
        sentences = self._split_sentences(text)
        paragraphs = self._split_paragraphs(text)

        # Compute compression ratios at different scales
        for algo_name, compress_func in self.compression_algorithms.items():
            # Sentence-level average
            if sentences:
                sentence_ratios = []
                for sent in sentences:
                    if sent.strip():
                        sent_bytes = sent.encode('utf-8')
                        compressed = compress_func(sent_bytes)
                        ratio = len(compressed) / max(len(sent_bytes), 1)
                        sentence_ratios.append(ratio)
                features.append(np.mean(sentence_ratios) if sentence_ratios else 1.0)
            else:
                features.append(1.0)

            # Paragraph-level average
            if paragraphs:
                para_ratios = []
                for para in paragraphs:
                    if para.strip():
                        para_bytes = para.encode('utf-8')
                        compressed = compress_func(para_bytes)
                        ratio = len(compressed) / max(len(para_bytes), 1)
                        para_ratios.append(ratio)
                features.append(np.mean(para_ratios) if para_ratios else 1.0)
            else:
                features.append(1.0)

            # Full document
            full_bytes = text.encode('utf-8')
            compressed = compress_func(full_bytes)
            ratio = len(compressed) / max(len(full_bytes), 1)
            features.append(ratio)

        return features

    def _compute_cross_complexity_features(self, text: str) -> List[float]:
        """Compute cross-complexity statistics between text segments using improved NCD"""
        features = []
        sentences = self._split_sentences(text)

        if len(sentences) < 2:
            return [0.0] * 12

        # 1. Pairwise NCD statistics (using proper NCD formula)
        ncd_values = []
        for i in range(min(5, len(sentences) - 1)):  # Sample first few pairs
            ncd = self.compute_ncd(sentences[i], sentences[i + 1], 'zlib')
            ncd_values.append(ncd)

        if ncd_values:
            features.extend([
                np.mean(ncd_values),
                np.std(ncd_values),
                np.min(ncd_values),
                np.max(ncd_values),
            ])
        else:
            features.extend([0.0] * 4)

        # 2. Multiset NCD for first few sentences
        if len(sentences) >= 3:
            multiset_ncd = self.compute_ncd_multiset(sentences[:5], 'zlib')
            features.append(multiset_ncd)
        else:
            features.append(0.0)

        # 3. Conditional complexity statistics
        conditional_complexities = []
        for i in range(min(4, len(sentences) - 1)):
            cc = self.compute_conditional_complexity(sentences[i], sentences[i + 1], 'zlib')
            conditional_complexities.append(cc)

        if conditional_complexities:
            features.extend([
                np.mean(conditional_complexities),
                np.std(conditional_complexities),
            ])
        else:
            features.extend([0.0] * 2)

        # 4. Asymmetry in conditional complexity (forward vs backward)
        if len(sentences) >= 3:
            forward_cc = self.compute_conditional_complexity(sentences[0], sentences[1], 'zlib')
            backward_cc = self.compute_conditional_complexity(sentences[1], sentences[0], 'zlib')
            asymmetry = abs(forward_cc - backward_cc)
            features.append(asymmetry)
        else:
            features.append(0.0)

        # 5. NCD with multiple compressors (diversity measure)
        if len(sentences) >= 2:
            ncd_zlib = self.compute_ncd(sentences[0], sentences[1], 'zlib')
            ncd_bz2 = self.compute_ncd(sentences[0], sentences[1], 'bz2')
            ncd_lzma = self.compute_ncd(sentences[0], sentences[1], 'lzma')
            compressor_agreement = np.std([ncd_zlib, ncd_bz2, ncd_lzma])
            features.append(compressor_agreement)
        else:
            features.append(0.0)

        # 6. Distant segment NCD (beginning vs end)
        if len(sentences) >= 4:
            distant_ncd = self.compute_ncd(sentences[0], sentences[-1], 'zlib')
            features.append(distant_ncd)
        else:
            features.append(0.0)

        # 7. Average pairwise NCD across all segments (for longer texts)
        if len(sentences) >= 4:
            all_pairs_ncd = []
            for i in range(min(3, len(sentences))):
                for j in range(i + 1, min(i + 3, len(sentences))):
                    ncd = self.compute_ncd(sentences[i], sentences[j], 'zlib')
                    all_pairs_ncd.append(ncd)
            features.append(np.mean(all_pairs_ncd) if all_pairs_ncd else 0.0)
        else:
            features.append(0.0)

        # Ensure exactly 12 features
        while len(features) < 12:
            features.append(0.0)

        return features[:12]

    def _compute_divergence_features(self, text: str) -> List[float]:
        """Compute relative complexity divergence measures using NCD"""
        features = []

        # Generate shuffled version (breaks structure but preserves vocabulary)
        shuffled_text = self._shuffle_text(text)

        # 1. NCD between original and shuffled (structural similarity)
        # Low NCD means high similarity despite shuffle = less structure
        # High NCD means low similarity after shuffle = more structure
        ncd_original_shuffled = self.compute_ncd(text, shuffled_text, 'zlib')
        features.append(ncd_original_shuffled)

        # 2. NCD with all three compressors
        ncd_bz2 = self.compute_ncd(text, shuffled_text, 'bz2')
        ncd_lzma = self.compute_ncd(text, shuffled_text, 'lzma')
        features.extend([ncd_bz2, ncd_lzma])

        # 3. Compressor agreement on structure
        compressor_std = np.std([ncd_original_shuffled, ncd_bz2, ncd_lzma])
        features.append(compressor_std)

        # 4. Traditional compression ratio divergence
        ratios = self.compute_compression_ratios(text)
        shuffled_ratios = self.compute_compression_ratios(shuffled_text)

        for algo in ['zlib', 'bz2', 'lzma']:
            divergence = shuffled_ratios[algo] - ratios[algo]  # Positive = original more compressible
            features.append(divergence)

        # 5. Sentence-level complexity profile
        sentences = self._split_sentences(text)
        if len(sentences) >= 3:
            sentence_complexities = []
            for sent in sentences[:10]:  # Sample first 10 sentences
                if sent.strip():
                    ratios_sent = self.compute_compression_ratios(sent)
                    avg_ratio = np.mean(list(ratios_sent.values()))
                    sentence_complexities.append(avg_ratio)

            if sentence_complexities:
                features.extend([
                    np.mean(sentence_complexities),
                    np.std(sentence_complexities),
                    np.max(sentence_complexities) - np.min(sentence_complexities),  # Range
                ])
            else:
                features.extend([0.0] * 3)
        else:
            features.extend([0.0] * 3)

        # Ensure exactly 15 features
        while len(features) < 15:
            features.append(0.0)

        return features[:15]

    def _compute_behavior_features(self, text: str) -> List[float]:
        """Compute compression behavior statistics"""
        features = []

        # Incremental compression analysis
        sentences = self._split_sentences(text)
        if len(sentences) >= 3:
            incremental_ratios = []
            accumulated_text = ""
            for sent in sentences[:8]:  # First 8 sentences
                if sent.strip():
                    accumulated_text += " " + sent
                    ratios = self.compute_compression_ratios(accumulated_text)
                    avg_ratio = np.mean(list(ratios.values()))
                    incremental_ratios.append(avg_ratio)

            if len(incremental_ratios) >= 2:
                # Trend: are we getting more or less compressible?
                trend = np.polyfit(range(len(incremental_ratios)), incremental_ratios, 1)[0]
                features.append(trend)
                features.append(np.std(incremental_ratios))
                features.append(incremental_ratios[-1] - incremental_ratios[0])  # Change
            else:
                features.extend([0.0] * 3)
        else:
            features.extend([0.0] * 3)

        # Algorithm agreement: do different algorithms agree on compressibility?
        ratios = self.compute_compression_ratios(text)
        ratio_values = list(ratios.values())
        features.append(np.std(ratio_values))  # Low std = high agreement
        features.append(np.max(ratio_values) - np.min(ratio_values))  # Range

        # Context sensitivity: how much does context help?
        if len(sentences) >= 3:
            context_benefit = []
            for i in range(min(3, len(sentences) - 1)):
                isolated = len(zlib.compress(sentences[i].encode('utf-8')))
                with_context = len(zlib.compress((sentences[i-1] + sentences[i] if i > 0 else sentences[i] + sentences[i+1]).encode('utf-8')))
                benefit = isolated / max(with_context, 1)
                context_benefit.append(benefit)
            features.append(np.mean(context_benefit) if context_benefit else 1.0)
        else:
            features.append(1.0)

        # Add padding if needed
        while len(features) < 12:
            features.append(0.0)

        return features[:12]

    def _split_sentences(self, text: str) -> List[str]:
        """Simple sentence splitting"""
        # Basic sentence splitting on common terminators
        import re
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _split_paragraphs(self, text: str) -> List[str]:
        """Simple paragraph splitting"""
        paragraphs = text.split('\n')
        return [p.strip() for p in paragraphs if p.strip()]

    def _shuffle_text(self, text: str) -> str:
        """Shuffle words in text while preserving word boundaries"""
        # Use hash-based deterministic shuffle to ensure consistency
        words = text.split()
        # Create deterministic shuffle based on text content
        indexed_words = list(enumerate(words))
        indexed_words.sort(key=lambda x: hash(x[1] + str(self.random_seed)))
        shuffled_words = [word for _, word in indexed_words]
        return ' '.join(shuffled_words)
