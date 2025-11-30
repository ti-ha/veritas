"""
Topological Data Analysis (TDA) for text

Simplified implementation that analyzes the topological structure of text
embeddings using persistence homology concepts.
"""

import numpy as np
from typing import List
from scipy.spatial.distance import pdist, squareform
from src.analyzers.base import BaseAnalyzer


class TDAAnalyzer(BaseAnalyzer):
    """
    Topological Data Analysis analyzer for AI text detection.

    Extracts 64-dimensional feature vectors capturing topological
    signatures of text embeddings.
    """

    def __init__(self, embedding_dim: int = 128, random_seed: int = 42):
        """Initialize TDA analyzer"""
        super().__init__(random_seed)
        self._name = "tda"
        self._feature_count = 64

        self.embedding_dim = embedding_dim
        np.random.seed(random_seed)

    def _extract_features_impl(self, text: str) -> np.ndarray:
        """
        Extract 64-dimensional topological feature vector.

        Args:
            text: Input text

        Returns:
            64-dimensional numpy array
        """
        # Generate simple embeddings (in production, use sentence transformers)
        embeddings = self._generate_simple_embeddings(text)

        if len(embeddings) < 2:
            return np.zeros(64)

        features = []

        # Compute distance matrix
        dist_matrix = squareform(pdist(embeddings, metric='euclidean'))

        # 1. Distance-based topology features (20 features)
        features.extend(self._compute_distance_features(dist_matrix))

        # 2. Connectivity features (20 features)
        features.extend(self._compute_connectivity_features(dist_matrix))

        # 3. Clustering coefficient features (12 features)
        features.extend(self._compute_clustering_features(dist_matrix))

        # 4. Dimensional features (12 features)
        features.extend(self._compute_dimensional_features(embeddings))

        return np.array(features[:64], dtype=np.float64)

    def _generate_simple_embeddings(self, text: str) -> np.ndarray:
        """Generate simple embeddings from text (simplified version)"""
        sentences = [s.strip() for s in text.split('.') if s.strip()]

        if not sentences:
            return np.zeros((1, self.embedding_dim))

        embeddings = []
        for sent in sentences[:50]:  # Limit to 50 sentences
            # Simple bag-of-words style embedding
            words = sent.lower().split()
            embedding = np.zeros(self.embedding_dim)

            for i, word in enumerate(words[:20]):  # First 20 words
                # Hash word to embedding dimensions
                for j, char in enumerate(word[:10]):
                    idx = (hash(char) + i + j) % self.embedding_dim
                    embedding[idx] += 1.0

            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding /= norm

            embeddings.append(embedding)

        return np.array(embeddings)

    def _compute_distance_features(self, dist_matrix: np.ndarray) -> List[float]:
        """Compute features from distance matrix"""
        features = []

        # Basic statistics
        features.append(np.mean(dist_matrix[np.triu_indices_from(dist_matrix, k=1)]))
        features.append(np.std(dist_matrix[np.triu_indices_from(dist_matrix, k=1)]))
        features.append(np.min(dist_matrix[np.nonzero(dist_matrix)]) if np.any(dist_matrix > 0) else 0)
        features.append(np.max(dist_matrix))

        # Percentiles
        upper_tri = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
        for p in [25, 50, 75, 90]:
            features.append(np.percentile(upper_tri, p))

        # Distance variance across rows (semantic consistency)
        row_means = np.mean(dist_matrix, axis=1)
        features.append(np.mean(row_means))
        features.append(np.std(row_means))
        features.append(np.max(row_means) - np.min(row_means))

        # Nearest neighbor statistics
        # Make a copy to avoid modifying the original matrix
        dist_matrix_copy = dist_matrix.copy()
        np.fill_diagonal(dist_matrix_copy, np.inf)
        nearest_dists = np.min(dist_matrix_copy, axis=1)
        features.append(np.mean(nearest_dists))
        features.append(np.std(nearest_dists))
        features.append(np.min(nearest_dists) if len(nearest_dists) > 0 else 0)
        features.append(np.max(nearest_dists) if len(nearest_dists) > 0 else 0)

        # Eccentricity (use original dist_matrix, not the copy with inf)
        dist_matrix_copy2 = dist_matrix.copy()
        np.fill_diagonal(dist_matrix_copy2, 0)
        max_dists = np.max(dist_matrix_copy2, axis=1)
        features.append(np.mean(max_dists))
        features.append(np.std(max_dists))

        # Padding
        while len(features) < 20:
            features.append(0.0)

        return features[:20]

    def _compute_connectivity_features(self, dist_matrix: np.ndarray) -> List[float]:
        """Compute connectivity-based features"""
        features = []
        n = len(dist_matrix)

        # Compute connectivity at different thresholds
        thresholds = np.percentile(dist_matrix[np.nonzero(dist_matrix)], [10, 25, 50, 75, 90])

        for threshold in thresholds:
            # Create adjacency matrix
            adj_matrix = (dist_matrix < threshold).astype(int)
            np.fill_diagonal(adj_matrix, 0)

            # Number of connections
            num_connections = np.sum(adj_matrix) / 2
            features.append(num_connections / max(n * (n - 1) / 2, 1))

            # Average degree
            degrees = np.sum(adj_matrix, axis=1)
            features.append(np.mean(degrees) / max(n - 1, 1))

        # Graph density progression
        for i in range(len(thresholds) - 1):
            features.append(features[i * 2] - features[(i + 1) * 2])

        # Padding
        while len(features) < 20:
            features.append(0.0)

        return features[:20]

    def _compute_clustering_features(self, dist_matrix: np.ndarray) -> List[float]:
        """Compute clustering coefficient features"""
        features = []
        n = len(dist_matrix)

        if n < 3:
            return [0.0] * 12

        # Use median distance as threshold
        threshold = np.median(dist_matrix[np.nonzero(dist_matrix)])
        adj_matrix = (dist_matrix < threshold).astype(int)
        np.fill_diagonal(adj_matrix, 0)

        # Compute clustering coefficients
        clustering_coeffs = []
        for i in range(n):
            neighbors = np.where(adj_matrix[i] > 0)[0]
            if len(neighbors) < 2:
                clustering_coeffs.append(0.0)
                continue

            # Count triangles
            possible_connections = len(neighbors) * (len(neighbors) - 1) / 2
            actual_connections = 0
            for j in range(len(neighbors)):
                for k in range(j + 1, len(neighbors)):
                    if adj_matrix[neighbors[j], neighbors[k]] > 0:
                        actual_connections += 1

            clustering_coeffs.append(actual_connections / max(possible_connections, 1))

        features.append(np.mean(clustering_coeffs))
        features.append(np.std(clustering_coeffs))
        features.append(np.min(clustering_coeffs))
        features.append(np.max(clustering_coeffs))

        # Distribution of clustering coefficients
        hist, _ = np.histogram(clustering_coeffs, bins=4, range=(0, 1))
        features.extend((hist / max(np.sum(hist), 1)).tolist())

        # Padding
        while len(features) < 12:
            features.append(0.0)

        return features[:12]

    def _compute_dimensional_features(self, embeddings: np.ndarray) -> List[float]:
        """Compute intrinsic dimensionality features"""
        features = []

        # PCA-based features
        if len(embeddings) > 2:
            centered = embeddings - np.mean(embeddings, axis=0)
            cov = np.cov(centered.T)
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.sort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Remove near-zero

            if len(eigenvalues) > 0:
                # Explained variance ratios
                total_var = np.sum(eigenvalues)
                cumsum = np.cumsum(eigenvalues) / total_var

                # How many dimensions explain 90% variance?
                dims_90 = np.searchsorted(cumsum, 0.9) + 1
                features.append(dims_90 / len(eigenvalues))

                # Entropy of eigenvalues
                probs = eigenvalues / total_var
                entropy = -np.sum(probs * np.log(probs + 1e-10))
                features.append(entropy)

                # Ratio of top eigenvalue to sum
                features.append(eigenvalues[0] / total_var)

                # Effective rank
                features.append(np.exp(entropy))
            else:
                features.extend([0.0] * 4)
        else:
            features.extend([0.0] * 4)

        # Spread features
        pairwise_dists = pdist(embeddings)
        features.append(np.mean(pairwise_dists))
        features.append(np.std(pairwise_dists))
        features.append(np.max(pairwise_dists) - np.min(pairwise_dists))

        # Volume estimation
        features.append(np.std(embeddings.flatten()))

        # Padding
        while len(features) < 12:
            features.append(0.0)

        return features[:12]
