"""
Ensemble Classification System for VERITAS

Combines features from all analysis modules and provides hierarchical
classification with calibrated uncertainty.
"""

import numpy as np
from typing import Dict, Tuple, List
from src.algorithms.kcda import KCDAAnalyzer
from src.algorithms.tda import TDAAnalyzer
from src.algorithms.fractal import FractalAnalyzer
from src.algorithms.ergodic import ErgodicAnalyzer


class VERITASClassifier:
    """
    Multi-modal ensemble classifier combining KCDA, TDA, Fractal, and Ergodic analysis.

    Provides hierarchical classification:
    - Level 1: Definitive (high confidence across all models)
    - Level 2: Probabilistic (majority agreement)
    - Level 3: Inconclusive (model disagreement)
    """

    def __init__(self, random_seed: int = 42):
        """Initialize the VERITAS classifier"""
        self.random_seed = random_seed

        # Initialize all analyzers
        self.kcda = KCDAAnalyzer(random_seed=random_seed)
        self.tda = TDAAnalyzer(random_seed=random_seed)
        self.fractal = FractalAnalyzer(random_seed=random_seed)
        self.ergodic = ErgodicAnalyzer(random_seed=random_seed)

        # Load ML classifier (stacking ensemble if available, otherwise single model)
        self.ml_base_model = None
        self.ml_meta_model = None
        self.feature_selector = None
        try:
            import joblib
            from pathlib import Path
            models_dir = Path(__file__).parent.parent.parent / "models"

            # Attempt to load stacking ensemble models
            base_path = models_dir / "GradientBoosting_stacking_base.pkl"
            meta_path = models_dir / "LogisticRegression_stacking_meta.pkl"
            selector_path = models_dir / "feature_selector.pkl"

            if base_path.exists() and meta_path.exists() and selector_path.exists():
                self.ml_base_model = joblib.load(base_path)
                self.ml_meta_model = joblib.load(meta_path)
                self.feature_selector = joblib.load(selector_path)
                print(f"[ML] Loaded stacking ensemble: base + meta models with feature selector")
            else:
                # Load single model if stacking ensemble unavailable
                single_model_path = models_dir / "GradientBoosting_classifier.pkl"
                if single_model_path.exists():
                    self.ml_base_model = joblib.load(single_model_path)
                    print(f"[ML] Loaded single classifier from {single_model_path}")
        except Exception as e:
            print(f"[ML] No trained classifier found, using heuristic scoring: {e}")

        # Module reliability weights (based on theoretical strength and expected accuracy)
        # These would ideally be learned from validation data
        self.module_weights = {
            'kcda': 0.30,    # Strong theoretical foundation, compression-based
            'tda': 0.25,     # Novel approach, captures semantic structure
            'fractal': 0.25, # Good at detecting uniformity patterns
            'ergodic': 0.20  # Complementary statistical analysis
        }

        # Improved thresholds for classification
        self.thresholds = {
            'definitive_ai': 0.75,        # Strong AI signal
            'probable_ai': 0.60,          # Moderate AI signal
            'ambiguous_high': 0.55,       # Upper ambiguous zone
            'ambiguous_low': 0.45,        # Lower ambiguous zone
            'probable_human': 0.40,       # Moderate human signal
            'definitive_human': 0.25      # Strong human signal
        }

        # Agreement thresholds
        self.agreement_thresholds = {
            'high': 0.85,      # Near-unanimous
            'moderate': 0.70,  # Strong majority
            'low': 0.50        # Weak majority
        }

    def analyze_text(self, text: str) -> Dict:
        """
        Analyze text and provide comprehensive detection results.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary containing:
            - classification_level: 1 (definitive), 2 (probabilistic), or 3 (inconclusive)
            - ai_probability: Estimated probability of AI authorship
            - confidence: Confidence in the classification
            - features: Feature breakdown from each module
            - explanation: Human-readable explanation
        """
        if not text or len(text.strip()) < 50:
            return {
                'classification_level': 3,
                'ai_probability': 0.5,
                'confidence': 0.0,
                'explanation': 'Text too short for reliable analysis (minimum 50 characters)',
                'features': {}
            }

        # Extract features from all modules
        try:
            kcda_features = self.kcda.extract_features(text)
            tda_features = self.tda.extract_features(text)
            fractal_features = self.fractal.extract_features(text)
            ergodic_features = self.ergodic.extract_features(text)
        except Exception as e:
            return {
                'classification_level': 3,
                'ai_probability': 0.5,
                'confidence': 0.0,
                'explanation': f'Analysis error: {str(e)}',
                'features': {}
            }

        # Compute module-level scores
        kcda_score = self._score_kcda(kcda_features)
        tda_score = self._score_tda(tda_features)
        fractal_score = self._score_fractal(fractal_features)
        ergodic_score = self._score_ergodic(ergodic_features)

        module_scores = {
            'kcda': kcda_score,
            'tda': tda_score,
            'fractal': fractal_score,
            'ergodic': ergodic_score
        }

        # Compute agreement based on module scores
        scores_array = np.array(list(module_scores.values()))
        agreement = self._compute_agreement(scores_array)
        outliers = self._detect_outliers(module_scores)

        if self.ml_base_model is not None:
            # Combine raw features (168D) with module scores (4D) = 172D total
            combined_features = np.concatenate([
                kcda_features,
                tda_features,
                fractal_features,
                ergodic_features,
                [kcda_score, tda_score, fractal_score, ergodic_score]
            ])

            if self.ml_meta_model is not None and self.feature_selector is not None:
                # Apply stacking ensemble with feature selection
                combined_features_selected = self.feature_selector.transform([combined_features])
                base_probability = self.ml_base_model.predict_proba(combined_features_selected)[0][1]

                # Meta-model combines base probability with module scores
                meta_features = np.array([[base_probability, kcda_score, tda_score, fractal_score, ergodic_score]])
                ml_probability = self.ml_meta_model.predict_proba(meta_features)[0][1]
                weighted_score = ml_probability
            else:
                # Use single model on raw features only
                ml_probability = self.ml_base_model.predict_proba([combined_features[:168]])[0][1]
                weighted_score = ml_probability

        else:
            # Use weighted ensemble of heuristic module scores
            weighted_score, agreement, outliers = self._compute_weighted_ensemble(module_scores)

        # Determine classification level based on weighted score and agreement
        classification_level, confidence = self._determine_classification_level(
            module_scores, weighted_score, agreement, outliers
        )

        # Generate explanation
        explanation = self._generate_explanation(
            classification_level, weighted_score, module_scores, text, outliers
        )

        return {
            'classification_level': classification_level,
            'ai_probability': float(weighted_score),
            'confidence': float(confidence),
            'explanation': explanation,
            'features': {
                'kcda_score': float(kcda_score),
                'tda_score': float(tda_score),
                'fractal_score': float(fractal_score),
                'ergodic_score': float(ergodic_score),
                'feature_vector_size': len(combined_features),
                'agreement_score': float(agreement),
                'outlier_modules': outliers
            },
            'word_count': len(text.split()),
            'module_scores': {k: float(v) for k, v in module_scores.items()}
        }

    def _score_kcda(self, features: np.ndarray) -> float:
        """Score KCDA features for AI likelihood"""
        # AI text tends to be more compressible (lower compression ratios)
        # and shows less cross-complexity variation
        compression_ratios = features[:9]
        cross_complexity_stats = features[9:21]

        # Lower compression ratio suggests AI (more predictable)
        compression_score = 1.0 - np.mean(compression_ratios)

        # Lower cross-complexity variance suggests AI (more consistent)
        if np.std(cross_complexity_stats) > 0:
            consistency_score = 1.0 / (1.0 + np.std(cross_complexity_stats))
        else:
            consistency_score = 0.8

        return (compression_score + consistency_score) / 2

    def _score_tda(self, features: np.ndarray) -> float:
        """Score TDA features for AI likelihood"""
        # RECALIBRATED: Return neutral score to reduce TDA bias
        # Original scoring showed systematic overestimation (0.76 avg on mixed data)
        # Need more research to determine correct topological signatures

        # For now, return a conservative estimate based on feature variance
        all_features = features[:64]
        feature_variance = np.std(all_features) if len(all_features) > 0 else 0.5

        # Moderate scoring: higher variance = more human-like irregularity
        # Lower variance = more AI-like regularity
        score = 0.5 + (0.3 * (0.5 - min(feature_variance, 1.0)))  # Range: ~0.35-0.65

        return np.clip(score, 0.0, 1.0)

    def _score_fractal(self, features: np.ndarray) -> float:
        """Score fractal features for AI likelihood"""
        # RECALIBRATED: Moderate scoring to avoid extremes
        # AI text tends to have more uniform fractal properties
        box_counting_features = features[:12]
        fractality_degrees = features[12:22]

        # Lower variance in fractality suggests AI (more uniform)
        fractality_variance = np.std(fractality_degrees) if len(fractality_degrees) > 0 else 0.5
        uniformity_score = 1.0 / (1.0 + fractality_variance * 2.0)  # Scaled for sensitivity

        # Use mean box-counting value directly (normalized)
        box_mean = np.mean(box_counting_features) if len(box_counting_features) > 0 else 0.5
        dimension_score = np.clip(box_mean, 0.0, 1.0)

        return (uniformity_score * 0.6 + dimension_score * 0.4)

    def _score_ergodic(self, features: np.ndarray) -> float:
        """Score ergodic features for AI likelihood"""
        # AI text shows faster mixing (shorter decorrelation time)
        autocorr_features = features[:8]
        mixing_features = features[8:16]

        # Faster decay in autocorrelation suggests AI
        decay_rate = autocorr_features[5] if len(autocorr_features) > 5 else 0.5
        mixing_score = min(decay_rate, 1.0)

        # Higher entropy suggests AI (more uniform distribution)
        if len(mixing_features) > 1:
            entropy = mixing_features[1]
            entropy_score = min(entropy / 10.0, 1.0)  # Normalize
        else:
            entropy_score = 0.5

        return (mixing_score + entropy_score) / 2

    def _compute_weighted_ensemble(self, module_scores: Dict[str, float]) -> Tuple[float, float, List[str]]:
        """
        Compute weighted ensemble score with outlier detection.

        Returns:
            (weighted_score, agreement, outliers)
        """
        # 1. Compute weighted average
        weighted_score = sum(
            score * self.module_weights[module]
            for module, score in module_scores.items()
        )

        # 2. Detect outliers using IQR method
        scores_array = np.array(list(module_scores.values()))
        q1 = np.percentile(scores_array, 25)
        q3 = np.percentile(scores_array, 75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = []
        for module, score in module_scores.items():
            if score < lower_bound or score > upper_bound:
                outliers.append(module)

        # 3. Compute agreement score (pairwise agreement)
        agreement = self._compute_agreement(scores_array)

        return weighted_score, agreement, outliers

    def _compute_agreement(self, scores: np.ndarray) -> float:
        """
        Compute inter-module agreement using pairwise agreement measure.

        Agreement is high when modules give similar scores.
        """
        # Convert to binary classification (AI vs Human)
        binary = (scores > 0.5).astype(int)

        # Compute pairwise agreement
        n = len(binary)
        if n < 2:
            return 1.0

        agreements = 0
        total_pairs = 0

        for i in range(n):
            for j in range(i+1, n):
                # Agreement on classification direction
                if binary[i] == binary[j]:
                    agreements += 1
                total_pairs += 1

        basic_agreement = agreements / max(total_pairs, 1)

        # Also consider score similarity (not just direction)
        score_std = np.std(scores)
        similarity_score = 1.0 / (1.0 + score_std)  # Lower std = higher similarity

        # Combine both measures
        agreement = 0.6 * basic_agreement + 0.4 * similarity_score

        return agreement

    def _detect_outliers(self, module_scores: Dict[str, float]) -> List[str]:
        """
        Detect modules with outlier scores using IQR method.

        Returns:
            List of module names that are statistical outliers
        """
        scores = np.array(list(module_scores.values()))
        q1 = np.percentile(scores, 25)
        q3 = np.percentile(scores, 75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = []
        for module, score in module_scores.items():
            if score < lower_bound or score > upper_bound:
                outliers.append(module)

        return outliers

    def _determine_classification_level(
        self, module_scores: Dict[str, float], weighted_score: float,
        agreement: float, outliers: List[str]
    ) -> Tuple[int, float]:
        """
        Improved classification level determination using weighted voting and agreement.

        Returns:
            (classification_level, confidence)
        """
        # Level 1 (Definitive): High agreement + strong signal + no outliers
        if (agreement > self.agreement_thresholds['high'] and
            len(outliers) == 0):
            if weighted_score > self.thresholds['definitive_ai']:
                return 1, 0.95
            elif weighted_score < self.thresholds['definitive_human']:
                return 1, 0.95

        # Level 1 with moderate agreement but very strong signal
        if (agreement > self.agreement_thresholds['moderate'] and
            len(outliers) == 0):
            if weighted_score > 0.85 or weighted_score < 0.15:
                return 1, 0.90

        # Level 2 (Probabilistic): Moderate agreement + moderate signal
        if agreement > self.agreement_thresholds['moderate']:
            if (weighted_score > self.thresholds['probable_ai'] or
                weighted_score < self.thresholds['probable_human']):
                confidence = 0.70 + (agreement - self.agreement_thresholds['moderate']) * 0.5
                return 2, min(confidence, 0.85)

        # Level 2 with lower agreement but clear majority
        if agreement > self.agreement_thresholds['low']:
            if (weighted_score > self.thresholds['probable_ai'] or
                weighted_score < self.thresholds['probable_human']):
                return 2, 0.65

        # Level 3 (Inconclusive): Low agreement or ambiguous signal or outliers present
        if (agreement < self.agreement_thresholds['low'] or
            len(outliers) > 1 or
            (self.thresholds['ambiguous_low'] < weighted_score < self.thresholds['ambiguous_high'])):
            return 3, 0.40

        # Default to Level 2 with moderate confidence
        return 2, 0.60

    def _generate_explanation(
        self, level: int, score: float, module_scores: Dict[str, float],
        text: str, outliers: List[str]
    ) -> str:
        """Generate human-readable explanation of the analysis with outlier information"""
        word_count = len(text.split())

        if level == 1:
            if score > 0.5:
                return (f"DEFINITIVE AI: Strong consensus across all analysis modules ({score:.1%} AI probability). "
                       f"All four frameworks (KCDA, TDA, Fractal, Ergodic) independently detect AI signatures. "
                       f"Compression patterns ({module_scores['kcda']:.2f}), topological structure "
                       f"({module_scores['tda']:.2f}), fractal properties ({module_scores['fractal']:.2f}), "
                       f"and ergodic behavior ({module_scores['ergodic']:.2f}) all indicate AI generation. "
                       f"Analyzed {word_count} words with high confidence.")
            else:
                return (f"DEFINITIVE HUMAN: Strong consensus across all modules ({(1-score):.1%} human probability). "
                       f"All frameworks independently detect human authorship signatures. Natural compression patterns, "
                       f"complex semantic topology, irregular fractal word distributions, and non-Markovian statistical "
                       f"dependencies all indicate authentic human writing. Analyzed {word_count} words with high confidence.")

        elif level == 2:
            if score > 0.5:
                strongest = max(module_scores, key=module_scores.get)
                weakest = min(module_scores, key=module_scores.get)
                outlier_note = f" Note: {', '.join(outliers)} showing outlier behavior." if outliers else ""
                return (f"PROBABLE AI: Majority of modules indicate AI generation ({score:.1%} probability). "
                       f"Strongest indicator: {strongest.upper()} ({module_scores[strongest]:.2f}). "
                       f"Weakest indicator: {weakest.upper()} ({module_scores[weakest]:.2f}). "
                       f"Some inter-module variation suggests possible human editing or collaborative authorship.{outlier_note} "
                       f"Analyzed {word_count} words.")
            else:
                outlier_note = f" Note: {', '.join(outliers)} showing outlier behavior." if outliers else ""
                return (f"PROBABLE HUMAN: Majority of modules indicate human authorship ({(1-score):.1%} probability). "
                       f"Text exhibits irregular patterns characteristic of human writing, though some structured elements "
                       f"detected. Mixed signals suggest sophisticated writing or domain-specific language.{outlier_note} "
                       f"Analyzed {word_count} words.")

        else:  # level == 3
            outlier_info = f" Outlier modules: {', '.join(outliers)}." if outliers else ""
            score_spread = max(module_scores.values()) - min(module_scores.values())
            return (f"INCONCLUSIVE: Significant disagreement between analysis modules (score spread: {score_spread:.2f}).{outlier_info} "
                   f"Module scores: KCDA={module_scores['kcda']:.2f}, TDA={module_scores['tda']:.2f}, "
                   f"Fractal={module_scores['fractal']:.2f}, Ergodic={module_scores['ergodic']:.2f}. "
                   f"This pattern may indicate: (1) heavily edited AI text, (2) human-AI collaborative writing, "
                   f"(3) unusual writing style or domain-specific language, or (4) text at detection boundary. "
                   f"Analyzed {word_count} words. Manual review recommended for critical applications.")
