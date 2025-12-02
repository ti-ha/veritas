"""
Train production ML classifier for VERITAS using prepared dataset.

This script:
1. Loads train/validation/test splits from prepared data
2. Extracts 168D feature vectors from all samples
3. Trains multiple ML classifiers with hyperparameter tuning
4. Evaluates on validation set
5. Tests on hold-out test set
6. Saves the best model and integration code

Usage:
    python scripts/train_production_model.py
"""

import json
import numpy as np
import sys
from pathlib import Path
from typing import Tuple, Dict
import joblib
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# ML libraries
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc
)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA

# VERITAS
from src.models.ensemble import VERITASClassifier


def load_prepared_data() -> Dict[str, Tuple[list, list]]:
    """
    Load prepared train/val/test splits.

    Returns:
        Dictionary with 'train', 'val', 'test' keys containing (texts, labels)
    """
    data_dir = Path("data")
    splits = {}

    for split in ['train', 'val', 'test']:
        filepath = data_dir / f"{split}_samples.json"

        if not filepath.exists():
            print(f"[WARNING] {filepath} not found, skipping {split} split")
            splits[split] = ([], [])
            continue

        with open(filepath, 'r', encoding='utf-8') as f:
            samples = json.load(f)

        texts = [s['text'] for s in samples]
        labels = [1 if s['label'] == 'ai' else 0 for s in samples]

        splits[split] = (texts, labels)
        print(f"[LOAD] {split}: {len(texts)} samples ({sum(1 for l in labels if l == 0)} human, {sum(labels)} AI)")

    return splits


def extract_features(texts: list, name: str = "samples") -> np.ndarray:
    """
    Extract 168D feature vectors from texts.

    Returns:
        numpy array of shape (n_samples, 168)
    """
    print(f"\n[FEATURES] Extracting from {len(texts)} {name}...")

    clf = VERITASClassifier()
    features_list = []

    for i, text in enumerate(texts):
        try:
            # Extract features from each analyzer
            kcda_features = clf.kcda.extract_features(text)
            tda_features = clf.tda.extract_features(text)
            fractal_features = clf.fractal.extract_features(text)
            ergodic_features = clf.ergodic.extract_features(text)

            # Compute module-level scores
            kcda_score = clf._score_kcda(kcda_features)
            tda_score = clf._score_tda(tda_features)
            fractal_score = clf._score_fractal(fractal_features)
            ergodic_score = clf._score_ergodic(ergodic_features)

            # Combine raw features (168D) with module scores (4D) = 172D total
            features = np.concatenate([
                kcda_features,
                tda_features,
                fractal_features,
                ergodic_features,
                [kcda_score, tda_score, fractal_score, ergodic_score]
            ])

            features_list.append(features)

            if (i + 1) % 100 == 0:
                print(f"  Progress: {i + 1}/{len(texts)} ({(i+1)/len(texts)*100:.1f}%)")

        except Exception as e:
            print(f"  [ERROR] Sample {i}: {e}")
            # Use zero vector on error
            features_list.append(np.zeros(172))  # Updated size

    print(f"[OK] Feature extraction complete")
    return np.array(features_list)


def perform_feature_selection(X_train: np.ndarray, y_train: np.ndarray,
                              X_val: np.ndarray, n_features: int = 84) -> Tuple:
    """
    Perform feature selection to reduce dimensionality.

    Args:
        X_train: Training features (581, 172)
        y_train: Training labels
        X_val: Validation features
        n_features: Number of top features to select (default: 84, down from 172)

    Returns:
        (X_train_selected, X_val_selected, selector)
    """
    print("\n" + "="*60)
    print(f"FEATURE SELECTION (172 -> {n_features} features)")
    print("="*60)

    # Method 1: F-statistic (ANOVA F-value)
    print(f"\n[1] Computing F-statistics for all features...")
    selector_f = SelectKBest(f_classif, k=n_features)
    X_train_f = selector_f.fit_transform(X_train, y_train)

    # Method 2: Mutual Information
    print(f"[2] Computing Mutual Information for all features...")
    selector_mi = SelectKBest(mutual_info_classif, k=n_features)
    X_train_mi = selector_mi.fit_transform(X_train, y_train)

    # Use F-statistic by default (faster and often better for linear relationships)
    selector = selector_f
    X_train_selected = X_train_f
    X_val_selected = selector.transform(X_val)

    # Show feature importance
    feature_scores = selector.scores_
    selected_indices = selector.get_support(indices=True)

    print(f"\n[OK] Selected top {n_features} features out of 172")
    print(f"[INFO] Feature score range: {feature_scores[selected_indices].min():.2f} - {feature_scores[selected_indices].max():.2f}")
    print(f"[INFO] Selected feature indices: {list(selected_indices[:10])}... (showing first 10)")

    # Show which module features were selected
    kcda_count = sum(1 for i in selected_indices if i < 48)
    tda_count = sum(1 for i in selected_indices if 48 <= i < 112)
    fractal_count = sum(1 for i in selected_indices if 112 <= i < 144)
    ergodic_count = sum(1 for i in selected_indices if 144 <= i < 168)
    meta_count = sum(1 for i in selected_indices if i >= 168)

    print(f"\n[DISTRIBUTION] Features per module:")
    print(f"  KCDA:         {kcda_count}/{48} ({kcda_count/48*100:.1f}%)")
    print(f"  TDA:          {tda_count}/{64} ({tda_count/64*100:.1f}%)")
    print(f"  Fractal:      {fractal_count}/{32} ({fractal_count/32*100:.1f}%)")
    print(f"  Ergodic:      {ergodic_count}/{24} ({ergodic_count/24*100:.1f}%)")
    print(f"  Meta-scores:  {meta_count}/4 ({meta_count/4*100:.1f}%)")

    return X_train_selected, X_val_selected, selector


def compute_agreement_weights(module_scores: np.ndarray, agreement_threshold: float = 0.6) -> np.ndarray:
    """
    Compute sample weights based on module agreement.

    Samples where modules strongly agree get higher weight during training.
    This helps model focus on clear signal and reduces impact of ambiguous samples.

    Args:
        module_scores: (N, 4) array of module scores for N samples
        agreement_threshold: Minimum agreement to get full weight

    Returns:
        weights: (N,) array of sample weights in range [0.3, 1.0]
    """
    weights = []

    for scores in module_scores:
        # Compute agreement as 1 - (std of scores)
        # High std = low agreement, low std = high agreement
        std = np.std(scores)
        agreement = 1.0 - min(std, 1.0)  # Clamp to [0, 1]

        # Convert agreement to weight
        # High agreement (>threshold) → weight = 1.0
        # Low agreement → weight = 0.3 (still train, but with less emphasis)
        if agreement >= agreement_threshold:
            weight = 1.0
        else:
            # Linear interpolation: 0.0 agreement → 0.3 weight, threshold agreement → 1.0 weight
            weight = 0.3 + (agreement / agreement_threshold) * 0.7

        weights.append(weight)

    return np.array(weights)


def extract_features_with_agreement(texts: list, labels: list, name: str = "samples") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract features AND compute module scores + agreement weights.

    Returns:
        features: (N, 172) feature array
        module_scores: (N, 4) module score array
        sample_weights: (N,) agreement-based weights
    """
    print(f"\n[FEATURES] Extracting from {len(texts)} {name} with agreement weighting...")

    clf = VERITASClassifier()
    features_list = []
    module_scores_list = []

    for i, text in enumerate(texts):
        try:
            # Extract features from each analyzer
            kcda_features = clf.kcda.extract_features(text)
            tda_features = clf.tda.extract_features(text)
            fractal_features = clf.fractal.extract_features(text)
            ergodic_features = clf.ergodic.extract_features(text)

            # Compute module-level scores
            kcda_score = clf._score_kcda(kcda_features)
            tda_score = clf._score_tda(tda_features)
            fractal_score = clf._score_fractal(fractal_features)
            ergodic_score = clf._score_ergodic(ergodic_features)

            # Concatenate raw features + module scores
            features = np.concatenate([
                kcda_features,
                tda_features,
                fractal_features,
                ergodic_features,
                [kcda_score, tda_score, fractal_score, ergodic_score]
            ])

            features_list.append(features)
            module_scores_list.append([kcda_score, tda_score, fractal_score, ergodic_score])

            if (i + 1) % 100 == 0:
                print(f"  Progress: {i + 1}/{len(texts)} ({(i+1)/len(texts)*100:.1f}%)")

        except Exception as e:
            print(f"  [ERROR] Sample {i}: {e}")
            features_list.append(np.zeros(172))
            module_scores_list.append([0.0, 0.0, 0.0, 0.0])

    features_array = np.array(features_list)
    module_scores_array = np.array(module_scores_list)

    # Compute agreement-based weights
    sample_weights = compute_agreement_weights(module_scores_array)

    print(f"[OK] Feature extraction complete")
    print(f"[WEIGHTS] Sample weight range: {sample_weights.min():.2f} - {sample_weights.max():.2f}")
    print(f"[WEIGHTS] High agreement samples (weight=1.0): {np.sum(sample_weights == 1.0)}/{len(sample_weights)} ({np.sum(sample_weights == 1.0)/len(sample_weights)*100:.1f}%)")

    return features_array, module_scores_array, sample_weights


def train_stacking_ensemble(X_train: np.ndarray, y_train: np.ndarray,
                            X_val: np.ndarray, y_val: np.ndarray,
                            module_scores_train: np.ndarray,
                            module_scores_val: np.ndarray,
                            sample_weights: np.ndarray = None) -> Dict:
    """
    Train a 2-level stacking ensemble.

    Level 1: Base model predicts from 84 selected features → probability
    Level 2: Meta-learner combines [probability, 4 module scores] → final prediction

    Args:
        X_train: (N, 84) selected features for training
        y_train: (N,) labels
        X_val: (M, 84) selected features for validation
        y_val: (M,) labels
        module_scores_train: (N, 4) module scores for training
        module_scores_val: (M, 4) module scores for validation
        sample_weights: (N,) agreement-based weights (optional)

    Returns:
        Dictionary with base_model, meta_model, and performance metrics
    """
    print("\n" + "="*60)
    print("STACKING ENSEMBLE (2-Level Architecture)")
    print("="*60)

    # Train base model on selected features
    print("\n[LEVEL 1] Training base model on selected features...")
    base_model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        min_samples_split=20,
        subsample=0.9,
        random_state=42
    )

    if sample_weights is not None:
        base_model.fit(X_train, y_train, sample_weight=sample_weights)
        print(f"  [OK] Trained with agreement-based sample weighting")
    else:
        base_model.fit(X_train, y_train)

    train_proba = base_model.predict_proba(X_train)[:, 1]
    val_proba = base_model.predict_proba(X_val)[:, 1]

    train_acc = base_model.score(X_train, y_train)
    val_acc = base_model.score(X_val, y_val)
    print(f"  Base model - Train: {train_acc:.2%}, Val: {val_acc:.2%}")

    # Train meta-learner combining base probability with module scores
    print("\n[LEVEL 2] Training meta-learner on base probability and module scores...")
    X_meta_train = np.column_stack([
        train_proba.reshape(-1, 1),
        module_scores_train
    ])  # Shape: (N, 5)

    X_meta_val = np.column_stack([
        val_proba.reshape(-1, 1),
        module_scores_val
    ])  # Shape: (M, 5)

    # Meta-learner: Simple logistic regression (prevents overfitting on 5 features)
    meta_model = LogisticRegression(
        C=1.0,  # Moderate regularization
        max_iter=1000,
        random_state=42
    )

    if sample_weights is not None:
        meta_model.fit(X_meta_train, y_train, sample_weight=sample_weights)
    else:
        meta_model.fit(X_meta_train, y_train)

    # Evaluate stacking ensemble
    train_acc_stack = meta_model.score(X_meta_train, y_train)
    val_acc_stack = meta_model.score(X_meta_val, y_val)

    print(f"  Meta-learner - Train: {train_acc_stack:.2%}, Val: {val_acc_stack:.2%}")
    print(f"  Improvement: Train {train_acc_stack - train_acc:+.2%}, Val {val_acc_stack - val_acc:+.2%}")

    # Show meta-learner feature importance (coefficients)
    coeffs = meta_model.coef_[0]
    feature_names = ['ML_Probability', 'KCDA', 'TDA', 'Fractal', 'Ergodic']
    print(f"\n[META-WEIGHTS] Feature importance:")
    for name, coef in zip(feature_names, coeffs):
        print(f"  {name:15s}: {coef:+.3f}")

    return {
        'base_model': base_model,
        'meta_model': meta_model,
        'train_acc_base': train_acc,
        'val_acc_base': val_acc,
        'train_acc_stack': train_acc_stack,
        'val_acc_stack': val_acc_stack
    }


def train_baseline_models(X_train: np.ndarray, y_train: np.ndarray,
                         X_val: np.ndarray, y_val: np.ndarray) -> Dict:
    """
    Train baseline models quickly to establish performance floor.
    """
    print("\n" + "="*60)
    print("BASELINE MODELS")
    print("="*60)

    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'RandomForest_100': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        'GradientBoosting_100': GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42),
    }

    results = {}

    for name, model in models.items():
        print(f"\n{name}:")

        # Train
        model.fit(X_train, y_train)

        # Evaluate
        train_acc = model.score(X_train, y_train)
        val_acc = model.score(X_val, y_val)

        print(f"  Train Accuracy: {train_acc:.2%}")
        print(f"  Val Accuracy:   {val_acc:.2%}")

        results[name] = {
            'model': model,
            'train_acc': train_acc,
            'val_acc': val_acc
        }

    return results


def train_optimized_models(X_train: np.ndarray, y_train: np.ndarray,
                          X_val: np.ndarray, y_val: np.ndarray) -> Dict:
    """
    Train models with hyperparameter tuning.
    """
    print("\n" + "="*60)
    print("OPTIMIZED MODELS (Hyperparameter Tuning)")
    print("="*60)

    # Define models with parameter grids (with stronger regularization)
    model_configs = {
        'GradientBoosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [2, 3, 4],  # Reduced from [3, 5, 7] for stronger regularization
                'learning_rate': [0.01, 0.05, 0.1],  # Reduced from [0.01, 0.1, 0.2]
                'min_samples_split': [10, 20, 30],  # Increased from [2, 5, 10] for stronger regularization
                'subsample': [0.8, 0.9, 1.0]  # Added: fraction of samples for fitting individual trees
            }
        },
        'RandomForest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        },
        'MLP': {
            'model': MLPClassifier(random_state=42, max_iter=500),
            'params': {
                'hidden_layer_sizes': [(100,), (100, 50), (168, 84, 42)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['adaptive']
            }
        }
    }

    results = {}

    for name, config in model_configs.items():
        print(f"\n{name}:")
        print(f"  Searching {len(list(config['params'].values())[0]) ** len(config['params'])} configurations...")

        # Grid search with cross-validation
        grid_search = GridSearchCV(
            config['model'],
            config['params'],
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
            verbose=0
        )

        grid_search.fit(X_train, y_train)

        # Best model
        best_model = grid_search.best_estimator_

        # Evaluate
        train_acc = best_model.score(X_train, y_train)
        val_acc = best_model.score(X_val, y_val)

        print(f"  Best params: {grid_search.best_params_}")
        print(f"  Train Accuracy: {train_acc:.2%}")
        print(f"  Val Accuracy:   {val_acc:.2%}")

        results[name] = {
            'model': best_model,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'params': grid_search.best_params_
        }

    return results


def detailed_evaluation(model, X: np.ndarray, y: np.ndarray, name: str = "Test"):
    """Comprehensive model evaluation."""
    print("\n" + "="*60)
    print(f"{name} Set Evaluation")
    print("="*60)

    # Predictions
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=['Human', 'AI'], digits=4))

    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y, y_pred)
    print(f"                 Predicted")
    print(f"                 Human    AI")
    print(f"Actual  Human    {cm[0,0]:6d}  {cm[0,1]:6d}")
    print(f"        AI       {cm[1,0]:6d}  {cm[1,1]:6d}")

    # Metrics
    if y_proba is not None:
        try:
            roc_auc = roc_auc_score(y, y_proba)
            precision, recall, _ = precision_recall_curve(y, y_proba)
            pr_auc = auc(recall, precision)

            print(f"\nROC-AUC Score: {roc_auc:.4f}")
            print(f"PR-AUC Score:  {pr_auc:.4f}")
        except Exception as e:
            print(f"\n[WARNING] Could not compute AUC: {e}")

    # Per-class metrics
    human_mask = y == 0
    ai_mask = y == 1

    if np.sum(human_mask) > 0:
        human_correct = np.sum((y_pred == 0) & human_mask)
        human_total = np.sum(human_mask)
        human_fpr = np.sum((y_pred == 1) & human_mask) / human_total
        print(f"\nHuman Detection:")
        print(f"  Accuracy: {human_correct/human_total:.2%}")
        print(f"  FPR (False Positive Rate): {human_fpr:.2%}")

    if np.sum(ai_mask) > 0:
        ai_correct = np.sum((y_pred == 1) & ai_mask)
        ai_total = np.sum(ai_mask)
        ai_tpr = ai_correct / ai_total
        print(f"\nAI Detection:")
        print(f"  Accuracy (TPR): {ai_tpr:.2%}")
        print(f"  Missed: {ai_total - ai_correct}/{ai_total}")

    return {
        'confusion_matrix': cm,
        'roc_auc': roc_auc if y_proba is not None else None,
        'human_fpr': human_fpr if np.sum(human_mask) > 0 else None,
        'ai_tpr': ai_tpr if np.sum(ai_mask) > 0 else None
    }


def save_production_model(model, scaler, metadata: dict, output_dir: str = "models"):
    """Save model with metadata for production use."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save model
    model_file = output_path / f"veritas_classifier_{timestamp}.pkl"
    joblib.dump(model, model_file)

    # Save scaler if used
    if scaler is not None:
        scaler_file = output_path / f"veritas_scaler_{timestamp}.pkl"
        joblib.dump(scaler, scaler_file)

    # Save metadata
    metadata_file = output_path / f"model_metadata_{timestamp}.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[SAVE] Production model saved:")
    print(f"  Model: {model_file}")
    if scaler:
        print(f"  Scaler: {scaler_file}")
    print(f"  Metadata: {metadata_file}")

    return model_file


def main():
    """Main training pipeline."""
    print("="*60)
    print("VERITAS PRODUCTION MODEL TRAINING")
    print("="*60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. Load data
    print("\n[STEP 1] Loading prepared data...")
    data_splits = load_prepared_data()

    train_texts, train_labels = data_splits['train']
    val_texts, val_labels = data_splits['val']
    test_texts, test_labels = data_splits['test']

    if len(train_texts) == 0:
        print("\n[ERROR] No training data found!")
        print("Please run: python scripts/prepare_training_data.py")
        return

    # 2. Extract features WITH module scores and agreement weights
    print("\n[STEP 2] Extracting features with module scores...")
    X_train, module_scores_train, weights_train = extract_features_with_agreement(
        train_texts, train_labels, "training samples"
    )

    if len(val_texts) > 0:
        X_val, module_scores_val, weights_val = extract_features_with_agreement(
            val_texts, val_labels, "validation samples"
        )
    else:
        X_val, module_scores_val, weights_val = np.array([]), np.array([]), np.array([])

    if len(test_texts) > 0:
        X_test, module_scores_test, weights_test = extract_features_with_agreement(
            test_texts, test_labels, "test samples"
        )
    else:
        X_test, module_scores_test, weights_test = np.array([]), np.array([]), np.array([])

    y_train = np.array(train_labels)
    y_val = np.array(val_labels) if len(val_labels) > 0 else np.array([])
    y_test = np.array(test_labels) if len(test_labels) > 0 else np.array([])

    # Save features, module scores, and weights
    np.save("data/X_train.npy", X_train)
    np.save("data/y_train.npy", y_train)
    np.save("data/module_scores_train.npy", module_scores_train)
    np.save("data/weights_train.npy", weights_train)
    if len(X_val) > 0:
        np.save("data/X_val.npy", X_val)
        np.save("data/y_val.npy", y_val)
        np.save("data/module_scores_val.npy", module_scores_val)
        np.save("data/weights_val.npy", weights_val)
    if len(X_test) > 0:
        np.save("data/X_test.npy", X_test)
        np.save("data/y_test.npy", y_test)
        np.save("data/module_scores_test.npy", module_scores_test)
        np.save("data/weights_test.npy", weights_test)

    print(f"\n[OK] Features saved to data/ directory")
    print(f"  Training: {X_train.shape}")
    if len(X_val) > 0:
        print(f"  Validation: {X_val.shape}")
    if len(X_test) > 0:
        print(f"  Test: {X_test.shape}")

    # 2.5. Feature Selection (reduce from 172 to 84 features)
    print("\n[STEP 2.5] Performing feature selection...")
    X_train_selected, X_val_selected, feature_selector = perform_feature_selection(
        X_train, y_train, X_val, n_features=84
    )

    # Also transform test set
    X_test_selected = feature_selector.transform(X_test) if len(X_test) > 0 else np.array([])

    # Save feature selector
    import joblib
    joblib.dump(feature_selector, "models/feature_selector.pkl")
    print(f"[SAVE] Feature selector saved to models/feature_selector.pkl")

    # Use selected features for training
    X_train_for_training = X_train_selected
    X_val_for_training = X_val_selected
    X_test_for_training = X_test_selected

    # 3. Train baseline models
    if len(X_val) > 0:
        print("\n[STEP 3] Training baseline models...")
        baseline_results = train_baseline_models(X_train_for_training, y_train, X_val_for_training, y_val)

    # 4. Train optimized models (with stronger regularization)
    if len(X_val) > 0:
        print("\n[STEP 4] Training optimized models with stronger regularization...")
        optimized_results = train_optimized_models(X_train_for_training, y_train, X_val_for_training, y_val)

        # Combine all results
        all_results = {**baseline_results, **optimized_results}

    # 5. Train stacking ensemble with agreement weighting
    if len(X_val) > 0:
        print("\n[STEP 5] Training stacking ensemble with agreement weighting...")

        # Get module scores for selected features
        module_scores_train_selected = module_scores_train
        module_scores_val_selected = module_scores_val

        stacking_results = train_stacking_ensemble(
            X_train_for_training, y_train,
            X_val_for_training, y_val,
            module_scores_train_selected,
            module_scores_val_selected,
            sample_weights=weights_train
        )

        # Save stacking models
        joblib.dump(stacking_results['base_model'], "models/GradientBoosting_stacking_base.pkl")
        joblib.dump(stacking_results['meta_model'], "models/LogisticRegression_stacking_meta.pkl")
        print(f"\n[SAVE] Stacking models saved:")
        print(f"  Base:  models/GradientBoosting_stacking_base.pkl")
        print(f"  Meta:  models/LogisticRegression_stacking_meta.pkl")

        # Add to results
        all_results['Stacking_Ensemble'] = stacking_results

        # 5. Select best model
        print("\n[STEP 5] Selecting best model...")
        best_name = max(all_results.keys(), key=lambda k: all_results[k]['val_acc'])
        best_model = all_results[best_name]['model']

        print(f"\n[BEST] {best_name}")
        print(f"  Train Accuracy: {all_results[best_name]['train_acc']:.2%}")
        print(f"  Val Accuracy:   {all_results[best_name]['val_acc']:.2%}")
    else:
        # No validation set, just train one model
        print("\n[STEP 3-5] Training final model (no validation set)...")
        best_model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        best_model.fit(X_train, y_train)
        best_name = "GradientBoosting"

    # 6. Final evaluation on test set
    if len(X_test) > 0:
        print("\n[STEP 6] Final evaluation on test set...")
        test_metrics = detailed_evaluation(best_model, X_test, y_test, "Test")
    else:
        print("\n[STEP 6] Evaluating on training set (no test set)...")
        test_metrics = detailed_evaluation(best_model, X_train, y_train, "Training")

    # 7. Save production model
    print("\n[STEP 7] Saving production model...")
    metadata = {
        'model_type': best_name,
        'training_date': datetime.now().isoformat(),
        'training_samples': len(train_texts),
        'validation_samples': len(val_texts),
        'test_samples': len(test_texts),
        'feature_dimensions': 168,
        'test_metrics': {
            'roc_auc': float(test_metrics['roc_auc']) if test_metrics['roc_auc'] else None,
            'human_fpr': float(test_metrics['human_fpr']) if test_metrics['human_fpr'] else None,
            'ai_tpr': float(test_metrics['ai_tpr']) if test_metrics['ai_tpr'] else None
        }
    }

    model_file = save_production_model(best_model, None, metadata)

    # 8. Summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nModel: {best_name}")
    print(f"Saved to: {model_file}")
    print(f"\nNext steps:")
    print(f"1. Integrate model into VERITASClassifier")
    print(f"2. Run validation tests: pytest tests/test_validation.py -v")
    print(f"3. Test API: uvicorn src.api.main:app --reload")
    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
