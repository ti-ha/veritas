"""
Train ML classifier for VERITAS AI detection.

This script:
1. Loads samples from data/test_samples.json
2. Extracts 168D feature vectors from all samples
3. Trains multiple ML classifiers
4. Evaluates and compares performance
5. Saves the best model

Usage:
    python scripts/train_ml_classifier.py
"""

import json
import numpy as np
import sys
from pathlib import Path
from typing import Tuple, Dict
import joblib

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# ML libraries
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# VERITAS
from src.models.ensemble import VERITASClassifier


def load_samples(samples_path: str = "data/test_samples.json") -> Tuple[list, list]:
    """Load samples from JSON file."""
    with open(samples_path, 'r', encoding='utf-8') as f:
        samples = json.load(f)

    texts = [s['text'] for s in samples]
    labels = [1 if s['label'] == 'ai' else 0 for s in samples]

    return texts, labels


def extract_features(texts: list) -> np.ndarray:
    """
    Extract 168D feature vectors from texts.

    Returns:
        numpy array of shape (n_samples, 168)
    """
    print(f"Extracting features from {len(texts)} samples...")

    clf = VERITASClassifier()
    features_list = []

    for i, text in enumerate(texts):
        try:
            # Extract features from each analyzer
            kcda_features = clf.kcda.extract_features(text)
            tda_features = clf.tda.extract_features(text)
            fractal_features = clf.fractal.extract_features(text)
            ergodic_features = clf.ergodic.extract_features(text)

            # Concatenate all features (48 + 64 + 32 + 24 = 168)
            features = np.concatenate([
                kcda_features,
                tda_features,
                fractal_features,
                ergodic_features
            ])

            features_list.append(features)

            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(texts)} samples")

        except Exception as e:
            print(f"  Error processing sample {i}: {e}")
            # Use zero vector on error
            features_list.append(np.zeros(168))

    print(f"[OK] Feature extraction complete")
    return np.array(features_list)


def train_models(X: np.ndarray, y: np.ndarray) -> Dict:
    """
    Train multiple ML classifiers and compare performance.

    Returns:
        Dictionary of trained models with their scores
    """
    print(f"\nTraining classifiers on {len(X)} samples...")
    print(f"Feature dimensions: {X.shape[1]}")
    print(f"Class distribution: {np.sum(y == 0)} human, {np.sum(y == 1)} AI")

    # Define models to try
    models = {
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        ),
        'LogisticRegression': LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    }

    results = {}

    for name, model in models.items():
        print(f"\n{name}:")

        # Cross-validation (if enough samples)
        if len(X) >= 10:
            cv_scores = cross_val_score(model, X, y, cv=min(5, len(X) // 2))
            print(f"  CV Accuracy: {cv_scores.mean():.2%} ± {cv_scores.std():.2%}")

        # Train on all data
        model.fit(X, y)
        train_accuracy = model.score(X, y)
        print(f"  Train Accuracy: {train_accuracy:.2%}")

        # Feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            top_indices = np.argsort(importances)[-10:][::-1]
            print(f"  Top 10 features:")
            for idx in top_indices:
                analyzer = get_analyzer_name(idx)
                print(f"    Feature {idx} ({analyzer}): {importances[idx]:.4f}")

        results[name] = {
            'model': model,
            'train_accuracy': train_accuracy,
            'cv_scores': cv_scores if len(X) >= 10 else None
        }

    return results


def get_analyzer_name(feature_idx: int) -> str:
    """Get analyzer name for a feature index."""
    if feature_idx < 48:
        return f"KCDA_{feature_idx}"
    elif feature_idx < 112:  # 48 + 64
        return f"TDA_{feature_idx - 48}"
    elif feature_idx < 144:  # 48 + 64 + 32
        return f"Fractal_{feature_idx - 112}"
    else:
        return f"Ergodic_{feature_idx - 144}"


def evaluate_model(model, X: np.ndarray, y: np.ndarray):
    """Detailed evaluation of a model."""
    print("\n" + "="*60)
    print("Detailed Model Evaluation")
    print("="*60)

    # Predictions
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=['Human', 'AI']))

    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y, y_pred)
    print(f"                 Predicted")
    print(f"                 Human  AI")
    print(f"Actual  Human    {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"        AI       {cm[1,0]:4d}  {cm[1,1]:4d}")

    # ROC-AUC if probabilities available
    if y_proba is not None:
        try:
            auc = roc_auc_score(y, y_proba)
            print(f"\nROC-AUC Score: {auc:.4f}")
        except:
            print("\nROC-AUC: Cannot compute (need both classes in predictions)")

    # Per-class accuracy
    human_mask = y == 0
    ai_mask = y == 1

    if np.sum(human_mask) > 0:
        human_accuracy = np.sum((y_pred == y) & human_mask) / np.sum(human_mask)
        print(f"\nHuman Detection Accuracy: {human_accuracy:.2%}")

    if np.sum(ai_mask) > 0:
        ai_accuracy = np.sum((y_pred == y) & ai_mask) / np.sum(ai_mask)
        print(f"AI Detection Accuracy: {ai_accuracy:.2%}")


def save_model(model, name: str, output_dir: str = "models"):
    """Save trained model to disk."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    model_file = output_path / f"{name}_classifier.pkl"
    joblib.dump(model, model_file)
    print(f"\n[OK] Model saved to {model_file}")


def main():
    """Main training pipeline."""
    print("="*60)
    print("VERITAS ML Classifier Training")
    print("="*60)

    # 1. Load samples
    print("\n1. Loading samples...")
    texts, labels = load_samples()
    print(f"[OK] Loaded {len(texts)} samples ({np.sum(np.array(labels) == 0)} human, {np.sum(np.array(labels) == 1)} AI)")

    # 2. Extract features
    print("\n2. Extracting features...")
    X = extract_features(texts)
    y = np.array(labels)

    # Save features for future use
    np.save("data/X_features.npy", X)
    np.save("data/y_labels.npy", y)
    print("[OK] Features saved to data/X_features.npy")

    # 3. Train models
    print("\n3. Training classifiers...")
    results = train_models(X, y)

    # 4. Find best model
    print("\n4. Selecting best model...")
    best_name = max(results.keys(), key=lambda k: results[k]['train_accuracy'])
    best_model = results[best_name]['model']

    print(f"\n[OK] Best model: {best_name}")
    print(f"  Train Accuracy: {results[best_name]['train_accuracy']:.2%}")

    # 5. Detailed evaluation
    evaluate_model(best_model, X, y)

    # 6. Save best model
    save_model(best_model, best_name)

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"\nNext steps:")
    print(f"1. Add more samples to data/test_samples.json")
    print(f"2. Re-run this script to retrain")
    print(f"3. Integrate model into src/models/ensemble.py")
    print(f"4. Run validation tests: pytest tests/test_validation.py -v")


if __name__ == "__main__":
    main()
