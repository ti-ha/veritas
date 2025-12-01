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

            # Concatenate all features (48 + 64 + 32 + 24 = 168)
            features = np.concatenate([
                kcda_features,
                tda_features,
                fractal_features,
                ergodic_features
            ])

            features_list.append(features)

            if (i + 1) % 100 == 0:
                print(f"  Progress: {i + 1}/{len(texts)} ({(i+1)/len(texts)*100:.1f}%)")

        except Exception as e:
            print(f"  [ERROR] Sample {i}: {e}")
            # Use zero vector on error
            features_list.append(np.zeros(168))

    print(f"[OK] Feature extraction complete")
    return np.array(features_list)


def perform_feature_selection(X_train: np.ndarray, y_train: np.ndarray,
                              X_val: np.ndarray, n_features: int = 80) -> Tuple:
    """
    Perform feature selection to reduce dimensionality.

    Args:
        X_train: Training features (581, 168)
        y_train: Training labels
        X_val: Validation features
        n_features: Number of top features to select (default: 80, down from 168)

    Returns:
        (X_train_selected, X_val_selected, selector)
    """
    print("\n" + "="*60)
    print(f"FEATURE SELECTION (168 -> {n_features} features)")
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

    print(f"\n[OK] Selected top {n_features} features out of 168")
    print(f"[INFO] Feature score range: {feature_scores[selected_indices].min():.2f} - {feature_scores[selected_indices].max():.2f}")
    print(f"[INFO] Selected feature indices: {list(selected_indices[:10])}... (showing first 10)")

    # Show which module features were selected
    kcda_count = sum(1 for i in selected_indices if i < 48)
    tda_count = sum(1 for i in selected_indices if 48 <= i < 112)
    fractal_count = sum(1 for i in selected_indices if 112 <= i < 144)
    ergodic_count = sum(1 for i in selected_indices if i >= 144)

    print(f"\n[DISTRIBUTION] Features per module:")
    print(f"  KCDA:    {kcda_count}/{48} ({kcda_count/48*100:.1f}%)")
    print(f"  TDA:     {tda_count}/{64} ({tda_count/64*100:.1f}%)")
    print(f"  Fractal: {fractal_count}/{32} ({fractal_count/32*100:.1f}%)")
    print(f"  Ergodic: {ergodic_count}/{24} ({ergodic_count/24*100:.1f}%)")

    return X_train_selected, X_val_selected, selector


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

    # 2. Extract features
    print("\n[STEP 2] Extracting features...")
    X_train = extract_features(train_texts, "training samples")
    X_val = extract_features(val_texts, "validation samples") if len(val_texts) > 0 else np.array([])
    X_test = extract_features(test_texts, "test samples") if len(test_texts) > 0 else np.array([])

    y_train = np.array(train_labels)
    y_val = np.array(val_labels) if len(val_labels) > 0 else np.array([])
    y_test = np.array(test_labels) if len(test_labels) > 0 else np.array([])

    # Save features
    np.save("data/X_train.npy", X_train)
    np.save("data/y_train.npy", y_train)
    if len(X_val) > 0:
        np.save("data/X_val.npy", X_val)
        np.save("data/y_val.npy", y_val)
    if len(X_test) > 0:
        np.save("data/X_test.npy", X_test)
        np.save("data/y_test.npy", y_test)

    print(f"\n[OK] Features saved to data/ directory")
    print(f"  Training: {X_train.shape}")
    if len(X_val) > 0:
        print(f"  Validation: {X_val.shape}")
    if len(X_test) > 0:
        print(f"  Test: {X_test.shape}")

    # 2.5. Feature Selection (reduce from 168 to 80 features)
    print("\n[STEP 2.5] Performing feature selection...")
    X_train_selected, X_val_selected, feature_selector = perform_feature_selection(
        X_train, y_train, X_val, n_features=80
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
