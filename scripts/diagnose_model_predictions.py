"""
Diagnostic script to understand model predictions.

Analyzes a sample of test data to see what the model predicts.
"""

import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.ensemble import VERITASClassifier


def main():
    print("="*60)
    print("MODEL PREDICTION DIAGNOSIS")
    print("="*60)

    # Load classifier
    print("\n[1] Loading classifier...")
    classifier = VERITASClassifier()

    if classifier.ml_classifier is None:
        print("[ERROR] No ML model loaded!")
        return

    # Load test samples
    print("[2] Loading test samples...")
    test_file = Path("data/test_samples.json")
    with open(test_file, 'r', encoding='utf-8') as f:
        test_samples = json.load(f)

    # Sample 10 human and 10 AI
    human_samples = [s for s in test_samples if s['label'] == 'human'][:10]
    ai_samples = [s for s in test_samples if s['label'] == 'ai'][:10]

    print(f"[3] Testing on {len(human_samples)} human + {len(ai_samples)} AI samples\n")

    # Test human samples
    print("="*60)
    print("HUMAN SAMPLES (should predict LOW AI probability)")
    print("="*60)

    human_correct = 0
    for i, sample in enumerate(human_samples, 1):
        result = classifier.analyze_text(sample['text'])
        ai_prob = result['ai_probability']
        prediction = 'AI' if ai_prob > 0.5 else 'HUMAN'
        correct = prediction == 'HUMAN'

        if correct:
            human_correct += 1

        status = '[OK]' if correct else '[FAIL]'
        print(f"{status} Sample {i}: {ai_prob:.1%} AI - predicted {prediction}")
        print(f"     Source: {sample['source']}")
        print(f"     Module scores: KCDA={result['module_scores']['kcda']:.2f}, "
              f"TDA={result['module_scores']['tda']:.2f}, "
              f"Fractal={result['module_scores']['fractal']:.2f}, "
              f"Ergodic={result['module_scores']['ergodic']:.2f}")
        print()

    # Test AI samples
    print("="*60)
    print("AI SAMPLES (should predict HIGH AI probability)")
    print("="*60)

    ai_correct = 0
    for i, sample in enumerate(ai_samples, 1):
        result = classifier.analyze_text(sample['text'])
        ai_prob = result['ai_probability']
        prediction = 'AI' if ai_prob > 0.5 else 'HUMAN'
        correct = prediction == 'AI'

        if correct:
            ai_correct += 1

        status = '[OK]' if correct else '[FAIL]'
        print(f"{status} Sample {i}: {ai_prob:.1%} AI - predicted {prediction}")
        print(f"     Source: {sample['source']}")
        print(f"     Module scores: KCDA={result['module_scores']['kcda']:.2f}, "
              f"TDA={result['module_scores']['tda']:.2f}, "
              f"Fractal={result['module_scores']['fractal']:.2f}, "
              f"Ergodic={result['module_scores']['ergodic']:.2f}")
        print()

    # Summary
    print("="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Human accuracy: {human_correct}/{len(human_samples)} = {human_correct/len(human_samples):.1%}")
    print(f"AI accuracy:    {ai_correct}/{len(ai_samples)} = {ai_correct/len(ai_samples):.1%}")
    print(f"Overall:        {(human_correct + ai_correct)}/{len(human_samples) + len(ai_samples)} = "
          f"{(human_correct + ai_correct)/(len(human_samples) + len(ai_samples)):.1%}")
    print("="*60)

    if ai_correct == 0:
        print("\n[DIAGNOSIS] Model predicts HUMAN for all samples!")
        print("This indicates:")
        print("  1. Features are not discriminative enough")
        print("  2. Model may need more training data")
        print("  3. Hyperparameters may need tuning")
        print("  4. Feature engineering approach may need rethinking")


if __name__ == "__main__":
    main()
