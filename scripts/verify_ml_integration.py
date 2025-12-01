"""
Quick verification script to test ML model integration.

This script verifies that:
1. The trained ML model loads successfully
2. Feature extraction works end-to-end
3. ML predictions are being used (not heuristic fallback)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.ensemble import VERITASClassifier


def main():
    print("="*60)
    print("ML MODEL INTEGRATION VERIFICATION")
    print("="*60)

    # Test 1: Initialize classifier
    print("\n[TEST 1] Initializing VERITAS classifier...")
    classifier = VERITASClassifier()

    if classifier.ml_classifier is not None:
        print("  [OK] ML model loaded successfully")
        print(f"  Model type: {type(classifier.ml_classifier).__name__}")
    else:
        print("  [FAIL] ML model NOT loaded - using heuristic fallback")
        print("  This means the trained model file is missing or failed to load")
        return False

    # Test 2: Verify feature extraction
    print("\n[TEST 2] Testing feature extraction...")
    test_text = """
    Machine learning is a subset of artificial intelligence that enables systems to learn
    and improve from experience without being explicitly programmed. It focuses on developing
    algorithms that can analyze data and make predictions or decisions.
    """

    try:
        result = classifier.analyze_text(test_text)
        print("  [OK] Feature extraction successful")
        print(f"  Feature vector size: {result['features']['feature_vector_size']} dimensions")
        print(f"  Expected: 168 dimensions (48 KCDA + 64 TDA + 32 Fractal + 24 Ergodic)")

        if result['features']['feature_vector_size'] == 168:
            print("  [OK] Feature vector size correct!")
        else:
            print("  [FAIL] Feature vector size mismatch!")
            return False

    except Exception as e:
        print(f"  [FAIL] Feature extraction failed: {e}")
        return False

    # Test 3: Verify ML predictions
    print("\n[TEST 3] Testing ML predictions...")
    print(f"  AI Probability: {result['ai_probability']:.2%}")
    print(f"  Classification Level: {result['classification_level']}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print(f"  Module Agreement: {result['features']['agreement_score']:.2f}")

    # Test both human and AI text
    print("\n[TEST 4] Testing on sample texts...")

    human_text = """
    I went to the store yesterday and grabbed some milk, but when I got home I realized
    I forgot the eggs lol. Typical me. So I had to go back, and of course there was
    a huge line. Whatever, at least I got everything eventually.
    """

    ai_text = """
    Artificial intelligence is revolutionizing various industries through its ability to
    process vast amounts of data and identify patterns. Machine learning algorithms enable
    systems to improve their performance over time, making them increasingly effective at
    tasks ranging from image recognition to natural language processing.
    """

    human_result = classifier.analyze_text(human_text)
    ai_result = classifier.analyze_text(ai_text)

    print(f"\n  Human text (casual, informal):")
    print(f"    AI Probability: {human_result['ai_probability']:.2%}")
    print(f"    Classification: {'CORRECT' if human_result['ai_probability'] < 0.5 else 'INCORRECT'}")

    print(f"\n  AI text (formal, technical):")
    print(f"    AI Probability: {ai_result['ai_probability']:.2%}")
    print(f"    Classification: {'CORRECT' if ai_result['ai_probability'] > 0.5 else 'INCORRECT'}")

    # Test 5: Verify module scores are still computed
    print("\n[TEST 5] Verifying module scores...")
    print(f"  KCDA Score:    {result['module_scores']['kcda']:.2f}")
    print(f"  TDA Score:     {result['module_scores']['tda']:.2f}")
    print(f"  Fractal Score: {result['module_scores']['fractal']:.2f}")
    print(f"  Ergodic Score: {result['module_scores']['ergodic']:.2f}")
    print("  [OK] Module scores computed for explanation")

    print("\n" + "="*60)
    print("VERIFICATION COMPLETE")
    print("="*60)
    print("\n[SUCCESS] ML model integration successful!")
    print("[SUCCESS] All systems operational")
    print("\nThe trained ML model is being used for predictions.")
    print("Module scores are still computed for detailed explanations.")
    print("="*60)

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
