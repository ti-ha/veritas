"""
Prepare Training Data for VERITAS

Merges human and AI samples, performs quality checks, creates train/val/test splits,
and prepares the final dataset for ML training.

Usage:
    python scripts/prepare_training_data.py

This will:
1. Load human samples from data/human_samples.json
2. Load AI samples from data/ai_samples.json
3. Quality check and filter samples
4. Balance the dataset
5. Create train/validation/test splits
6. Save to data/test_samples.json (for backwards compatibility)
7. Also save separate train/val/test files
"""

import json
from pathlib import Path
from typing import List, Dict
import random


def load_samples(file_path: str) -> List[Dict]:
    """Load samples from JSON file."""
    path = Path(file_path)
    if not path.exists():
        print(f"[!] File not found: {file_path}")
        return []

    with open(path, 'r', encoding='utf-8') as f:
        samples = json.load(f)

    if isinstance(samples, dict):
        # Handle old format
        samples = [samples]

    print(f"[OK] Loaded {len(samples)} samples from {file_path}")
    return samples


def quality_check(sample: Dict) -> bool:
    """Check if sample meets quality criteria."""
    text = sample.get('text', '')

    # Length check
    if len(text) < 100:
        return False

    # Word count check
    word_count = len(text.split())
    if word_count < 20:
        return False

    # Must have label
    if 'label' not in sample:
        return False

    # Label must be valid
    if sample['label'] not in ['human', 'ai']:
        return False

    return True


def deduplicate(samples: List[Dict]) -> List[Dict]:
    """Remove duplicate samples based on text hash."""
    seen = set()
    unique = []

    for sample in samples:
        text_hash = hash(sample['text'][:200])  # Use first 200 chars for hash

        if text_hash not in seen:
            seen.add(text_hash)
            unique.append(sample)

    print(f"[OK] Removed {len(samples) - len(unique)} duplicates")
    return unique


def balance_dataset(samples: List[Dict], target_per_class: int = None) -> List[Dict]:
    """Balance the dataset to have equal human and AI samples."""
    human = [s for s in samples if s['label'] == 'human']
    ai = [s for s in samples if s['label'] == 'ai']

    print(f"[INFO] Class distribution before balancing:")
    print(f"  Human: {len(human)}")
    print(f"  AI: {len(ai)}")

    # Determine target size
    if target_per_class is None:
        target_per_class = min(len(human), len(ai))

    # Sample from each class
    if len(human) > target_per_class:
        human = random.sample(human, target_per_class)

    if len(ai) > target_per_class:
        ai = random.sample(ai, target_per_class)

    balanced = human + ai
    random.shuffle(balanced)

    print(f"[OK] Balanced dataset:")
    print(f"  Human: {len([s for s in balanced if s['label'] == 'human'])}")
    print(f"  AI: {len([s for s in balanced if s['label'] == 'ai'])}")
    print(f"  Total: {len(balanced)}")

    return balanced


def create_splits(samples: List[Dict], train_ratio: float = 0.7, val_ratio: float = 0.15):
    """Create train/validation/test splits."""
    random.shuffle(samples)

    total = len(samples)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)

    train = samples[:train_size]
    val = samples[train_size:train_size + val_size]
    test = samples[train_size + val_size:]

    print(f"[OK] Created splits:")
    print(f"  Train: {len(train)} ({len([s for s in train if s['label'] == 'human'])} human, {len([s for s in train if s['label'] == 'ai'])} AI)")
    print(f"  Val:   {len(val)} ({len([s for s in val if s['label'] == 'human'])} human, {len([s for s in val if s['label'] == 'ai'])} AI)")
    print(f"  Test:  {len(test)} ({len([s for s in test if s['label'] == 'human'])} human, {len([s for s in test if s['label'] == 'ai'])} AI)")

    return train, val, test


def save_samples(samples: List[Dict], file_path: str):
    """Save samples to JSON file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print(f"[OK] Saved {len(samples)} samples to {file_path}")


def print_statistics(samples: List[Dict], name: str = "Dataset"):
    """Print dataset statistics."""
    print(f"\n[STATS] {name} Statistics:")
    print(f"  Total samples: {len(samples)}")

    # By label
    human = [s for s in samples if s['label'] == 'human']
    ai = [s for s in samples if s['label'] == 'ai']
    print(f"  Human: {len(human)} ({len(human)/len(samples)*100:.1f}%)")
    print(f"  AI: {len(ai)} ({len(ai)/len(samples)*100:.1f}%)")

    # By source
    sources = {}
    for sample in samples:
        source = sample.get('source', 'unknown')
        sources[source] = sources.get(source, 0) + 1

    print(f"  Sources:")
    for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
        print(f"    {source}: {count}")

    # Text lengths
    lengths = [len(s['text']) for s in samples]
    print(f"  Text length (chars):")
    print(f"    Min: {min(lengths)}")
    print(f"    Max: {max(lengths)}")
    print(f"    Mean: {sum(lengths)/len(lengths):.0f}")

    # Word counts
    word_counts = [s.get('word_count', len(s['text'].split())) for s in samples]
    print(f"  Word count:")
    print(f"    Min: {min(word_counts)}")
    print(f"    Max: {max(word_counts)}")
    print(f"    Mean: {sum(word_counts)/len(word_counts):.0f}")


def main():
    print("="*60)
    print("VERITAS Training Data Preparation")
    print("="*60)

    # Load samples
    print("\n1. Loading samples...")
    human_samples = load_samples("data/human_samples.json")
    ai_samples = load_samples("data/ai_samples.json")

    # Combine
    all_samples = human_samples + ai_samples
    print(f"\n[OK] Combined: {len(all_samples)} total samples")

    # Quality check
    print("\n2. Quality checking...")
    before = len(all_samples)
    all_samples = [s for s in all_samples if quality_check(s)]
    after = len(all_samples)
    print(f"[OK] Passed quality check: {after}/{before} samples ({after/before*100:.1f}%)")

    # Deduplicate
    print("\n3. Deduplicating...")
    all_samples = deduplicate(all_samples)

    # Balance
    print("\n4. Balancing classes...")
    balanced = balance_dataset(all_samples)

    # Statistics before split
    print_statistics(balanced, "Balanced Dataset")

    # Create splits
    print("\n5. Creating train/val/test splits...")
    train, val, test = create_splits(balanced, train_ratio=0.7, val_ratio=0.15)

    # Save splits
    print("\n6. Saving splits...")
    save_samples(train, "data/train_samples.json")
    save_samples(val, "data/val_samples.json")
    save_samples(test, "data/test_samples.json")

    # Also save combined for backwards compatibility
    combined = train + val + test
    save_samples(combined, "data/test_samples.json")  # Old name for compatibility

    # Final statistics
    print_statistics(train, "Training Set")
    print_statistics(val, "Validation Set")
    print_statistics(test, "Test Set")

    print("\n" + "="*60)
    print("Data Preparation Complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run: python scripts/train_ml_classifier.py")
    print("2. This will train on the new balanced dataset")
    print("3. Then run: pytest tests/test_validation.py -v")
    print("="*60)


if __name__ == "__main__":
    main()
