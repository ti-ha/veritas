"""
Merge diverse Claude-generated samples with existing samples.

This script combines the diverse samples from Claude API with
the existing template-based samples to create a mixed dataset.
"""

import json
from pathlib import Path


def main():
    print("="*60)
    print("MERGE AI SAMPLES")
    print("="*60)

    # Load files
    diverse_file = Path("data/ai_samples_diverse_claude.json")
    existing_file = Path("data/ai_samples.json")
    output_file = Path("data/ai_samples_merged.json")

    if not diverse_file.exists():
        print(f"\n[ERROR] {diverse_file} not found!")
        print("Run generate_diverse_ai_with_claude.py first")
        return

    if not existing_file.exists():
        print(f"\n[ERROR] {existing_file} not found!")
        return

    with open(diverse_file, 'r', encoding='utf-8') as f:
        diverse_samples = json.load(f)

    with open(existing_file, 'r', encoding='utf-8') as f:
        existing_samples = json.load(f)

    print(f"\n[LOAD] Diverse samples: {len(diverse_samples)}")
    print(f"[LOAD] Existing samples: {len(existing_samples)}")

    # Merge strategy
    print(f"\nMerge strategy:")
    print(f"1. Replace all with diverse (use only Claude samples)")
    print(f"2. Replace N templated with diverse (keep some templates)")
    print(f"3. Append diverse to existing (keep all)")

    choice = input("\nSelect (1/2/3): ")

    if choice == '1':
        merged = diverse_samples
        print(f"\n[MERGE] Using only diverse samples: {len(merged)}")

    elif choice == '2':
        # Replace first N templated samples with diverse
        n_replace = len(diverse_samples)
        merged = diverse_samples + existing_samples[n_replace:]
        print(f"\n[MERGE] Replaced first {n_replace} samples")
        print(f"[MERGE] Kept {len(existing_samples) - n_replace} template samples")
        print(f"[MERGE] Total: {len(merged)} samples")

    elif choice == '3':
        merged = existing_samples + diverse_samples
        print(f"\n[MERGE] Combined all samples: {len(merged)}")

    else:
        print("[CANCELLED] Invalid choice")
        return

    # Save
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f"\n[SAVE] Saved merged dataset to {output_file}")
    print(f"\nTo use this dataset:")
    print(f"  1. Backup current: copy data\\ai_samples.json data\\ai_samples.json.bak")
    print(f"  2. Replace: move {output_file} data\\ai_samples.json")
    print("="*60)


if __name__ == "__main__":
    main()
