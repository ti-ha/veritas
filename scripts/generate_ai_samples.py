"""
Generate diverse AI samples using Claude API.

This script uses the Anthropic Claude API to generate genuinely varied
AI responses to prompts from the existing ai_samples.json file.

Setup:
    1. Install: pip install anthropic
    2. Set environment variable: set ANTHROPIC_API_KEY=your_key_here
    3. Run: python scripts/generate_diverse_ai_with_claude.py

The API key is read from environment variable only - never stored in repo.
"""

import json
import os
import time
from pathlib import Path
from datetime import datetime
from anthropic import Anthropic


def generate_with_claude(client, prompt: str, max_retries: int = 3) -> str:
    """Generate AI response using Claude API."""

    system_prompt = """You are generating training data for an AI detection system.
Write natural, varied responses to prompts. Use different styles, tones, and structures.
Sometimes be concise, sometimes elaborate. Vary your approach - be conversational,
academic, creative, technical, or casual as appropriate. Write like a helpful AI assistant
would naturally respond."""

    for attempt in range(max_retries):
        try:
            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                temperature=0.8,  # Higher temperature for more variety
                system=system_prompt,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            return message.content[0].text

        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                print(f"  [!] Error: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"  [!] Failed after {max_retries} attempts: {e}")
                return None


def main():
    print("="*60)
    print("DIVERSE AI SAMPLE GENERATION WITH CLAUDE")
    print("="*60)

    # Check for API key
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("\n[ERROR] ANTHROPIC_API_KEY environment variable not set!")
        print("\nPlease set it with:")
        print("  Windows: set ANTHROPIC_API_KEY=your_key_here")
        print("  Linux/Mac: export ANTHROPIC_API_KEY=your_key_here")
        print("\nOr set it permanently in your system environment variables.")
        return

    print(f"[OK] API key found (length: {len(api_key)})")

    # Initialize Claude client
    client = Anthropic(api_key=api_key)

    # Load existing samples to get prompts
    input_file = Path("data/ai_samples.json")
    if not input_file.exists():
        print(f"\n[ERROR] {input_file} not found!")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        existing_samples = json.load(f)

    print(f"[LOAD] Found {len(existing_samples)} existing samples")

    # Ask how many to generate
    print(f"\nHow many diverse samples to generate?")
    print(f"  Recommended: 500-1000 for good variety")
    print(f"  Note: Claude API costs ~$0.003 per request")

    try:
        num_samples = int(input("Enter number (or press Enter for 500): ") or "500")
    except ValueError:
        num_samples = 500

    print(f"\n[TARGET] Generating {num_samples} diverse samples")
    print(f"[COST] Estimated cost: ${num_samples * 0.003:.2f}")

    confirm = input("\nProceed? (y/n): ")
    if confirm.lower() != 'y':
        print("[CANCELLED]")
        return

    # Generate diverse samples
    diverse_samples = []

    # Use a variety of existing prompts
    import random
    selected_prompts = random.sample(existing_samples, min(num_samples, len(existing_samples)))

    # Output file setup
    output_file = Path("data/ai_samples_diverse_claude.json")

    # Load existing diverse samples if resuming
    if output_file.exists():
        with open(output_file, 'r', encoding='utf-8') as f:
            diverse_samples = json.load(f)
        print(f"\n[RESUME] Found {len(diverse_samples)} existing diverse samples")
        print(f"[RESUME] Will generate {num_samples - len(diverse_samples)} more\n")

    print(f"\n[GENERATE] Starting generation...\n")

    for i, sample in enumerate(selected_prompts):
        # Skip if we already have enough samples
        if len(diverse_samples) >= num_samples:
            break

        prompt = sample['metadata']['prompt']

        print(f"[{len(diverse_samples)+1}/{num_samples}] {prompt[:60]}...")

        # Generate response
        response_text = generate_with_claude(client, prompt)

        if response_text:
            new_sample = {
                'id': f"claude_diverse_{len(diverse_samples)}_{hash(prompt)}",
                'text': response_text,
                'label': 'ai',
                'source': 'claude_sonnet_4',
                'generated_at': datetime.now().isoformat(),
                'word_count': len(response_text.split()),
                'metadata': {
                    'model': 'claude-sonnet-4-20250514',
                    'prompt': prompt,
                    'type': 'ai_generated'
                }
            }

            diverse_samples.append(new_sample)
            print(f"  [+] Generated ({len(response_text)} chars, {new_sample['word_count']} words)")

            # Save after each successful generation (fault tolerance)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(diverse_samples, f, indent=2, ensure_ascii=False)

        else:
            print(f"  [-] Failed to generate")

        # Rate limiting - be respectful to API
        if (len(diverse_samples)) % 10 == 0:
            print(f"\n[PROGRESS] {len(diverse_samples)}/{num_samples} completed")
            print(f"[SAVED] Progress saved to {output_file}")
            print("[SLEEP] Pausing 2s for rate limiting...\n")
            time.sleep(2)
        else:
            time.sleep(0.5)

    print("\n" + "="*60)
    print("GENERATION COMPLETE")
    print("="*60)
    print(f"[SAVE] Saved {len(diverse_samples)} samples to {output_file}")
    print(f"\nNext steps:")
    print(f"1. Review samples: open {output_file}")
    print(f"2. Merge with existing: python scripts/merge_ai_samples.py")
    print(f"3. Or replace existing: move {output_file} to data/ai_samples.json")
    print("="*60)


if __name__ == "__main__":
    main()
