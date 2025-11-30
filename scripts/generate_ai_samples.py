"""
AI Text Generator for VERITAS Training Dataset

Generates AI text samples using various free/public AI APIs for training data.

Sources:
- HuggingFace Inference API (free tier)
- Various open models available via API
- Different prompts and styles to create diversity

Usage:
    python scripts/generate_ai_samples.py [--max-samples 500] [--output data/ai_samples.json]

Note: You may need to set API keys as environment variables for some services.
"""

import json
import time
import argparse
import requests
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import random
import os


class AITextGenerator:
    """Generates diverse AI text samples for training."""

    def __init__(self, output_file: str = "data/ai_samples.json", max_samples: int = 500):
        self.output_file = Path(output_file)
        self.max_samples = max_samples
        self.samples = []
        self.collected_ids = set()

        # Load existing samples
        if self.output_file.exists():
            with open(self.output_file, 'r', encoding='utf-8') as f:
                existing = json.load(f)
                self.samples = existing if isinstance(existing, list) else []
                self.collected_ids = {s.get('id', '') for s in self.samples}
                print(f"[LOAD] Loaded {len(self.samples)} existing samples")

        # Diverse prompts for varied outputs
        self.prompts = [
            # Explanatory
            "Explain how photosynthesis works in plants.",
            "Describe the process of machine learning model training.",
            "What are the main causes of climate change?",
            "How does the internet work?",
            "Explain quantum computing to a beginner.",

            # Analytical
            "Analyze the impact of social media on society.",
            "Compare and contrast democracy and authoritarianism.",
            "Discuss the pros and cons of remote work.",
            "Evaluate the effectiveness of renewable energy sources.",
            "Analyze the themes in Shakespeare's Hamlet.",

            # Creative
            "Write a short story about a time traveler.",
            "Describe a futuristic city in the year 2100.",
            "Create a character description for a fantasy novel.",
            "Write a poem about autumn.",
            "Imagine a conversation between Albert Einstein and Nikola Tesla.",

            # Technical
            "Explain the difference between SQL and NoSQL databases.",
            "Describe how to implement a binary search algorithm.",
            "What are the key principles of object-oriented programming?",
            "Explain the concept of blockchain technology.",
            "How do neural networks learn?",

            # Opinion/Argumentative
            "Should artificial intelligence be regulated? Discuss.",
            "Is college education worth the cost?",
            "Discuss the ethics of genetic engineering.",
            "Should space exploration be a priority?",
            "Is social media beneficial or harmful?",

            # Instructional
            "How to bake a chocolate cake from scratch.",
            "Steps to learn a new programming language.",
            "Guide to starting a small business.",
            "How to improve your writing skills.",
            "Tips for effective time management.",

            # Descriptive
            "Describe the Grand Canyon.",
            "What is life like in Tokyo?",
            "Describe the experience of skydiving.",
            "What does it feel like to fall in love?",
            "Describe a thunderstorm in detail.",

            # Informative
            "What is the history of the internet?",
            "Explain the water cycle.",
            "How do vaccines work?",
            "What causes earthquakes?",
            "The history of the Roman Empire.",
        ]

    def save_samples(self):
        """Save samples to disk."""
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(self.samples, f, indent=2, ensure_ascii=False)
        print(f"[SAVE] Saved {len(self.samples)} samples to {self.output_file}")

    def add_sample(self, text: str, model: str, prompt: str):
        """Add a generated sample."""
        if len(text) < 100:  # Too short
            return False

        if len(text) > 10000:  # Truncate if too long
            text = text[:10000]

        sample_id = f"{model}_{hash(prompt)}_{hash(text[:100])}"

        if sample_id in self.collected_ids:
            return False

        sample = {
            'id': sample_id,
            'text': text,
            'label': 'ai',
            'source': model,
            'generated_at': datetime.now().isoformat(),
            'word_count': len(text.split()),
            'metadata': {
                'model': model,
                'prompt': prompt,
                'type': 'ai_generated'
            }
        }

        self.samples.append(sample)
        self.collected_ids.add(sample_id)
        return True

    def generate_huggingface(self, model_name: str = "gpt2", prompt: str = None) -> str:
        """
        Generate text using HuggingFace Inference API.

        Free tier available for many models. No API key needed for some public models.

        Models to try:
        - gpt2 (free, no auth)
        - EleutherAI/gpt-neo-1.3B
        - EleutherAI/gpt-j-6B
        - bigscience/bloom-560m
        """
        try:
            if prompt is None:
                prompt = random.choice(self.prompts)

            # HuggingFace Inference API
            url = f"https://api-inference.huggingface.co/models/{model_name}"

            # Check if HF_TOKEN is available (optional for many models)
            headers = {}
            hf_token = os.getenv('HF_TOKEN')
            if hf_token:
                headers['Authorization'] = f'Bearer {hf_token}'

            payload = {
                'inputs': prompt,
                'parameters': {
                    'max_length': 500,
                    'temperature': 0.8,
                    'top_p': 0.9,
                    'do_sample': True
                }
            }

            response = requests.post(url, headers=headers, json=payload, timeout=30)

            if response.status_code == 200:
                result = response.json()

                if isinstance(result, list) and len(result) > 0:
                    generated_text = result[0].get('generated_text', '')
                    return generated_text
                elif isinstance(result, dict):
                    return result.get('generated_text', '')

            elif response.status_code == 503:
                # Model loading
                print(f"  [!] Model {model_name} is loading, waiting...")
                time.sleep(20)
                return None

            else:
                print(f"  [!] HuggingFace API error {response.status_code}: {response.text[:200]}")
                return None

        except Exception as e:
            print(f"  [!] Generation error: {e}")
            return None

        return None

    def generate_local_simple(self, prompt: str = None) -> str:
        """
        Generate text using a simple Markov chain (offline backup).

        This is a fallback for when APIs are unavailable. Not sophisticated,
        but creates varied text patterns different from human writing.
        """
        if prompt is None:
            prompt = random.choice(self.prompts)

        # Simple template-based generation as fallback
        templates = [
            f"{prompt}\n\nThis is an important topic that requires careful consideration. "
            "There are several key aspects to examine. First, we must understand the fundamentals. "
            "The primary factors include various elements that interact in complex ways. "
            "Research has shown that multiple approaches can be effective. "
            "In conclusion, this subject demonstrates the complexity of modern challenges.",

            f"When considering {prompt.lower()}, it's essential to examine multiple perspectives. "
            "The conventional view suggests one approach, while alternative theories propose different solutions. "
            "Historical context provides valuable insights into current understanding. "
            "Contemporary research continues to refine our knowledge. "
            "Ultimately, comprehensive analysis reveals nuanced conclusions.",

            f"The question of {prompt.lower()} has fascinated researchers for decades. "
            "Early studies established foundational principles that remain relevant today. "
            "Modern advancements have expanded our capabilities significantly. "
            "Current methodologies incorporate both traditional and innovative techniques. "
            "Future developments promise even greater understanding and applications.",
        ]

        return random.choice(templates)

    def run(self):
        """Main generation loop."""
        print("="*60)
        print("VERITAS AI Text Generator")
        print("="*60)
        print(f"Target: {self.max_samples} samples")
        print(f"Current: {len(self.samples)} samples")
        print(f"Output: {self.output_file}")
        print("\nGenerating using:")
        print("  - HuggingFace models (gpt2, gpt-neo, etc.)")
        print("  - Diverse prompts across domains")
        print("\nNote: Set HF_TOKEN environment variable for better models")
        print("Press Ctrl+C to stop gracefully")
        print("="*60)

        # Models to try (start with free ones)
        models = [
            "gpt2",  # Always free
            "gpt2-medium",
            "EleutherAI/gpt-neo-125M",
            "distilgpt2",
        ]

        iteration = 0
        consecutive_failures = 0

        try:
            while len(self.samples) < self.max_samples:
                iteration += 1
                print(f"\n[Iteration {iteration}] Samples: {len(self.samples)}/{self.max_samples}")

                batch_collected = 0

                # Try different models and prompts
                for _ in range(5):  # 5 attempts per iteration
                    model = random.choice(models)
                    prompt = random.choice(self.prompts)

                    print(f"\n[Generate] Model: {model}")
                    print(f"           Prompt: {prompt[:60]}...")

                    generated_text = self.generate_huggingface(model, prompt)

                    if generated_text:
                        if self.add_sample(generated_text, model, prompt):
                            batch_collected += 1
                            consecutive_failures = 0
                            print(f"  [+] Generated sample ({len(generated_text)} chars)")
                        else:
                            print(f"  [-] Sample too short or duplicate")
                    else:
                        consecutive_failures += 1
                        print(f"  [!] Generation failed")

                    # If too many failures, use simple fallback
                    if consecutive_failures >= 3:
                        print(f"\n[FALLBACK] Using simple generation...")
                        fallback_text = self.generate_local_simple(prompt)
                        if self.add_sample(fallback_text, "simple_template", prompt):
                            batch_collected += 1
                            consecutive_failures = 0
                            print(f"  [+] Fallback sample generated")

                    # Sleep to respect API limits
                    time.sleep(2)

                # Save progress
                if batch_collected > 0:
                    self.save_samples()
                    print(f"\n[PROGRESS] Total: {len(self.samples)}/{self.max_samples} ({batch_collected} this iteration)")

                # Check if done
                if len(self.samples) >= self.max_samples:
                    print(f"\n[COMPLETE] Reached target of {self.max_samples} samples!")
                    break

                # Sleep between iterations
                sleep_time = 5
                print(f"\n[SLEEP] Waiting {sleep_time} seconds...")
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            print(f"\n\n[STOP] Interrupted by user")

        finally:
            self.save_samples()
            print(f"\n[FINAL] Generated {len(self.samples)} total samples")
            print(f"[FINAL] Saved to {self.output_file}")

            # Statistics
            models_used = {}
            for sample in self.samples:
                model = sample['metadata']['model']
                models_used[model] = models_used.get(model, 0) + 1

            print("\n[STATS] Samples by model:")
            for model, count in sorted(models_used.items(), key=lambda x: x[1], reverse=True):
                print(f"  {model}: {count}")


def main():
    parser = argparse.ArgumentParser(description='Generate AI text samples for VERITAS training')
    parser.add_argument('--max-samples', type=int, default=500, help='Maximum samples to generate')
    parser.add_argument('--output', type=str, default='data/ai_samples.json', help='Output file')

    args = parser.parse_args()

    generator = AITextGenerator(
        output_file=args.output,
        max_samples=args.max_samples
    )

    generator.run()


if __name__ == "__main__":
    main()
