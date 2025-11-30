"""
Demo script for VERITAS AI Detection System
"""
import requests
import json


def test_veritas(text, description):
    """Test VERITAS with a text sample"""
    print(f"\n{'='*80}")
    print(f"Testing: {description}")
    print(f"{'='*80}")
    print(f"Text (first 200 chars): {text[:200]}...")
    print()

    try:
        response = requests.post(
            "http://localhost:8000/api/detect",
            json={"text": text},
            timeout=10
        )

        if response.status_code == 200:
            result = response.json()

            # Display results
            classification_levels = ["DEFINITIVE", "PROBABLE", "INCONCLUSIVE"]
            level_text = classification_levels[result['classification_level'] - 1]

            print(f"Classification Level: {level_text}")
            print(f"AI Probability: {result['ai_probability']*100:.1f}%")
            print(f"Confidence: {result['confidence']*100:.1f}%")
            print(f"Processing Time: {result['processing_time_ms']:.0f}ms")
            print()
            print(f"Explanation: {result['explanation']}")
            print()
            print("Module Scores:")
            for module, score in result['module_scores'].items():
                print(f"  - {module.upper()}: {score*100:.1f}%")

        else:
            print(f"Error: {response.status_code}")
            print(response.text)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("VERITAS - Next-Generation AI Detection System Demo")
    print("="*80)

    # Test 1: Likely human text (natural, conversational, with imperfections)
    human_text = """
    So I've been thinking about this whole AI detection thing lately, and honestly?
    It's way more complicated than I initially thought. Like, when you first hear
    about it, you're like "oh yeah, just check if it sounds robotic" - but that's
    not it at all. The real challenge is that AI is getting SO good now that it can
    mimic human writing pretty convincingly. But here's the thing - there are still
    these subtle patterns, y'know? Like the way information is structured, or how
    ideas connect across a text. AI tends to be more... consistent? I guess that's
    the word. Whereas humans are all over the place with our writing - sometimes
    verbose, sometimes concise, jumping between ideas in weird ways. Anyway, that's
    my two cents on it. What do you think?
    """

    # Test 2: Likely AI text (formal, structured, comprehensive)
    ai_text = """
    Artificial intelligence represents a transformative technological advancement with
    far-reaching implications across multiple sectors. The field encompasses various
    methodologies including machine learning, neural networks, and natural language
    processing. These technologies enable systems to analyze data, recognize patterns,
    and make informed decisions with minimal human intervention. In the healthcare
    sector, AI applications demonstrate significant potential for improving diagnostic
    accuracy and treatment optimization. Financial institutions leverage AI algorithms
    to detect fraudulent activities and assess risk profiles. The education sector
    benefits from personalized learning platforms that adapt to individual student needs.
    As AI continues to evolve, ongoing research focuses on addressing challenges related
    to algorithmic bias, data privacy, and ethical considerations in automated
    decision-making processes.
    """

    # Test 3: Mixed/ambiguous text
    mixed_text = """
    The implementation of AI detection systems requires careful consideration of
    multiple factors. First, you need to understand the underlying mathematical
    frameworks - stuff like Kolmogorov complexity and topological analysis. But
    honestly, that's just the technical side. The real challenge? Making sure the
    system is actually fair and doesn't discriminate against non-native speakers
    or people with different writing styles. I've seen too many tools that just
    flag anything formal as "AI-generated" which is ridiculous. Academic writing
    is supposed to be formal! Anyway, the key is using multiple analysis methods
    simultaneously. This provides robustness and helps avoid false positives.
    """

    test_veritas(human_text, "Conversational Human Text")
    test_veritas(ai_text, "Formal AI-Generated Text")
    test_veritas(mixed_text, "Mixed/Ambiguous Text")

    print("\n" + "="*80)
    print("Demo Complete! The web interface is available at http://localhost:8000")
    print("="*80 + "\n")
