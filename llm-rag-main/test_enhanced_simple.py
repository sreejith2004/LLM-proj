#!/usr/bin/env python3
"""
Simple test for enhanced components.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

def test_ollama_connection():
    """Test if Ollama is available."""
    try:
        from src.utils.llm_client import OllamaClient

        client = OllamaClient()
        available = client.is_available()

        print("🔗 OLLAMA CONNECTION TEST")
        print("=" * 40)
        print(f"Ollama Available: {'✅ Yes' if available else '❌ No'}")

        if available:
            models = client.get_available_models()
            print(f"Available Models: {models}")

            # Test simple generation
            response = client.generate("Hello, this is a test.", max_tokens=50)
            print(f"Test Generation: {'✅ Success' if response.success else '❌ Failed'}")
            if response.success:
                print(f"Response: {response.content[:100]}...")

        return available

    except Exception as e:
        print(f"❌ Ollama test failed: {e}")
        return False

def test_section_predictor():
    """Test the fixed section predictor."""
    print("\n⚖️ SECTION PREDICTOR TEST")
    print("=" * 40)

    try:
        from src.legal_sections.section_predictor import LegalSectionPredictor

        predictor = LegalSectionPredictor()

        test_case = "Analyze a murder case under Section 302 IPC"
        print(f"Input: {test_case}")

        result = predictor.predict_sections(test_case)

        print(f"Case Category: {result.case_category}")
        print(f"Severity: {result.severity_assessment}")
        print(f"Primary Sections: {len(result.primary_sections)}")

        if result.primary_sections:
            for pred in result.primary_sections[:2]:
                print(f"  • Section {pred.section.section_number}: {pred.section.title}")
                print(f"    Confidence: {pred.confidence:.2f}")

        return True

    except Exception as e:
        print(f"❌ Section predictor test failed: {e}")
        return False

def test_enhanced_cot_reasoning():
    """Test enhanced Chain-of-Thought reasoning."""
    print("\n🧠 ENHANCED CHAIN-OF-THOUGHT TEST")
    print("=" * 50)

    try:
        from src.reasoning.cot_reasoning import ChainOfThoughtReasoner

        reasoner = ChainOfThoughtReasoner()

        # Test with valid legal case
        legal_case = "The accused was charged under Section 302 IPC for murder. The prosecution argued sufficient evidence."

        print(f"Input: {legal_case}")

        result = reasoner.analyze_case(legal_case)

        print(f"Final Verdict: {result.final_verdict.value}")
        print(f"Overall Confidence: {result.overall_confidence:.2f}")
        print(f"LAW Step: {result.law_analysis.reasoning[:100]}...")

        # Test with invalid input
        print("\n--- Testing Invalid Input ---")
        invalid_result = reasoner.analyze_case("zebi random nonsense")
        print(f"Invalid Input Verdict: {invalid_result.final_verdict.value}")
        print(f"Invalid Input Confidence: {invalid_result.overall_confidence:.2f}")

        return True

    except Exception as e:
        print(f"❌ CoT test failed: {e}")
        return False

def main():
    """Run enhanced component tests."""
    print("🚀 ENHANCED LEGAL AI SYSTEM - SIMPLE TEST")
    print("=" * 60)

    results = {
        "Ollama Connection": test_ollama_connection(),
        "Section Predictor": test_section_predictor(),
        "Enhanced CoT Reasoning": test_enhanced_cot_reasoning()
    }

    print("\n" + "=" * 60)
    print("📊 FINAL RESULTS")
    print("=" * 60)

    for component, success in results.items():
        status = "✅ WORKING" if success else "❌ FAILED"
        print(f"{component}: {status}")

    working_count = sum(results.values())
    total_count = len(results)

    print(f"\n🎯 SUMMARY: {working_count}/{total_count} components working")

    if working_count == total_count:
        print("🎉 ALL COMPONENTS ARE NOW REALLY WORKING!")
    elif working_count >= 2:
        print("✅ Most components working - system is functional!")
    else:
        print("⚠️ Several components need attention")

if __name__ == "__main__":
    main()
