#!/usr/bin/env python3
"""
Test Chain-of-Thought reasoning with different inputs.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

def test_cot_with_different_inputs():
    """Test CoT with various inputs to see if it's really working."""
    try:
        from src.reasoning.cot_reasoning import ChainOfThoughtReasoner
        
        # Initialize reasoner
        reasoner = ChainOfThoughtReasoner()
        
        test_cases = [
            "zebi",  # Nonsense input
            "",      # Empty input
            "The accused murdered the victim under Section 302 IPC",  # Valid legal case
            "Random text with no legal content",  # Non-legal text
            "Contract dispute between parties"  # Civil case
        ]
        
        print("🧪 TESTING CHAIN-OF-THOUGHT REASONING")
        print("=" * 60)
        
        for i, test_input in enumerate(test_cases, 1):
            print(f"\n{i}. Testing input: '{test_input}'")
            print("-" * 40)
            
            try:
                result = reasoner.analyze_case(test_input)
                
                print(f"Final Verdict: {result.final_verdict.value}")
                print(f"Confidence: {result.overall_confidence:.2f}")
                print(f"LAW Step: {result.law_step.reasoning[:100]}...")
                print(f"FACT Step: {result.fact_step.reasoning[:100]}...")
                
                # Check if responses are identical (indicating static responses)
                if i > 1:
                    if (result.final_verdict == prev_result.final_verdict and 
                        result.overall_confidence == prev_result.overall_confidence):
                        print("⚠️  WARNING: Identical response to previous test!")
                
                prev_result = result
                
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print("\n" + "=" * 60)
        print("🎯 CONCLUSION:")
        print("If all responses are similar/identical, the system is using fallbacks!")
        
    except Exception as e:
        print(f"❌ Import Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_cot_with_different_inputs()
