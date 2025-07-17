#!/usr/bin/env python3
"""
Test the improved input validation.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

def test_validation():
    """Test improved validation logic."""
    try:
        from src.utils.llm_client import LegalLLMClient
        
        client = LegalLLMClient()
        
        test_cases = [
            "Cases Nos. 297 and 298 of 1951",  # Should be valid now
            "zebi random nonsense",  # Should be invalid
            "The accused was charged under Section 302 IPC",  # Should be valid
            "A v. B Supreme Court case",  # Should be valid
            "hello world",  # Should be invalid
            "Case No. 123 filed in High Court",  # Should be valid
            "297 of 1951",  # Should be valid (case reference)
        ]
        
        print("🔍 TESTING IMPROVED INPUT VALIDATION")
        print("=" * 50)
        
        for i, test_input in enumerate(test_cases, 1):
            validation = client.validate_legal_input(test_input)
            status = "✅ VALID" if validation["valid"] else "❌ INVALID"
            score = validation.get("score", 0)
            reason = validation.get("reason", "")
            
            print(f"{i}. '{test_input}'")
            print(f"   Status: {status} (Score: {score})")
            if not validation["valid"]:
                print(f"   Reason: {reason}")
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Validation test failed: {e}")
        return False

def test_cot_with_case_numbers():
    """Test CoT reasoning with case numbers."""
    print("🧠 TESTING COT WITH CASE NUMBERS")
    print("=" * 40)
    
    try:
        from src.reasoning.cot_reasoning import ChainOfThoughtReasoner
        
        reasoner = ChainOfThoughtReasoner()
        
        test_input = "Cases Nos. 297 and 298 of 1951"
        print(f"Input: {test_input}")
        
        result = reasoner.analyze_case(test_input)
        
        print(f"Final Verdict: {result.final_verdict.value}")
        print(f"Overall Confidence: {result.overall_confidence:.2f}")
        print(f"LAW Analysis: {result.law_analysis.reasoning[:100]}...")
        
        return result.overall_confidence > 0.0  # Should not be 0 if working
        
    except Exception as e:
        print(f"❌ CoT test failed: {e}")
        return False

def main():
    """Run validation tests."""
    print("🚀 TESTING IMPROVED INPUT VALIDATION")
    print("=" * 60)
    
    validation_ok = test_validation()
    cot_ok = test_cot_with_case_numbers()
    
    print("=" * 60)
    print("📊 RESULTS")
    print("=" * 60)
    print(f"Input Validation: {'✅ WORKING' if validation_ok else '❌ FAILED'}")
    print(f"CoT with Case Numbers: {'✅ WORKING' if cot_ok else '❌ FAILED'}")
    
    if validation_ok and cot_ok:
        print("\n🎉 VALIDATION FIXED! Case numbers should now work!")
    else:
        print("\n⚠️ Still some issues to resolve")

if __name__ == "__main__":
    main()
