#!/usr/bin/env python3
"""
Test the fixed Section Predictor with murder case.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

def test_murder_case():
    """Test murder case prediction."""
    try:
        from src.legal_sections.section_predictor import LegalSectionPredictor
        
        # Initialize predictor
        predictor = LegalSectionPredictor()
        
        # Test case
        test_query = "Analyze a murder case under Section 302 IPC"
        
        print("🧪 TESTING SECTION PREDICTOR FIX")
        print("=" * 50)
        print(f"Query: {test_query}")
        print()
        
        # Predict sections
        result = predictor.predict_sections(test_query)
        
        print("📊 RESULTS:")
        print(f"Case Category: {result.case_category}")
        print(f"Severity: {result.severity_assessment}")
        print()
        
        if result.primary_sections:
            print("🎯 Primary Sections:")
            for pred in result.primary_sections:
                print(f"  • Section {pred.section.section_number} {pred.section.code.value}: {pred.section.title}")
                print(f"    Confidence: {pred.confidence:.2f}")
                print(f"    Reasoning: {pred.reasoning}")
                print()
        else:
            print("❌ No sections predicted")
        
        print(f"Analysis: {result.overall_analysis}")
        
        # Check if fix worked
        if result.primary_sections and result.primary_sections[0].section.section_number == "302":
            print("\n✅ FIX SUCCESSFUL: Section 302 (Murder) correctly identified!")
            print(f"✅ Severity correctly assessed as: {result.severity_assessment}")
            print(f"✅ Case category: {result.case_category}")
        else:
            print("\n❌ FIX FAILED: Section 302 not identified")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_murder_case()
