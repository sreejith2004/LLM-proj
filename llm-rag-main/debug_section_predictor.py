#!/usr/bin/env python3
"""
Debug the Section Predictor step by step.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

def debug_prediction():
    """Debug prediction step by step."""
    try:
        from src.legal_sections.section_predictor import LegalSectionPredictor
        
        # Initialize predictor
        predictor = LegalSectionPredictor()
        
        # Test case
        test_query = "Analyze a murder case under Section 302 IPC"
        
        print("🔍 DEBUGGING SECTION PREDICTOR")
        print("=" * 50)
        print(f"Query: {test_query}")
        print()
        
        # Step 1: Extract keywords
        keywords = predictor._extract_keywords(test_query)
        print(f"1. Keywords extracted: {keywords}")
        
        # Step 2: Identify patterns
        patterns = predictor._identify_patterns(test_query)
        print(f"2. Patterns identified: {patterns}")
        
        # Step 3: Check if Section 302 exists in database
        section_302 = predictor.database.get_section("302")
        if section_302:
            print(f"3. Section 302 found in database: {section_302.title}")
            print(f"   Keywords: {section_302.keywords}")
        else:
            print("3. ❌ Section 302 NOT found in database")
        
        # Step 4: Test pattern matching manually
        if "murder" in keywords:
            print("4. ✅ 'murder' keyword found")
        else:
            print("4. ❌ 'murder' keyword NOT found")
            
        # Step 5: Check violence patterns
        if "violence_patterns" in patterns and "murder" in patterns["violence_patterns"]:
            print("5. ✅ Murder pattern identified")
        else:
            print("5. ❌ Murder pattern NOT identified")
            
        # Step 6: Test confidence calculation
        if section_302:
            confidence = predictor._calculate_confidence(test_query, section_302)
            print(f"6. Confidence score for Section 302: {confidence}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_prediction()
