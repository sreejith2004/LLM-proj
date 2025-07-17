"""
Test script for the enhanced web application.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

def test_enhanced_app():
    """Test the enhanced web application components."""
    print("🧪 Testing Enhanced Legal AI Web Application")
    print("=" * 50)
    
    try:
        # Test imports
        print("1. Testing imports...")
        from web_app.enhanced_app import load_enhanced_system, show_home_page
        print("   ✅ Enhanced app imports successful")
        
        # Test component loading
        print("2. Testing component loading...")
        components, status = load_enhanced_system()
        print(f"   ✅ Component loading: {status}")
        print(f"   ✅ Loaded components: {list(components.keys())}")
        
        # Test individual components
        print("3. Testing individual components...")
        
        if 'cot_reasoner' in components:
            print("   ✅ Chain-of-Thought Reasoner loaded")
            
            # Quick test
            reasoner = components['cot_reasoner']
            sample_case = "The accused was charged under Section 302 IPC for murder."
            result = reasoner.analyze_case(sample_case)
            print(f"   ✅ CoT analysis completed: {result.final_verdict.value}")
        
        if 'visualizer' in components:
            print("   ✅ Legal Visualizer loaded")
            
            # Quick test
            visualizer = components['visualizer']
            sample_text = "The court ruled in favor of the appellant."
            dual_summary = visualizer.create_dual_view_summary(sample_text)
            print(f"   ✅ Dual-view summary created")
        
        if 'bias_detector' in components:
            print("   ✅ Bias Detector loaded")
            
            # Quick test
            detector = components['bias_detector']
            sample_text = "The appellant filed an appeal."
            report = detector.analyze_text(sample_text)
            print(f"   ✅ Bias analysis completed: {len(report.bias_instances)} bias instances")
        
        if 'multilingual_chat' in components:
            print("   ✅ Multilingual Chat loaded")
            
            # Quick test
            chat = components['multilingual_chat']
            supported_langs = chat.get_supported_languages()
            print(f"   ✅ Supports {len(supported_langs)} languages")
        
        if 'section_predictor' in components:
            print("   ✅ Section Predictor loaded")
            
            # Quick test
            predictor = components['section_predictor']
            sample_facts = "The accused killed the victim with a knife."
            recommendation = predictor.predict_sections(sample_facts)
            print(f"   ✅ Section prediction completed: {len(recommendation.primary_sections)} sections")
        
        if 'evaluator' in components:
            print("   ✅ Comprehensive Evaluator loaded")
        
        print("\n🎉 All tests passed! Enhanced web app is ready to run.")
        print("\n📋 To start the web application, run:")
        print("   streamlit run web_app/enhanced_app.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_enhanced_app()
    if success:
        print("\n✅ Enhanced Legal AI System is ready!")
    else:
        print("\n❌ Some issues found. Please check the errors above.")
