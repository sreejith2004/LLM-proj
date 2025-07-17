"""
Tests for Chain-of-Thought Reasoning Module.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.reasoning.cot_reasoning import ChainOfThoughtReasoner, OutcomeLabel, CoTResult

class TestChainOfThoughtReasoner:
    """Test cases for Chain-of-Thought reasoning."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.reasoner = ChainOfThoughtReasoner()
        self.sample_case = """
        The appellant was charged under Section 302 IPC for the murder of the deceased. 
        The prosecution argued that there was sufficient circumstantial evidence to prove guilt beyond reasonable doubt.
        The defense contended that the evidence was insufficient and the accused should be given benefit of doubt.
        The trial court convicted the accused, but the High Court acquitted him on appeal.
        """
    
    def test_reasoner_initialization(self):
        """Test reasoner initialization."""
        assert self.reasoner is not None
        assert self.reasoner.legal_kb is not None
        assert 'ipc_sections' in self.reasoner.legal_kb
        assert 'crpc_sections' in self.reasoner.legal_kb
    
    def test_analyze_case_returns_cot_result(self):
        """Test that analyze_case returns a CoTResult."""
        result = self.reasoner.analyze_case(self.sample_case)
        
        assert isinstance(result, CoTResult)
        assert result.law_analysis is not None
        assert result.fact_analysis is not None
        assert result.argument_analysis is not None
        assert result.outcome_prediction is not None
        assert isinstance(result.final_verdict, OutcomeLabel)
        assert 0 <= result.overall_confidence <= 1
    
    def test_extract_ipc_sections(self):
        """Test IPC section extraction."""
        text = "The accused was charged under Section 302 IPC and Section 307 IPC"
        sections = self.reasoner._extract_ipc_sections(text)
        
        assert '302' in sections
        assert '307' in sections
    
    def test_extract_crpc_sections(self):
        """Test CrPC section extraction."""
        text = "Under Section 197 CrPC, prior sanction is required"
        sections = self.reasoner._extract_crpc_sections(text)
        
        assert '197' in sections
    
    def test_confidence_scores_valid_range(self):
        """Test that all confidence scores are in valid range [0, 1]."""
        result = self.reasoner.analyze_case(self.sample_case)
        
        assert 0 <= result.law_analysis.confidence <= 1
        assert 0 <= result.fact_analysis.confidence <= 1
        assert 0 <= result.argument_analysis.confidence <= 1
        assert 0 <= result.outcome_prediction.confidence <= 1
        assert 0 <= result.overall_confidence <= 1
    
    def test_case_type_categorization(self):
        """Test case type categorization."""
        murder_case = "This is a case involving murder and homicide"
        theft_case = "The accused was charged with theft and burglary"
        
        murder_type = self.reasoner._categorize_case_type(murder_case)
        theft_type = self.reasoner._categorize_case_type(theft_case)
        
        assert "homicide" in murder_type
        assert "property crime" in theft_type
    
    def test_legal_principles_identification(self):
        """Test legal principles identification."""
        text = "The burden of proof lies on prosecution beyond reasonable doubt"
        principles = self.reasoner._identify_legal_principles(text)
        
        assert any("burden of proof" in p.lower() for p in principles)
        assert any("reasonable doubt" in p.lower() for p in principles)
    
    def test_empty_case_handling(self):
        """Test handling of empty or minimal case text."""
        result = self.reasoner.analyze_case("")
        
        # Should still return a valid result, even if with low confidence
        assert isinstance(result, CoTResult)
        assert result.overall_confidence >= 0
    
    def test_confidence_to_text_conversion(self):
        """Test confidence score to text conversion."""
        assert self.reasoner._confidence_to_text(0.9) == "High"
        assert self.reasoner._confidence_to_text(0.7) == "Medium"
        assert self.reasoner._confidence_to_text(0.3) == "Low"

if __name__ == "__main__":
    # Run tests
    test_reasoner = TestChainOfThoughtReasoner()
    test_reasoner.setup_method()
    
    print("Running Chain-of-Thought Reasoning Tests...")
    
    try:
        test_reasoner.test_reasoner_initialization()
        print("✅ Reasoner initialization test passed")
        
        test_reasoner.test_analyze_case_returns_cot_result()
        print("✅ Analyze case test passed")
        
        test_reasoner.test_extract_ipc_sections()
        print("✅ IPC section extraction test passed")
        
        test_reasoner.test_extract_crpc_sections()
        print("✅ CrPC section extraction test passed")
        
        test_reasoner.test_confidence_scores_valid_range()
        print("✅ Confidence scores test passed")
        
        test_reasoner.test_case_type_categorization()
        print("✅ Case type categorization test passed")
        
        test_reasoner.test_legal_principles_identification()
        print("✅ Legal principles identification test passed")
        
        test_reasoner.test_empty_case_handling()
        print("✅ Empty case handling test passed")
        
        test_reasoner.test_confidence_to_text_conversion()
        print("✅ Confidence to text conversion test passed")
        
        print("\n🎉 All Chain-of-Thought tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise
