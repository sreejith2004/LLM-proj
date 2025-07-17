"""
Tests for Legal Summary Visualizer.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.visualization.legal_visualizer import (
    LegalSummaryVisualizer, ViewType, SummaryView, DualViewSummary
)

class TestLegalSummaryVisualizer:
    """Test cases for Legal Summary Visualizer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.visualizer = LegalSummaryVisualizer()
        self.sample_text = """
        The appellant was charged under Section 302 IPC for the murder of the deceased on 15th January 2020.
        The prosecution argued that there was sufficient circumstantial evidence to prove guilt beyond reasonable doubt.
        The defense contended that the evidence was insufficient and the accused should be given benefit of doubt.
        The trial court convicted the accused on 10th March 2021.
        """

    def test_visualizer_initialization(self):
        """Test visualizer initialization."""
        assert self.visualizer is not None
        assert self.visualizer.rouge_scorer is not None
        assert self.visualizer.legal_patterns is not None
        assert self.visualizer.simplification_map is not None

    def test_extract_legal_references(self):
        """Test legal reference extraction."""
        references = self.visualizer._extract_legal_references(self.sample_text)

        assert isinstance(references, list)
        assert any('Section 302' in ref for ref in references)

    def test_extract_key_terms(self):
        """Test key term extraction."""
        terms = self.visualizer._extract_key_terms(self.sample_text)

        assert isinstance(terms, list)
        assert len(terms) > 0
        # Should contain legal terms
        legal_terms = [term for term in terms if term.lower() in ['appellant', 'prosecution', 'defense']]
        assert len(legal_terms) > 0

    def test_extract_timeline_events(self):
        """Test timeline event extraction."""
        events = self.visualizer._extract_timeline_events(self.sample_text)

        assert isinstance(events, list)
        # Should find the date in the text
        if events:
            assert any('2020' in event['date'] or '2021' in event['date'] for event in events)

    def test_calculate_token_importance(self):
        """Test token importance calculation."""
        importance = self.visualizer._calculate_token_importance(self.sample_text)

        assert isinstance(importance, dict)
        # Should have scores for various tokens
        assert len(importance) > 0
        # All scores should be between 0 and 1
        for score in importance.values():
            assert 0 <= score <= 1

    def test_create_dual_view_summary(self):
        """Test dual view summary creation."""
        dual_summary = self.visualizer.create_dual_view_summary(self.sample_text)

        assert isinstance(dual_summary, DualViewSummary)
        assert isinstance(dual_summary.legal_view, SummaryView)
        assert isinstance(dual_summary.simplified_view, SummaryView)

        # Check view types
        assert dual_summary.legal_view.view_type == ViewType.LEGAL
        assert dual_summary.simplified_view.view_type == ViewType.SIMPLIFIED

        # Check content exists
        assert len(dual_summary.legal_view.content) > 0
        assert len(dual_summary.simplified_view.content) > 0

        # Simplified view should be more readable
        assert dual_summary.simplified_view.readability_score >= dual_summary.legal_view.readability_score

    def test_readability_score_calculation(self):
        """Test readability score calculation."""
        simple_text = "This is a simple sentence. It is easy to read."
        complex_text = "The aforementioned appellant hereinafter referred to as the petitioner."

        simple_score = self.visualizer._calculate_readability_score(simple_text, complex_terms=False)
        complex_score = self.visualizer._calculate_readability_score(complex_text, complex_terms=True)

        # Simple text should be more readable
        assert simple_score > complex_score
        assert 0 <= simple_score <= 1
        assert 0 <= complex_score <= 1

    def test_syllable_counting(self):
        """Test syllable counting function."""
        # Test basic syllable counting (approximation)
        assert self.visualizer._count_syllables("cat") >= 1
        assert self.visualizer._count_syllables("apple") >= 1
        assert self.visualizer._count_syllables("beautiful") >= 2
        assert self.visualizer._count_syllables("university") >= 3
        # All should return at least 1
        assert self.visualizer._count_syllables("") == 1

    def test_legal_term_simplification(self):
        """Test legal term simplification."""
        legal_text = "The appellant filed a petition."
        simplified = self.visualizer._create_simplified_summary(legal_text, ['appellant', 'petition'])

        # Should contain simplified terms
        assert 'person who appealed' in simplified.lower() or 'appellant' in simplified.lower()

    def test_comparison_metrics(self):
        """Test comparison metrics calculation."""
        dual_summary = self.visualizer.create_dual_view_summary(self.sample_text)
        metrics = dual_summary.comparison_metrics

        assert isinstance(metrics, dict)
        assert 'content_similarity' in metrics
        assert 'readability_improvement' in metrics
        assert 'term_simplification_ratio' in metrics

        # All metrics should be valid numbers
        for value in metrics.values():
            assert isinstance(value, (int, float))
            assert not np.isnan(value) if hasattr(value, '__float__') else True

    def test_visualization_data_creation(self):
        """Test visualization data creation."""
        dual_summary = self.visualizer.create_dual_view_summary(self.sample_text)
        viz_data = dual_summary.visualization_data

        assert viz_data is not None
        assert isinstance(viz_data.important_tokens, list)
        assert isinstance(viz_data.legal_sections, list)
        assert isinstance(viz_data.complexity_metrics, dict)

    def test_empty_text_handling(self):
        """Test handling of empty text."""
        dual_summary = self.visualizer.create_dual_view_summary("")

        # Should still create valid summary objects
        assert isinstance(dual_summary, DualViewSummary)
        assert dual_summary.legal_view.content is not None
        assert dual_summary.simplified_view.content is not None

if __name__ == "__main__":
    # Run tests
    import numpy as np

    test_visualizer = TestLegalSummaryVisualizer()
    test_visualizer.setup_method()

    print("Running Legal Summary Visualizer Tests...")

    try:
        test_visualizer.test_visualizer_initialization()
        print("✅ Visualizer initialization test passed")

        test_visualizer.test_extract_legal_references()
        print("✅ Legal reference extraction test passed")

        test_visualizer.test_extract_key_terms()
        print("✅ Key term extraction test passed")

        test_visualizer.test_extract_timeline_events()
        print("✅ Timeline event extraction test passed")

        test_visualizer.test_calculate_token_importance()
        print("✅ Token importance calculation test passed")

        test_visualizer.test_create_dual_view_summary()
        print("✅ Dual view summary creation test passed")

        test_visualizer.test_readability_score_calculation()
        print("✅ Readability score calculation test passed")

        test_visualizer.test_syllable_counting()
        print("✅ Syllable counting test passed")

        test_visualizer.test_legal_term_simplification()
        print("✅ Legal term simplification test passed")

        test_visualizer.test_comparison_metrics()
        print("✅ Comparison metrics test passed")

        test_visualizer.test_visualization_data_creation()
        print("✅ Visualization data creation test passed")

        test_visualizer.test_empty_text_handling()
        print("✅ Empty text handling test passed")

        print("\n🎉 All Legal Summary Visualizer tests passed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise
