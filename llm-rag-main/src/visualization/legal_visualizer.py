"""
Simplified Legal Summary Visualizer.

This module provides dual-view summarization:
1. Legal terms view (with section references)
2. Simplified view for public/law students

Includes visual timeline and token highlighting with ROUGE-based importance.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
from rouge_score import rouge_scorer

# Import our real LLM client
try:
    from src.utils.llm_client import get_legal_llm, LLMResponse
    HAS_LLM = True
except ImportError:
    HAS_LLM = False
# Optional visualization imports - will be imported when needed
# import matplotlib.pyplot as plt
# import seaborn as sns
# from wordcloud import WordCloud
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import plotly.express as px

logger = logging.getLogger(__name__)

class ViewType(Enum):
    """Types of summary views."""
    LEGAL = "legal"
    SIMPLIFIED = "simplified"
    BOTH = "both"

@dataclass
class SummaryView:
    """A single summary view."""
    view_type: ViewType
    title: str
    content: str
    key_terms: List[str]
    legal_references: List[str]
    importance_scores: Dict[str, float]
    readability_score: float

@dataclass
class VisualizationData:
    """Data for visualization components."""
    timeline_events: List[Dict]
    important_tokens: List[Tuple[str, float]]
    legal_sections: List[str]
    key_concepts: List[str]
    complexity_metrics: Dict[str, float]

@dataclass
class DualViewSummary:
    """Complete dual-view summary with visualizations."""
    legal_view: SummaryView
    simplified_view: SummaryView
    visualization_data: VisualizationData
    comparison_metrics: Dict[str, float]

class LegalSummaryVisualizer:
    """
    Dual-view legal summary visualizer with visual components.
    """

    def __init__(self):
        """Initialize the visualizer."""
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

        # Legal term patterns
        self.legal_patterns = {
            'sections': r'(?:Section|Sec\.?)\s*(\d+[A-Z]?)',
            'acts': r'([A-Z][a-z\s]+(?:Act|Code))\s*(?:,?\s*\d{4})?',
            'courts': r'(Supreme Court|High Court|District Court|Tribunal)',
            'legal_terms': r'\b(appellant|respondent|petitioner|defendant|plaintiff|accused|prosecution|defense|judgment|decree|order|writ|mandamus|certiorari|habeas corpus|quo warranto)\b'
        }

        # Simplification mappings
        self.simplification_map = {
            'appellant': 'person who appealed',
            'respondent': 'other party in appeal',
            'petitioner': 'person who filed petition',
            'defendant': 'person being sued',
            'plaintiff': 'person filing lawsuit',
            'accused': 'person charged with crime',
            'prosecution': 'government lawyer',
            'defense': 'accused person\'s lawyer',
            'judgment': 'court decision',
            'decree': 'court order',
            'writ': 'court command',
            'mandamus': 'order to do something',
            'certiorari': 'review of lower court',
            'habeas corpus': 'protection from illegal detention',
            'quo warranto': 'challenge to authority'
        }

        logger.info("Legal Summary Visualizer initialized")

    def create_dual_view_summary(self, legal_text: str,
                                case_title: Optional[str] = None) -> DualViewSummary:
        """
        Create dual-view summary with visualizations using LLM + rule-based methods.

        Args:
            legal_text: Original legal document text
            case_title: Optional case title

        Returns:
            DualViewSummary: Complete dual-view summary
        """
        logger.info("Creating enhanced dual-view summary with LLM")

        if HAS_LLM:
            return self._create_summary_with_llm(legal_text, case_title)
        else:
            logger.warning("LLM not available, using rule-based summarization")
            return self._create_summary_rule_based(legal_text, case_title)

    def _create_summary_with_llm(self, legal_text: str, case_title: Optional[str] = None) -> DualViewSummary:
        """Create summary using LLM + rule-based methods."""
        try:
            llm_client = get_legal_llm()

            # Get LLM summaries for both views
            legal_response = llm_client.summarize_legal_text(legal_text, "professional")
            simplified_response = llm_client.summarize_legal_text(legal_text, "public")

            # Extract key information using rule-based methods
            legal_refs = self._extract_legal_references(legal_text)
            key_terms = self._extract_key_terms(legal_text)
            timeline_events = self._extract_timeline_events(legal_text)
            token_importance = self._calculate_token_importance(legal_text)

            # Create enhanced views combining LLM and rule-based
            legal_view = self._create_enhanced_legal_view(
                legal_text, legal_response, legal_refs, key_terms, token_importance
            )

            simplified_view = self._create_enhanced_simplified_view(
                legal_text, simplified_response, legal_refs, key_terms, token_importance
            )

            # Create visualization data
            viz_data = self._create_visualization_data(legal_text, legal_refs, key_terms,
                                                     timeline_events, token_importance)

            # Calculate comparison metrics
            comparison_metrics = self._calculate_comparison_metrics(legal_view, simplified_view)

            return DualViewSummary(
                legal_view=legal_view,
                simplified_view=simplified_view,
                visualization_data=viz_data,
                comparison_metrics=comparison_metrics
            )

        except Exception as e:
            logger.error(f"Error in LLM summarization: {e}")
            return self._create_summary_rule_based(legal_text, case_title)

    def _create_summary_rule_based(self, legal_text: str, case_title: Optional[str] = None) -> DualViewSummary:
        """Original rule-based summary creation."""
        # Extract key information
        legal_refs = self._extract_legal_references(legal_text)
        key_terms = self._extract_key_terms(legal_text)
        timeline_events = self._extract_timeline_events(legal_text)

        # Calculate token importance
        token_importance = self._calculate_token_importance(legal_text)

        # Create legal view
        legal_view = self._create_legal_view(legal_text, legal_refs, key_terms, token_importance)

        # Create simplified view
        simplified_view = self._create_simplified_view(legal_text, legal_refs, key_terms, token_importance)

        # Create visualization data
        viz_data = self._create_visualization_data(legal_text, legal_refs, key_terms,
                                                 timeline_events, token_importance)

        # Calculate comparison metrics
        comparison_metrics = self._calculate_comparison_metrics(legal_view, simplified_view)

        return DualViewSummary(
            legal_view=legal_view,
            simplified_view=simplified_view,
            visualization_data=viz_data,
            comparison_metrics=comparison_metrics
        )

    def _create_enhanced_legal_view(self, text: str, llm_response: 'LLMResponse',
                                   legal_refs: List[str], key_terms: List[str],
                                   token_importance: Dict[str, float]) -> SummaryView:
        """Create enhanced legal view using LLM + rule-based."""
        summary_parts = []

        # Use LLM summary if successful
        if llm_response.success and llm_response.content:
            summary_parts.append("**AI-Enhanced Legal Analysis:**")
            summary_parts.append(llm_response.content)
            summary_parts.append("")

        # Add legal references section
        if legal_refs:
            summary_parts.append("**Legal References:**")
            for ref in legal_refs[:5]:
                summary_parts.append(f"- {ref}")
            summary_parts.append("")

        # Add key legal terms with enhanced analysis
        if key_terms:
            summary_parts.append("**Key Legal Terms:**")
            legal_key_terms = [term for term in key_terms if term.lower() in self.simplification_map]
            for term in legal_key_terms[:5]:
                importance = token_importance.get(term.lower(), 0.0)
                summary_parts.append(f"- {term} (importance: {importance:.2f})")

        content = "\n".join(summary_parts)

        return SummaryView(
            view_type=ViewType.LEGAL,
            title="Enhanced Legal Professional View",
            content=content,
            key_terms=key_terms,
            legal_references=legal_refs,
            importance_scores=token_importance,
            readability_score=self._calculate_readability_score(content, complex_terms=True)
        )

    def _create_enhanced_simplified_view(self, text: str, llm_response: 'LLMResponse',
                                        legal_refs: List[str], key_terms: List[str],
                                        token_importance: Dict[str, float]) -> SummaryView:
        """Create enhanced simplified view using LLM + rule-based."""
        summary_parts = []

        # Use LLM simplified summary if successful
        if llm_response.success and llm_response.content:
            summary_parts.append("**AI-Generated Simple Summary:**")
            summary_parts.append(llm_response.content)
            summary_parts.append("")
        else:
            # Fallback to rule-based simplification
            simplified_summary = self._create_simplified_summary(text, key_terms)
            summary_parts.append("**Case Summary (Simplified):**")
            summary_parts.append(simplified_summary)
            summary_parts.append("")

        # Add simplified explanations of legal terms
        if key_terms:
            summary_parts.append("**Key Terms Explained:**")
            legal_terms_found = [term for term in key_terms if term.lower() in self.simplification_map]
            for term in legal_terms_found[:5]:
                explanation = self.simplification_map.get(term.lower(), term)
                summary_parts.append(f"- **{term}**: {explanation}")

        # Add court information in simple terms
        if legal_refs:
            courts = [ref for ref in legal_refs if 'Court' in ref]
            if courts:
                summary_parts.append("\n**Courts Involved:**")
                for court in courts[:3]:
                    summary_parts.append(f"- {court}")

        content = "\n".join(summary_parts)

        return SummaryView(
            view_type=ViewType.SIMPLIFIED,
            title="Enhanced Public/Student View",
            content=content,
            key_terms=[term for term in key_terms if term.lower() in self.simplification_map],
            legal_references=legal_refs,
            importance_scores=token_importance,
            readability_score=self._calculate_readability_score(content, complex_terms=False)
        )

    def _extract_legal_references(self, text: str) -> List[str]:
        """Extract legal references from text."""
        references = []

        for ref_type, pattern in self.legal_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if ref_type == 'sections':
                references.extend([f"Section {match}" for match in matches])
            elif ref_type == 'acts':
                references.extend(matches)
            elif ref_type == 'courts':
                references.extend(matches)

        return list(set(references))

    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key legal terms from text."""
        terms = []

        # Extract legal terms using pattern
        legal_term_matches = re.findall(self.legal_patterns['legal_terms'], text, re.IGNORECASE)
        terms.extend(legal_term_matches)

        # Add other important terms (simple frequency-based)
        words = re.findall(r'\b[A-Za-z]{4,}\b', text.lower())
        word_freq = {}
        for word in words:
            if word not in ['that', 'this', 'with', 'from', 'they', 'have', 'been', 'were', 'said']:
                word_freq[word] = word_freq.get(word, 0) + 1

        # Get top frequent terms
        top_terms = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        terms.extend([term[0] for term in top_terms])

        return list(set(terms))

    def _extract_timeline_events(self, text: str) -> List[Dict]:
        """Extract timeline events from legal text."""
        events = []

        # Simple date pattern matching
        date_patterns = [
            r'(\d{1,2}[/-]\d{1,2}[/-]\d{4})',  # DD/MM/YYYY or DD-MM-YYYY
            r'(\d{4}[/-]\d{1,2}[/-]\d{1,2})',  # YYYY/MM/DD or YYYY-MM-DD
            r'(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})',
            r'((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})'
        ]

        sentences = text.split('.')
        for i, sentence in enumerate(sentences):
            for pattern in date_patterns:
                matches = re.findall(pattern, sentence, re.IGNORECASE)
                for match in matches:
                    events.append({
                        'date': match,
                        'event': sentence.strip()[:100] + '...' if len(sentence) > 100 else sentence.strip(),
                        'order': i
                    })

        return events[:10]  # Limit to 10 events

    def _calculate_token_importance(self, text: str) -> Dict[str, float]:
        """Calculate token importance using ROUGE-based scoring."""
        sentences = text.split('.')
        if len(sentences) < 2:
            return {}

        # Create a reference summary (first few sentences)
        reference = '. '.join(sentences[:min(3, len(sentences))])

        token_scores = {}
        words = re.findall(r'\b[A-Za-z]{3,}\b', text.lower())

        for word in set(words):
            # Calculate ROUGE score for each word
            word_text = word
            try:
                scores = self.rouge_scorer.score(reference, word_text)
                # Use ROUGE-1 F-measure as importance score
                token_scores[word] = scores['rouge1'].fmeasure
            except:
                token_scores[word] = 0.0

        return token_scores

    def _create_legal_view(self, text: str, legal_refs: List[str],
                          key_terms: List[str], token_importance: Dict[str, float]) -> SummaryView:
        """Create legal professional view."""

        # Create legal summary with references
        summary_parts = []

        # Add legal references section
        if legal_refs:
            summary_parts.append("**Legal References:**")
            for ref in legal_refs[:5]:  # Top 5 references
                summary_parts.append(f"- {ref}")
            summary_parts.append("")

        # Create main summary with legal terminology
        main_summary = self._create_summary_with_legal_terms(text, key_terms)
        summary_parts.append("**Legal Analysis:**")
        summary_parts.append(main_summary)

        # Add key legal terms
        if key_terms:
            summary_parts.append("\n**Key Legal Terms:**")
            legal_key_terms = [term for term in key_terms if term.lower() in self.simplification_map]
            for term in legal_key_terms[:5]:
                summary_parts.append(f"- {term}")

        content = "\n".join(summary_parts)

        return SummaryView(
            view_type=ViewType.LEGAL,
            title="Legal Professional View",
            content=content,
            key_terms=key_terms,
            legal_references=legal_refs,
            importance_scores=token_importance,
            readability_score=self._calculate_readability_score(content, complex_terms=True)
        )

    def _create_simplified_view(self, text: str, legal_refs: List[str],
                               key_terms: List[str], token_importance: Dict[str, float]) -> SummaryView:
        """Create simplified public view."""

        summary_parts = []

        # Create simplified summary
        simplified_summary = self._create_simplified_summary(text, key_terms)
        summary_parts.append("**Case Summary (Simplified):**")
        summary_parts.append(simplified_summary)

        # Add simplified explanations of legal terms
        if key_terms:
            summary_parts.append("\n**Key Terms Explained:**")
            legal_terms_found = [term for term in key_terms if term.lower() in self.simplification_map]
            for term in legal_terms_found[:5]:
                explanation = self.simplification_map.get(term.lower(), term)
                summary_parts.append(f"- **{term}**: {explanation}")

        # Add court information in simple terms
        if legal_refs:
            courts = [ref for ref in legal_refs if 'Court' in ref]
            if courts:
                summary_parts.append("\n**Courts Involved:**")
                for court in courts[:3]:
                    summary_parts.append(f"- {court}")

        content = "\n".join(summary_parts)

        return SummaryView(
            view_type=ViewType.SIMPLIFIED,
            title="Public/Student View",
            content=content,
            key_terms=[term for term in key_terms if term.lower() in self.simplification_map],
            legal_references=legal_refs,
            importance_scores=token_importance,
            readability_score=self._calculate_readability_score(content, complex_terms=False)
        )

    def _create_summary_with_legal_terms(self, text: str, key_terms: List[str]) -> str:
        """Create summary preserving legal terminology."""
        sentences = text.split('.')

        # Select important sentences based on key terms
        important_sentences = []
        for sentence in sentences:
            score = 0
            for term in key_terms:
                if term.lower() in sentence.lower():
                    score += 1
            if score > 0:
                important_sentences.append((sentence.strip(), score))

        # Sort by importance and take top sentences
        important_sentences.sort(key=lambda x: x[1], reverse=True)
        selected_sentences = [sent[0] for sent in important_sentences[:5]]

        return '. '.join(selected_sentences) + '.' if selected_sentences else text[:500] + '...'

    def _create_simplified_summary(self, text: str, key_terms: List[str]) -> str:
        """Create simplified summary with plain language."""
        # Get base summary
        base_summary = self._create_summary_with_legal_terms(text, key_terms)

        # Simplify legal terms
        simplified = base_summary
        for legal_term, simple_term in self.simplification_map.items():
            pattern = r'\b' + re.escape(legal_term) + r'\b'
            simplified = re.sub(pattern, simple_term, simplified, flags=re.IGNORECASE)

        # Simplify sentence structure (basic)
        simplified = re.sub(r'\b(?:whereas|wherefore|heretofore|hereinafter)\b', '', simplified, flags=re.IGNORECASE)
        simplified = re.sub(r'\s+', ' ', simplified)  # Clean up extra spaces

        return simplified.strip()

    def _calculate_readability_score(self, text: str, complex_terms: bool = True) -> float:
        """Calculate readability score (simplified Flesch score)."""
        sentences = len(text.split('.'))
        words = len(text.split())

        if sentences == 0 or words == 0:
            return 0.0

        # Count syllables (approximation)
        syllables = sum([self._count_syllables(word) for word in text.split()])

        # Simplified Flesch Reading Ease formula
        if sentences > 0 and words > 0:
            avg_sentence_length = words / sentences
            avg_syllables_per_word = syllables / words

            flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)

            # Adjust for legal complexity
            if complex_terms:
                flesch_score -= 20  # Legal documents are inherently more complex

            # Normalize to 0-1 scale
            return max(0, min(100, flesch_score)) / 100

        return 0.5

    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (approximation)."""
        word = word.lower()
        vowels = 'aeiouy'
        syllable_count = 0
        prev_was_vowel = False

        for char in word:
            if char in vowels:
                if not prev_was_vowel:
                    syllable_count += 1
                prev_was_vowel = True
            else:
                prev_was_vowel = False

        # Handle silent 'e'
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1

        return max(1, syllable_count)

    def _create_visualization_data(self, text: str, legal_refs: List[str],
                                  key_terms: List[str], timeline_events: List[Dict],
                                  token_importance: Dict[str, float]) -> VisualizationData:
        """Create data for visualizations."""

        # Get top important tokens
        important_tokens = sorted(token_importance.items(), key=lambda x: x[1], reverse=True)[:20]

        # Extract legal sections
        legal_sections = [ref for ref in legal_refs if 'Section' in ref]

        # Get key concepts (high-importance terms)
        key_concepts = [term for term in key_terms if token_importance.get(term.lower(), 0) > 0.1]

        # Calculate complexity metrics
        complexity_metrics = {
            'legal_term_density': len([t for t in key_terms if t.lower() in self.simplification_map]) / max(len(key_terms), 1),
            'reference_density': len(legal_refs) / max(len(text.split()), 1) * 1000,  # per 1000 words
            'sentence_complexity': np.mean([len(s.split()) for s in text.split('.') if s.strip()]),
            'readability_score': self._calculate_readability_score(text)
        }

        return VisualizationData(
            timeline_events=timeline_events,
            important_tokens=important_tokens,
            legal_sections=legal_sections,
            key_concepts=key_concepts,
            complexity_metrics=complexity_metrics
        )

    def _calculate_comparison_metrics(self, legal_view: SummaryView,
                                    simplified_view: SummaryView) -> Dict[str, float]:
        """Calculate metrics comparing the two views."""

        # ROUGE scores between views
        rouge_scores = self.rouge_scorer.score(legal_view.content, simplified_view.content)

        # Length comparison
        legal_length = len(legal_view.content.split())
        simplified_length = len(simplified_view.content.split())
        length_ratio = simplified_length / max(legal_length, 1)

        # Readability improvement
        readability_improvement = simplified_view.readability_score - legal_view.readability_score

        # Term simplification ratio
        legal_terms_count = len([t for t in legal_view.key_terms if t.lower() in self.simplification_map])
        simplified_terms_count = len([t for t in simplified_view.key_terms if t.lower() in self.simplification_map])
        simplification_ratio = 1 - (simplified_terms_count / max(legal_terms_count, 1))

        return {
            'content_similarity': rouge_scores['rougeL'].fmeasure,
            'length_reduction_ratio': 1 - length_ratio,
            'readability_improvement': readability_improvement,
            'term_simplification_ratio': simplification_ratio,
            'legal_view_readability': legal_view.readability_score,
            'simplified_view_readability': simplified_view.readability_score
        }

    def generate_visualizations(self, dual_summary: DualViewSummary,
                               output_dir: Optional[str] = None) -> Dict[str, str]:
        """Generate visualization plots."""
        viz_files = {}

        try:
            # Import plotly only when needed
            import plotly.graph_objects as go

            # 1. Token importance visualization
            if dual_summary.visualization_data.important_tokens:
                fig = self._create_token_importance_plot(dual_summary.visualization_data.important_tokens)
                if output_dir:
                    filepath = f"{output_dir}/token_importance.html"
                    fig.write_html(filepath)
                    viz_files['token_importance'] = filepath

            # 2. Timeline visualization
            if dual_summary.visualization_data.timeline_events:
                fig = self._create_timeline_plot(dual_summary.visualization_data.timeline_events)
                if output_dir:
                    filepath = f"{output_dir}/timeline.html"
                    fig.write_html(filepath)
                    viz_files['timeline'] = filepath

            # 3. Complexity comparison
            fig = self._create_complexity_comparison(dual_summary)
            if output_dir:
                filepath = f"{output_dir}/complexity_comparison.html"
                fig.write_html(filepath)
                viz_files['complexity'] = filepath

            logger.info(f"Generated {len(viz_files)} visualizations")

        except ImportError:
            logger.warning("Plotly not available. Skipping visualizations.")
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")

        return viz_files

    def _create_token_importance_plot(self, important_tokens: List[Tuple[str, float]]):
        """Create token importance bar plot."""
        import plotly.graph_objects as go

        tokens, scores = zip(*important_tokens[:15])  # Top 15 tokens

        fig = go.Figure(data=[
            go.Bar(
                x=list(scores),
                y=list(tokens),
                orientation='h',
                marker_color='lightblue'
            )
        ])

        fig.update_layout(
            title="Token Importance Scores (ROUGE-based)",
            xaxis_title="Importance Score",
            yaxis_title="Tokens",
            height=500
        )

        return fig

    def _create_timeline_plot(self, timeline_events: List[Dict]):
        """Create timeline visualization."""
        import plotly.graph_objects as go

        if not timeline_events:
            return go.Figure()

        # Sort events by order
        events = sorted(timeline_events, key=lambda x: x.get('order', 0))

        fig = go.Figure()

        for i, event in enumerate(events):
            fig.add_trace(go.Scatter(
                x=[i],
                y=[0],
                mode='markers+text',
                marker=dict(size=15, color='red'),
                text=event['date'],
                textposition="top center",
                hovertext=event['event'],
                name=f"Event {i+1}"
            ))

        fig.update_layout(
            title="Case Timeline",
            xaxis_title="Event Sequence",
            yaxis=dict(visible=False),
            height=300,
            showlegend=False
        )

        return fig

    def _create_complexity_comparison(self, dual_summary: DualViewSummary):
        """Create complexity comparison chart."""
        import plotly.graph_objects as go

        metrics = dual_summary.comparison_metrics

        categories = ['Readability', 'Term Complexity', 'Content Similarity']
        legal_scores = [
            metrics['legal_view_readability'],
            1 - metrics['term_simplification_ratio'],  # Inverse for complexity
            1.0  # Reference point
        ]
        simplified_scores = [
            metrics['simplified_view_readability'],
            metrics['term_simplification_ratio'],
            metrics['content_similarity']
        ]

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=legal_scores,
            theta=categories,
            fill='toself',
            name='Legal View',
            line_color='red'
        ))

        fig.add_trace(go.Scatterpolar(
            r=simplified_scores,
            theta=categories,
            fill='toself',
            name='Simplified View',
            line_color='blue'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            title="Complexity Comparison: Legal vs Simplified Views"
        )

        return fig

def main():
    """Example usage of Legal Summary Visualizer."""
    # Sample legal text
    sample_text = """
    The appellant was charged under Section 302 IPC for the murder of the deceased on 15th January 2020.
    The prosecution argued that there was sufficient circumstantial evidence to prove guilt beyond reasonable doubt.
    The defense contended that the evidence was insufficient and the accused should be given benefit of doubt.
    The trial court convicted the accused on 10th March 2021, but the High Court acquitted him on appeal on 5th September 2022.
    The Supreme Court heard the matter on 20th November 2023.
    """

    # Initialize visualizer
    visualizer = LegalSummaryVisualizer()

    # Create dual-view summary
    dual_summary = visualizer.create_dual_view_summary(sample_text, "Sample Murder Case")

    # Display results
    print("=" * 60)
    print("DUAL-VIEW LEGAL SUMMARY")
    print("=" * 60)

    print(f"\n{dual_summary.legal_view.title}")
    print("-" * 40)
    print(dual_summary.legal_view.content)
    print(f"Readability Score: {dual_summary.legal_view.readability_score:.2f}")

    print(f"\n{dual_summary.simplified_view.title}")
    print("-" * 40)
    print(dual_summary.simplified_view.content)
    print(f"Readability Score: {dual_summary.simplified_view.readability_score:.2f}")

    print("\nComparison Metrics:")
    print("-" * 20)
    for metric, value in dual_summary.comparison_metrics.items():
        print(f"{metric}: {value:.3f}")

    print(f"\nVisualization Data:")
    print(f"- Timeline Events: {len(dual_summary.visualization_data.timeline_events)}")
    print(f"- Important Tokens: {len(dual_summary.visualization_data.important_tokens)}")
    print(f"- Legal Sections: {len(dual_summary.visualization_data.legal_sections)}")

if __name__ == "__main__":
    main()
