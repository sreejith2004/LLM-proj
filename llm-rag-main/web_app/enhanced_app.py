"""
Enhanced Streamlit web application for Legal LLM system with all novel components.
"""

import streamlit as st
import sys
from pathlib import Path
import logging
import json
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

# Import all our novel components
from src.reasoning.cot_reasoning import ChainOfThoughtReasoner, OutcomeLabel
from src.visualization.legal_visualizer import LegalSummaryVisualizer, ViewType
from src.bias.bias_detector import BiasDetector, BiasType, SeverityLevel
from src.multilingual.multilingual_chat import MultilingualLegalChat, SupportedLanguage
from src.legal_sections.section_predictor import LegalSectionPredictor
from src.evaluation.comprehensive_evaluator import ComprehensiveEvaluator, EvaluationMetric, TaskType

# Import existing components
from src.chatbot.legal_chatbot import LegalChatbot
from src.models.summarizer import LegalSummarizer
from src.rag.legal_rag import LegalRAG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Enhanced Legal AI Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f4e79;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_enhanced_system():
    """Load all enhanced system components with caching."""
    try:
        components = {}

        # Load Chain-of-Thought Reasoner
        components['cot_reasoner'] = ChainOfThoughtReasoner()

        # Load Legal Visualizer
        components['visualizer'] = LegalSummaryVisualizer()

        # Load Bias Detector
        components['bias_detector'] = BiasDetector()

        # Load Multilingual Chat
        components['multilingual_chat'] = MultilingualLegalChat()

        # Load Section Predictor
        components['section_predictor'] = LegalSectionPredictor()

        # Load Comprehensive Evaluator
        components['evaluator'] = ComprehensiveEvaluator()

        # Try to load existing components
        try:
            components['chatbot'] = LegalChatbot(
                summarizer_model_path="./results",
                rag_index_path="./data/rag_index"
            )
            components['multilingual_chat'].base_chatbot = components['chatbot']
        except Exception as e:
            logger.warning(f"Could not load full chatbot: {e}")
            components['chatbot'] = None

        return components, "✅ Enhanced system loaded successfully"

    except Exception as e:
        logger.error(f"Failed to load enhanced system: {e}")
        return {}, f"❌ Failed to load system: {e}"

def main():
    """Main application function."""

    # Header
    st.markdown('<h1 class="main-header">⚖️ Enhanced Legal AI Assistant</h1>', unsafe_allow_html=True)

    # Load system components
    components, status_message = load_enhanced_system()

    # Display system status
    if "✅" in status_message:
        st.markdown(f'<div class="success-box">{status_message}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="warning-box">{status_message}</div>', unsafe_allow_html=True)

    # Sidebar navigation
    st.sidebar.title("🔧 Navigation")

    # Feature selection
    features = [
        "🏠 Home",
        "🧠 Chain-of-Thought Analysis",
        "📊 Dual-View Summarizer",
        "🔍 Bias & Hallucination Detection",
        "🌐 Multilingual Legal Chat",
        "⚖️ Legal Section Predictor",
        "📈 Performance Evaluation",
        "💬 Enhanced Legal Chat",
        "🔬 System Analytics"
    ]

    selected_feature = st.sidebar.selectbox("Select Feature", features)

    # Feature routing
    if selected_feature == "🏠 Home":
        show_home_page(components)
    elif selected_feature == "🧠 Chain-of-Thought Analysis":
        show_cot_analysis(components)
    elif selected_feature == "📊 Dual-View Summarizer":
        show_dual_view_summarizer(components)
    elif selected_feature == "🔍 Bias & Hallucination Detection":
        show_bias_detection(components)
    elif selected_feature == "🌐 Multilingual Legal Chat":
        show_multilingual_chat(components)
    elif selected_feature == "⚖️ Legal Section Predictor":
        show_section_predictor(components)
    elif selected_feature == "📈 Performance Evaluation":
        show_performance_evaluation(components)
    elif selected_feature == "💬 Enhanced Legal Chat":
        show_enhanced_chat(components)
    elif selected_feature == "🔬 System Analytics":
        show_system_analytics(components)

def show_home_page(components):
    """Display home page with system overview."""
    st.markdown("## 🎯 Welcome to Enhanced Legal AI Assistant")

    st.markdown("""
    This enhanced system includes **6 novel components** that make it unique in the legal AI landscape:
    """)

    # Feature overview in columns
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="feature-card">
        <h4>🧠 Chain-of-Thought Reasoning</h4>
        <p>Structured legal analysis: LAW → FACT → ARGUMENT → OUTCOME with confidence scores</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="feature-card">
        <h4>📊 Dual-View Summarizer</h4>
        <p>Legal professional view + simplified public view with visual timeline</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="feature-card">
        <h4>🔍 Bias Detection</h4>
        <p>Real-time detection of gender/religion/caste bias + hallucination checking</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
        <h4>🌐 Multilingual Support</h4>
        <p>Support for Hindi, Tamil, Malayalam + 7 other Indian languages</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="feature-card">
        <h4>⚖️ Section Predictor</h4>
        <p>Predict applicable IPC/CrPC sections from case facts with explanations</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="feature-card">
        <h4>📈 Performance Evaluation</h4>
        <p>Comprehensive LLM evaluation with ROUGE, BLEU, legal accuracy metrics</p>
        </div>
        """, unsafe_allow_html=True)

    # System statistics
    st.markdown("## 📊 System Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="metric-card">
        <h3>6</h3>
        <p>Novel Components</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
        <h3>10</h3>
        <p>Supported Languages</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
        <h3>50+</h3>
        <p>Legal Sections</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="metric-card">
        <h3>10+</h3>
        <p>Evaluation Metrics</p>
        </div>
        """, unsafe_allow_html=True)

    # Quick start guide
    st.markdown("## 🚀 Quick Start Guide")

    st.markdown("""
    1. **🧠 Try Chain-of-Thought Analysis**: Analyze legal cases with structured reasoning
    2. **📊 Use Dual-View Summarizer**: Get both professional and simplified summaries
    3. **🔍 Check for Bias**: Detect potential bias and hallucinations in legal text
    4. **🌐 Chat in Your Language**: Ask legal questions in Hindi, Tamil, or other Indian languages
    5. **⚖️ Predict Legal Sections**: Get applicable IPC/CrPC sections from case facts
    6. **📈 Evaluate Performance**: Test system performance with comprehensive metrics
    """)

def show_cot_analysis(components):
    """Display Chain-of-Thought analysis interface."""
    st.markdown("## 🧠 Chain-of-Thought Legal Reasoning")

    st.markdown("""
    This module provides structured legal analysis following the pattern:
    **LAW → FACT → ARGUMENT → OUTCOME** with confidence scores and verdict prediction.
    """)

    if 'cot_reasoner' not in components:
        st.error("Chain-of-Thought Reasoner not available")
        return

    # Input section
    st.markdown("### 📝 Case Input")

    # Sample cases for quick testing
    sample_cases = {
        "Murder Case": "The appellant was charged under Section 302 IPC for the murder of the deceased. The prosecution argued that there was sufficient circumstantial evidence to prove guilt beyond reasonable doubt. The defense contended that the evidence was insufficient and the accused should be given benefit of doubt. The trial court convicted the accused, but the High Court acquitted him on appeal.",
        "Fraud Case": "The accused cheated the complainant by making false representations about investment returns. He dishonestly induced the victim to transfer Rs. 5 lakhs. The accused used the money for personal purposes and disappeared.",
        "Domestic Violence": "The husband and his relatives subjected the wife to physical and mental cruelty. They demanded additional dowry and threatened to throw her out of the house. The wife filed a complaint under Section 498A IPC."
    }

    col1, col2 = st.columns([2, 1])

    with col1:
        case_text = st.text_area(
            "Enter case details:",
            height=200,
            placeholder="Describe the legal case facts, arguments, and circumstances..."
        )

    with col2:
        st.markdown("**Quick Examples:**")
        for case_name, case_content in sample_cases.items():
            if st.button(f"Load {case_name}", key=f"load_{case_name}"):
                st.session_state.case_text = case_content
                st.experimental_rerun()

    # Use session state for case text
    if 'case_text' in st.session_state:
        case_text = st.session_state.case_text

    # Analysis section
    if st.button("🔍 Analyze Case", type="primary"):
        if case_text.strip():
            with st.spinner("Performing Chain-of-Thought analysis..."):
                try:
                    reasoner = components['cot_reasoner']
                    result = reasoner.analyze_case(case_text)

                    # Display results
                    st.markdown("### 📊 Analysis Results")

                    # Overall verdict and confidence
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Final Verdict", result.final_verdict.value)
                    with col2:
                        st.metric("Overall Confidence", f"{result.overall_confidence:.2f}")

                    # Step-by-step analysis
                    st.markdown("### 🔄 Step-by-Step Reasoning")

                    # Create tabs for each step
                    tab1, tab2, tab3, tab4 = st.tabs(["1️⃣ LAW", "2️⃣ FACT", "3️⃣ ARGUMENT", "4️⃣ OUTCOME"])

                    with tab1:
                        st.markdown(f"**Confidence:** {result.law_analysis.confidence:.2f}")
                        st.markdown(result.law_analysis.reasoning)
                        if result.law_analysis.key_points:
                            st.markdown("**Key Points:**")
                            for point in result.law_analysis.key_points:
                                st.markdown(f"• {point}")

                    with tab2:
                        st.markdown(f"**Confidence:** {result.fact_analysis.confidence:.2f}")
                        st.markdown(result.fact_analysis.reasoning)
                        if result.fact_analysis.key_points:
                            st.markdown("**Key Points:**")
                            for point in result.fact_analysis.key_points:
                                st.markdown(f"• {point}")

                    with tab3:
                        st.markdown(f"**Confidence:** {result.argument_analysis.confidence:.2f}")
                        st.markdown(result.argument_analysis.reasoning)
                        if result.argument_analysis.key_points:
                            st.markdown("**Key Points:**")
                            for point in result.argument_analysis.key_points:
                                st.markdown(f"• {point}")

                    with tab4:
                        st.markdown(f"**Confidence:** {result.outcome_prediction.confidence:.2f}")
                        st.markdown(result.outcome_prediction.reasoning)
                        if result.outcome_prediction.key_points:
                            st.markdown("**Key Points:**")
                            for point in result.outcome_prediction.key_points:
                                st.markdown(f"• {point}")

                    # Confidence visualization
                    st.markdown("### 📈 Confidence Analysis")

                    confidence_data = {
                        'Step': ['LAW', 'FACT', 'ARGUMENT', 'OUTCOME'],
                        'Confidence': [
                            result.law_analysis.confidence,
                            result.fact_analysis.confidence,
                            result.argument_analysis.confidence,
                            result.outcome_prediction.confidence
                        ]
                    }

                    fig = px.bar(
                        confidence_data,
                        x='Step',
                        y='Confidence',
                        title="Confidence Scores by Analysis Step",
                        color='Confidence',
                        color_continuous_scale='RdYlGn'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)

                    # Complete reasoning chain
                    st.markdown("### 🔗 Complete Reasoning Chain")
                    st.text_area("Reasoning Summary:", result.reasoning_chain, height=200)

                except Exception as e:
                    st.error(f"Error during analysis: {e}")
        else:
            st.warning("Please enter case details to analyze.")

def show_dual_view_summarizer(components):
    """Display dual-view summarizer interface."""
    st.markdown("## 📊 Dual-View Legal Summarizer")

    st.markdown("""
    This module provides two complementary views:
    - **Legal Professional View**: Technical terms with section references
    - **Simplified Public View**: Plain language for students and general public
    """)

    if 'visualizer' not in components:
        st.error("Legal Visualizer not available")
        return

    # Input section
    st.markdown("### 📝 Document Input")

    # Sample documents for quick testing
    sample_docs = {
        "Supreme Court Judgment": """The Supreme Court in the case of Kesavananda Bharati v. State of Kerala held that the basic structure of the Constitution cannot be amended. The appellant challenged the constitutional validity of certain amendments. The Court ruled that while Parliament has wide powers to amend the Constitution under Article 368, these powers are not unlimited. The basic structure doctrine was established to preserve the fundamental framework of the Constitution. The judgment was delivered by a 13-judge bench and is considered a landmark decision in constitutional law.""",

        "Criminal Case Summary": """The accused was charged under Section 302 IPC for the murder of the deceased on 15th January 2020. The prosecution argued that there was sufficient circumstantial evidence to prove guilt beyond reasonable doubt. The defense contended that the evidence was insufficient and the accused should be given benefit of doubt. The trial court convicted the accused on 10th March 2021, but the High Court acquitted him on appeal on 5th September 2022. The case involved witness testimony, forensic evidence, and expert opinions.""",

        "Contract Dispute": """The plaintiff filed a suit for breach of contract against the defendant under the Indian Contract Act, 1872. The contract was for supply of goods worth Rs. 10 lakhs. The defendant failed to deliver the goods within the stipulated time period. The plaintiff claimed damages for loss of business and reputation. The court examined the terms of the contract, the circumstances of breach, and the quantum of damages. The judgment was in favor of the plaintiff with compensation awarded."""
    }

    col1, col2 = st.columns([2, 1])

    with col1:
        document_text = st.text_area(
            "Enter legal document or case text:",
            height=250,
            placeholder="Paste your legal document, judgment, or case summary here..."
        )

        case_title = st.text_input(
            "Case Title (optional):",
            placeholder="e.g., Kesavananda Bharati v. State of Kerala"
        )

    with col2:
        st.markdown("**Sample Documents:**")
        for doc_name, doc_content in sample_docs.items():
            if st.button(f"Load {doc_name}", key=f"load_doc_{doc_name}"):
                st.session_state.document_text = doc_content
                st.session_state.case_title = doc_name
                st.experimental_rerun()

    # Use session state for document text
    if 'document_text' in st.session_state:
        document_text = st.session_state.document_text
    if 'case_title' in st.session_state:
        case_title = st.session_state.case_title

    # Analysis section
    if st.button("📊 Generate Dual-View Summary", type="primary"):
        if document_text.strip():
            with st.spinner("Creating dual-view summary..."):
                try:
                    visualizer = components['visualizer']
                    dual_summary = visualizer.create_dual_view_summary(
                        document_text,
                        case_title if case_title else None
                    )

                    # Display results in tabs
                    st.markdown("### 📋 Summary Results")

                    tab1, tab2, tab3, tab4 = st.tabs([
                        "⚖️ Legal View",
                        "👥 Simplified View",
                        "📊 Comparison",
                        "📈 Analytics"
                    ])

                    with tab1:
                        st.markdown("#### Legal Professional View")
                        st.markdown(f"**Readability Score:** {dual_summary.legal_view.readability_score:.2f}")
                        st.markdown(dual_summary.legal_view.content)

                        if dual_summary.legal_view.legal_references:
                            st.markdown("**Legal References Found:**")
                            for ref in dual_summary.legal_view.legal_references[:5]:
                                st.markdown(f"• {ref}")

                        if dual_summary.legal_view.key_terms:
                            st.markdown("**Key Legal Terms:**")
                            terms_text = ", ".join(dual_summary.legal_view.key_terms[:10])
                            st.markdown(terms_text)

                    with tab2:
                        st.markdown("#### Simplified Public View")
                        st.markdown(f"**Readability Score:** {dual_summary.simplified_view.readability_score:.2f}")
                        st.markdown(dual_summary.simplified_view.content)

                        if dual_summary.simplified_view.key_terms:
                            st.markdown("**Key Terms (Simplified):**")
                            for term in dual_summary.simplified_view.key_terms[:5]:
                                st.markdown(f"• {term}")

                    with tab3:
                        st.markdown("#### Comparison Metrics")

                        # Metrics in columns
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric(
                                "Content Similarity",
                                f"{dual_summary.comparison_metrics['content_similarity']:.3f}"
                            )

                        with col2:
                            st.metric(
                                "Readability Improvement",
                                f"{dual_summary.comparison_metrics['readability_improvement']:.3f}"
                            )

                        with col3:
                            st.metric(
                                "Term Simplification",
                                f"{dual_summary.comparison_metrics['term_simplification_ratio']:.3f}"
                            )

                        # Readability comparison chart
                        readability_data = {
                            'View': ['Legal Professional', 'Simplified Public'],
                            'Readability Score': [
                                dual_summary.legal_view.readability_score,
                                dual_summary.simplified_view.readability_score
                            ]
                        }

                        fig = px.bar(
                            readability_data,
                            x='View',
                            y='Readability Score',
                            title="Readability Comparison",
                            color='Readability Score',
                            color_continuous_scale='RdYlGn'
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)

                    with tab4:
                        st.markdown("#### Document Analytics")

                        viz_data = dual_summary.visualization_data

                        # Timeline events
                        if viz_data.timeline_events:
                            st.markdown("**Timeline Events:**")
                            for event in viz_data.timeline_events:
                                st.markdown(f"• **{event['date']}**: {event['event'][:100]}...")

                        # Important tokens visualization
                        if viz_data.important_tokens:
                            st.markdown("**Most Important Terms:**")

                            # Create importance chart
                            tokens, scores = zip(*viz_data.important_tokens[:10])

                            importance_fig = go.Figure(data=[
                                go.Bar(
                                    x=list(scores),
                                    y=list(tokens),
                                    orientation='h',
                                    marker_color='lightblue'
                                )
                            ])

                            importance_fig.update_layout(
                                title="Token Importance Scores",
                                xaxis_title="Importance Score",
                                yaxis_title="Terms",
                                height=400
                            )

                            st.plotly_chart(importance_fig, use_container_width=True)

                        # Complexity metrics
                        st.markdown("**Complexity Metrics:**")
                        complexity = viz_data.complexity_metrics

                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Legal Term Density", f"{complexity['legal_term_density']:.3f}")
                            st.metric("Reference Density", f"{complexity['reference_density']:.3f}")

                        with col2:
                            st.metric("Sentence Complexity", f"{complexity['sentence_complexity']:.1f}")
                            st.metric("Overall Readability", f"{complexity['readability_score']:.3f}")

                    # Download options
                    st.markdown("### 💾 Export Options")

                    col1, col2 = st.columns(2)

                    with col1:
                        # Export legal view
                        legal_export = {
                            'title': case_title or 'Legal Document',
                            'view_type': 'Legal Professional',
                            'content': dual_summary.legal_view.content,
                            'readability_score': dual_summary.legal_view.readability_score,
                            'legal_references': dual_summary.legal_view.legal_references,
                            'key_terms': dual_summary.legal_view.key_terms
                        }

                        st.download_button(
                            "📄 Download Legal View",
                            data=json.dumps(legal_export, indent=2),
                            file_name=f"legal_view_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )

                    with col2:
                        # Export simplified view
                        simplified_export = {
                            'title': case_title or 'Legal Document',
                            'view_type': 'Simplified Public',
                            'content': dual_summary.simplified_view.content,
                            'readability_score': dual_summary.simplified_view.readability_score,
                            'key_terms': dual_summary.simplified_view.key_terms
                        }

                        st.download_button(
                            "📄 Download Simplified View",
                            data=json.dumps(simplified_export, indent=2),
                            file_name=f"simplified_view_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )

                except Exception as e:
                    st.error(f"Error creating dual-view summary: {e}")
                    import traceback
                    st.text(traceback.format_exc())
        else:
            st.warning("Please enter document text to summarize.")

def show_bias_detection(components):
    """Display bias detection interface."""
    st.markdown("## 🔍 Bias & Hallucination Detection")

    st.markdown("""
    This module detects:
    - **Gender/Religion/Caste Bias**: Identifies potentially biased language
    - **Hallucinated Citations**: Flags incorrect or non-existent legal references
    - **Factual Inconsistencies**: Checks for contradictory statements
    """)

    if 'bias_detector' not in components:
        st.error("Bias Detector not available")
        return

    # Input section
    st.markdown("### 📝 Text Analysis Input")

    # Sample texts with potential bias for testing
    sample_texts = {
        "Biased Gender Language": "The hysterical woman clearly provoked the incident, as is typical woman behavior. She was being emotional and irrational during the proceedings.",

        "Religious Bias Example": "The Muslim community tends to be more aggressive in such matters. This is characteristic behavior of people from that religious background.",

        "Caste Bias Example": "The upper caste mentality is evident in this case. The accused shows typical Brahminical attitude towards the lower caste victim.",

        "Hallucinated Citations": "In the landmark case of Fake Case v. Non-existent Party (2025), the Supreme Court ruled on this matter. Under Section 999 of the Imaginary Act, the accused should be punished. The Delhi High Court in Fictional Case v. Made-up Respondent (2024) also supported this view.",

        "Clean Legal Text": "The appellant filed an appeal against the judgment of the trial court. The case was heard by a three-judge bench of the High Court. The court examined the evidence and legal precedents before delivering the judgment."
    }

    col1, col2 = st.columns([2, 1])

    with col1:
        analysis_text = st.text_area(
            "Enter text to analyze for bias and hallucinations:",
            height=200,
            placeholder="Paste legal text, judgment, or any content you want to check for bias and factual accuracy..."
        )

    with col2:
        st.markdown("**Test Examples:**")
        for text_name, text_content in sample_texts.items():
            if st.button(f"Load {text_name}", key=f"load_bias_{text_name}"):
                st.session_state.analysis_text = text_content
                st.experimental_rerun()

    # Use session state for analysis text
    if 'analysis_text' in st.session_state:
        analysis_text = st.session_state.analysis_text

    # Configuration options
    st.markdown("### ⚙️ Detection Settings")

    col1, col2, col3 = st.columns(3)

    with col1:
        detect_gender = st.checkbox("Gender Bias Detection", value=True)
        detect_religion = st.checkbox("Religious Bias Detection", value=True)

    with col2:
        detect_caste = st.checkbox("Caste Bias Detection", value=True)
        detect_citations = st.checkbox("Citation Verification", value=True)

    with col3:
        sensitivity = st.selectbox(
            "Detection Sensitivity",
            ["Low", "Medium", "High"],
            index=1
        )

    # Analysis section
    if st.button("🔍 Analyze for Bias & Hallucinations", type="primary"):
        if analysis_text.strip():
            with st.spinner("Analyzing text for bias and hallucinations..."):
                try:
                    detector = components['bias_detector']

                    # Update detector configuration based on user settings
                    detector.config['enable_gender_bias'] = detect_gender
                    detector.config['enable_religion_bias'] = detect_religion
                    detector.config['enable_caste_bias'] = detect_caste
                    detector.config['enable_citation_check'] = detect_citations

                    # Adjust thresholds based on sensitivity
                    if sensitivity == "Low":
                        detector.config['bias_threshold'] = 0.8
                        detector.config['hallucination_threshold'] = 0.8
                    elif sensitivity == "High":
                        detector.config['bias_threshold'] = 0.4
                        detector.config['hallucination_threshold'] = 0.5

                    # Perform analysis
                    report = detector.analyze_text(analysis_text)

                    # Display results
                    st.markdown("### 📈 Analysis Results")

                    # Overall scores
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(
                            "Overall Bias Score",
                            f"{report.overall_bias_score:.3f}"
                        )

                    with col2:
                        st.metric(
                            "Hallucination Score",
                            f"{report.overall_hallucination_score:.3f}"
                        )

                    with col3:
                        total_issues = len(report.bias_instances) + len(report.hallucination_instances)
                        st.metric("Total Issues Found", total_issues)

                    # Summary
                    st.markdown("### 📝 Analysis Summary")
                    st.info(report.summary)

                    # Detailed results in tabs
                    if report.bias_instances or report.hallucination_instances:
                        tab1, tab2, tab3, tab4 = st.tabs([
                            "⚠️ Bias Issues",
                            "🔍 Hallucinations",
                            "📈 Detailed Analysis",
                            "💡 Recommendations"
                        ])

                        with tab1:
                            if report.bias_instances:
                                st.markdown(f"#### Found {len(report.bias_instances)} Bias Instance(s)")

                                for i, instance in enumerate(report.bias_instances, 1):
                                    # Color code by severity
                                    if instance.severity.value == "critical":
                                        severity_color = "🔴"
                                    elif instance.severity.value == "high":
                                        severity_color = "🟠"
                                    elif instance.severity.value == "medium":
                                        severity_color = "🟡"
                                    else:
                                        severity_color = "🟢"

                                    with st.expander(f"{severity_color} Issue {i}: {instance.bias_type.value.title()} Bias ({instance.severity.value})"):
                                        st.markdown(f"**Text:** '{instance.text_snippet}'")
                                        st.markdown(f"**Explanation:** {instance.explanation}")
                                        st.markdown(f"**Confidence:** {instance.confidence:.2f}")

                                        if instance.suggested_alternative:
                                            st.markdown(f"**Suggestion:** {instance.suggested_alternative}")

                                        # Position in text
                                        st.markdown(f"**Position:** Characters {instance.start_pos}-{instance.end_pos}")
                            else:
                                st.success("✅ No bias detected in the text!")

                        with tab2:
                            if report.hallucination_instances:
                                st.markdown(f"#### Found {len(report.hallucination_instances)} Hallucination(s)")

                                for i, instance in enumerate(report.hallucination_instances, 1):
                                    # Color code by verification status
                                    if instance.verified_status == "not_found":
                                        status_color = "🔴"
                                    elif instance.verified_status == "suspicious":
                                        status_color = "🟠"
                                    else:
                                        status_color = "🟡"

                                    with st.expander(f"{status_color} Citation {i}: {instance.citation}"):
                                        st.markdown(f"**Full Text:** '{instance.text_snippet}'")
                                        st.markdown(f"**Status:** {instance.verified_status.replace('_', ' ').title()}")
                                        st.markdown(f"**Reason:** {instance.reason}")
                                        st.markdown(f"**Confidence:** {instance.confidence:.2f}")
                                        st.markdown(f"**Position:** Characters {instance.start_pos}-{instance.end_pos}")
                            else:
                                st.success("✅ All citations appear to be valid!")

                        with tab3:
                            st.markdown("#### Bias Distribution")

                            if report.bias_instances:
                                # Bias type distribution
                                bias_types = [instance.bias_type.value for instance in report.bias_instances]
                                bias_counts = pd.Series(bias_types).value_counts()

                                fig_bias = px.pie(
                                    values=bias_counts.values,
                                    names=bias_counts.index,
                                    title="Bias Types Distribution"
                                )
                                st.plotly_chart(fig_bias, use_container_width=True)

                                # Severity distribution
                                severity_levels = [instance.severity.value for instance in report.bias_instances]
                                severity_counts = pd.Series(severity_levels).value_counts()

                                fig_severity = px.bar(
                                    x=severity_counts.index,
                                    y=severity_counts.values,
                                    title="Bias Severity Distribution",
                                    color=severity_counts.values,
                                    color_continuous_scale='Reds'
                                )
                                st.plotly_chart(fig_severity, use_container_width=True)

                            if report.hallucination_instances:
                                st.markdown("#### Hallucination Analysis")

                                # Hallucination status distribution
                                hall_status = [instance.verified_status for instance in report.hallucination_instances]
                                hall_counts = pd.Series(hall_status).value_counts()

                                fig_hall = px.bar(
                                    x=hall_counts.index,
                                    y=hall_counts.values,
                                    title="Citation Verification Status",
                                    color=hall_counts.values,
                                    color_continuous_scale='Oranges'
                                )
                                st.plotly_chart(fig_hall, use_container_width=True)

                        with tab4:
                            st.markdown("#### Recommendations for Improvement")

                            for i, recommendation in enumerate(report.recommendations, 1):
                                st.markdown(f"{i}. {recommendation}")

                            # Additional tips
                            st.markdown("#### General Guidelines")
                            st.markdown("""
                            - **Use inclusive language**: Avoid generalizations about gender, religion, or caste
                            - **Verify all citations**: Double-check case names, section numbers, and legal references
                            - **Be factually consistent**: Ensure statements don't contradict each other
                            - **Use neutral tone**: Maintain objectivity in legal writing
                            - **Cite authoritative sources**: Reference only verified legal databases and official publications
                            """)

                    else:
                        st.success("✅ Excellent! No bias or hallucinations detected in the text.")
                        st.balloons()

                    # Export report
                    st.markdown("### 💾 Export Analysis Report")

                    report_data = {
                        'analysis_timestamp': datetime.now().isoformat(),
                        'text_analyzed': analysis_text,
                        'overall_bias_score': report.overall_bias_score,
                        'overall_hallucination_score': report.overall_hallucination_score,
                        'summary': report.summary,
                        'bias_instances': [
                            {
                                'type': instance.bias_type.value,
                                'severity': instance.severity.value,
                                'text': instance.text_snippet,
                                'explanation': instance.explanation,
                                'confidence': instance.confidence,
                                'suggestion': instance.suggested_alternative
                            } for instance in report.bias_instances
                        ],
                        'hallucination_instances': [
                            {
                                'citation': instance.citation,
                                'text': instance.text_snippet,
                                'status': instance.verified_status,
                                'reason': instance.reason,
                                'confidence': instance.confidence
                            } for instance in report.hallucination_instances
                        ],
                        'recommendations': report.recommendations,
                        'settings': {
                            'gender_detection': detect_gender,
                            'religion_detection': detect_religion,
                            'caste_detection': detect_caste,
                            'citation_verification': detect_citations,
                            'sensitivity': sensitivity
                        }
                    }

                    st.download_button(
                        "📄 Download Analysis Report",
                        data=json.dumps(report_data, indent=2),
                        file_name=f"bias_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )

                except Exception as e:
                    st.error(f"Error during bias analysis: {e}")
                    import traceback
                    st.text(traceback.format_exc())
        else:
            st.warning("Please enter text to analyze.")

def show_multilingual_chat(components):
    """Display multilingual chat interface."""
    st.markdown("## 🌐 Multilingual Legal Chat")

    if 'multilingual_chat' not in components or components['multilingual_chat'] is None:
        st.error("❌ Multilingual Chat not available")
        return

    st.markdown("""
    This module provides multilingual support for legal queries in:
    - **Hindi** (हिन्दी)
    - **Tamil** (தமிழ்)
    - **Malayalam** (മലയാളം)
    - **Bengali** (বাংলা)
    - And 6 more Indian languages
    """)

    # Language selection
    chat = components['multilingual_chat']
    supported_langs = chat.get_supported_languages()

    col1, col2 = st.columns([2, 1])

    with col1:
        query = st.text_input(
            "Ask your legal question in any supported language:",
            placeholder="What is bail? / जमानत क्या है? / பிணை என்றால் என்ன?"
        )

    with col2:
        st.markdown("**Supported Languages:**")
        for lang in supported_langs[:5]:  # Show first 5
            st.markdown(f"• {lang['native_name']}")

    if st.button("💬 Ask Question", type="primary"):
        if query.strip():
            with st.spinner("Processing multilingual query..."):
                try:
                    response = chat.process_query(query)

                    st.markdown("### 📝 Response")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Detected Language", response.detected_language.value)
                    with col2:
                        st.metric("Translation Confidence", f"{response.translation_confidence:.2f}")

                    if response.code_switching_detected:
                        st.info("🔄 Code-switching detected in your query")

                    st.markdown("**English Response:**")
                    st.markdown(response.english_response)

                    if response.native_response != response.english_response:
                        st.markdown("**Native Language Response:**")
                        st.markdown(response.native_response)

                except Exception as e:
                    st.error(f"Error processing query: {e}")
        else:
            st.warning("Please enter a question.")

def show_section_predictor(components):
    """Display section predictor interface."""
    st.markdown("## ⚖️ Legal Section Predictor")

    if 'section_predictor' not in components or components['section_predictor'] is None:
        st.error("❌ Section Predictor not available")
        return

    st.markdown("""
    This module predicts applicable IPC/CrPC sections based on case facts:
    - **Primary Sections**: Most relevant legal sections
    - **Secondary Sections**: Related provisions
    - **Procedural Sections**: Required procedures
    """)

    # Input section
    case_facts = st.text_area(
        "Enter case facts:",
        height=150,
        placeholder="Describe what happened in the case..."
    )

    # Sample cases
    sample_facts = {
        "Murder Case": "The accused intentionally killed the victim with a knife after a heated argument.",
        "Fraud Case": "The accused cheated the complainant by making false representations about investment returns.",
        "Domestic Violence": "The husband subjected his wife to physical and mental cruelty, demanding dowry."
    }

    st.markdown("**Sample Cases:**")
    for case_name, facts in sample_facts.items():
        if st.button(f"Load {case_name}", key=f"load_facts_{case_name}"):
            case_facts = facts
            st.rerun()

    if st.button("⚖️ Predict Sections", type="primary"):
        if case_facts.strip():
            with st.spinner("Predicting applicable legal sections..."):
                try:
                    predictor = components['section_predictor']
                    recommendation = predictor.predict_sections(case_facts)

                    st.markdown("### 📊 Prediction Results")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Case Category", recommendation.case_category)
                    with col2:
                        st.metric("Severity Assessment", recommendation.severity_assessment)

                    if recommendation.primary_sections:
                        st.markdown("### ⚖️ Primary Sections")
                        for i, pred in enumerate(recommendation.primary_sections, 1):
                            with st.expander(f"{i}. Section {pred.section.section_number} - {pred.section.title}"):
                                st.markdown(f"**Code:** {pred.section.code.value}")
                                st.markdown(f"**Confidence:** {pred.confidence:.2f}")
                                st.markdown(f"**Reasoning:** {pred.reasoning}")
                                if pred.section.punishment_details:
                                    st.markdown(f"**Punishment:** {pred.section.punishment_details}")

                    st.markdown(f"**Analysis:** {recommendation.overall_analysis}")

                except Exception as e:
                    st.error(f"Error predicting sections: {e}")
        else:
            st.warning("Please enter case facts.")

def show_performance_evaluation(components):
    """Display performance evaluation interface."""
    st.markdown("## 📈 Performance Evaluation")

    if 'evaluator' not in components or components['evaluator'] is None:
        st.error("❌ Performance Evaluator not available")
        return

    st.markdown("""
    This module provides comprehensive evaluation of the legal AI system:
    - **ROUGE Scores**: Content similarity metrics
    - **Legal Accuracy**: Legal entity and reasoning accuracy
    - **Factual Consistency**: Consistency with source material
    - **Readability**: Text readability assessment
    """)

    # Task type selection
    st.markdown("### 🎯 Evaluation Setup")

    from src.evaluation.comprehensive_evaluator import TaskType

    col1, col2 = st.columns(2)

    with col1:
        task_type = st.selectbox(
            "Select Task Type:",
            [TaskType.SUMMARIZATION, TaskType.QUESTION_ANSWERING, TaskType.CASE_ANALYSIS, TaskType.SECTION_PREDICTION],
            format_func=lambda x: x.value.replace('_', ' ').title()
        )

    with col2:
        evaluation_mode = st.selectbox(
            "Evaluation Mode:",
            ["Quick Evaluation", "Comprehensive Analysis", "Benchmark Comparison"]
        )

    # Input sections
    st.markdown("### 📝 Input Data")

    tab1, tab2, tab3 = st.tabs(["📄 Predictions", "🎯 References", "📚 Sources"])

    with tab1:
        st.markdown("**System Predictions/Outputs:**")
        predictions_text = st.text_area(
            "Enter the AI system's output:",
            height=150,
            placeholder="The court ruled that the accused is guilty of murder under Section 302 IPC...",
            key="predictions"
        )

        # Sample predictions
        sample_predictions = {
            "Legal Summary": "The Supreme Court held that the defendant committed murder as defined in Section 302 of the Indian Penal Code. The evidence was sufficient to prove guilt beyond reasonable doubt. The court considered all circumstances and found the prosecution's case compelling.",
            "Case Analysis": "Based on the facts presented, this case involves intentional killing with premeditation. The applicable section is IPC 302 with punishment of life imprisonment or death. The defendant's actions clearly demonstrate mens rea and actus reus.",
            "Legal Opinion": "The accused's actions constitute murder under Section 302 IPC. The prosecution has established mens rea and actus reus beyond reasonable doubt. The defense arguments regarding self-defense were not substantiated by evidence.",
            "Fraud Case Summary": "The accused cheated the complainant by making false representations about investment returns, dishonestly inducing delivery of Rs. 5 lakhs. This constitutes cheating under Section 420 IPC with imprisonment up to 7 years.",
            "Domestic Violence Analysis": "The husband subjected his wife to physical and mental cruelty, demanding dowry. This falls under Section 498A IPC (cruelty by husband) and Domestic Violence Act 2005, with imprisonment up to 3 years.",
            "Bail Application Summary": "The accused applied for anticipatory bail under Section 438 CrPC. Considering the nature of offense, likelihood of absconding, and tampering with evidence, bail is granted with conditions including surrender of passport."
        }

        st.markdown("**Sample Predictions:**")
        for pred_name, pred_text in sample_predictions.items():
            if st.button(f"Load {pred_name}", key=f"load_pred_{pred_name}"):
                st.session_state.predictions = pred_text
                st.rerun()

    with tab2:
        st.markdown("**Reference/Ground Truth:**")
        references_text = st.text_area(
            "Enter the reference/expected output:",
            height=150,
            placeholder="The Supreme Court convicted the defendant of murder under Section 302 IPC...",
            key="references"
        )

        # Sample references
        sample_references = {
            "Legal Summary Ref": "The Supreme Court convicted the defendant of murder under Section 302 of the Indian Penal Code. The court found sufficient evidence to establish guilt beyond reasonable doubt. All legal requirements for murder conviction were satisfied.",
            "Case Analysis Ref": "This case involves premeditated murder. The applicable provision is Section 302 IPC with penalties including life imprisonment or capital punishment. The defendant's intent and actions were clearly established.",
            "Legal Opinion Ref": "The defendant's conduct constitutes murder under IPC Section 302. The prosecution successfully proved both intent and action beyond reasonable doubt. The defense failed to establish any mitigating circumstances.",
            "Fraud Case Ref": "The accused committed cheating under Section 420 IPC by making false representations about investment schemes, dishonestly inducing the victim to part with Rs. 5 lakhs. The offense carries imprisonment up to 7 years and fine.",
            "Domestic Violence Ref": "The husband's conduct amounts to cruelty under Section 498A IPC and violates the Domestic Violence Act 2005. Physical and mental torture for dowry demands constitutes a serious offense with imprisonment up to 3 years.",
            "Bail Application Ref": "The accused sought anticipatory bail under Section 438 CrPC. Given the serious nature of the offense and risk factors, bail was granted with strict conditions including passport surrender and regular reporting."
        }

        st.markdown("**Sample References:**")
        for ref_name, ref_text in sample_references.items():
            if st.button(f"Load {ref_name}", key=f"load_ref_{ref_name}"):
                st.session_state.references = ref_text
                st.rerun()

    with tab3:
        st.markdown("**Source Documents:**")
        sources_text = st.text_area(
            "Enter the source material:",
            height=150,
            placeholder="The defendant killed the victim with premeditation. The Supreme Court convicted him of murder...",
            key="sources"
        )

        # Sample sources
        sample_sources = {
            "Murder Case Facts": "The defendant killed the victim with a knife after planning the attack for several days. Witnesses testified about the defendant's threats. The Supreme Court found him guilty of murder and sentenced him to life imprisonment. Evidence included CCTV footage and forensic reports.",
            "Court Judgment": "The accused intentionally caused the death of the deceased with premeditation. The court convicted him under Section 302 IPC for murder with life imprisonment. The prosecution proved mens rea and actus reus beyond reasonable doubt through circumstantial evidence.",
            "Legal Document": "Evidence shows premeditated killing by the accused including purchase of weapon, surveillance of victim, and deliberate planning. The Supreme Court held this constitutes murder under IPC 302 with appropriate sentencing considering aggravating factors.",
            "Fraud Case Facts": "The accused operated a fake investment scheme promising 30% returns. He collected Rs. 5 lakhs from the complainant through false documents and forged certificates. When confronted, he disappeared. Police investigation revealed multiple similar complaints.",
            "Domestic Violence Facts": "The husband regularly beat his wife and demanded additional dowry of Rs. 2 lakhs. He threatened to throw acid and prevented her from meeting family. Medical reports show injuries consistent with domestic violence. The wife filed complaint under DV Act 2005.",
            "Bail Case Facts": "The accused is charged with financial fraud involving Rs. 50 lakhs. He has no prior criminal record but owns properties worth Rs. 2 crores. Investigation is ongoing with possibility of more victims. He applied for anticipatory bail citing cooperation with investigation."
        }

        st.markdown("**Sample Sources:**")
        for src_name, src_text in sample_sources.items():
            if st.button(f"Load {src_name}", key=f"load_src_{src_name}"):
                st.session_state.sources = src_text
                st.rerun()

    # Use session state values if available
    if 'predictions' in st.session_state:
        predictions_text = st.session_state.predictions
    if 'references' in st.session_state:
        references_text = st.session_state.references
    if 'sources' in st.session_state:
        sources_text = st.session_state.sources

    # Evaluation button
    if st.button("📊 Run Evaluation", type="primary"):
        if predictions_text.strip() and references_text.strip():
            with st.spinner("Running comprehensive evaluation..."):
                try:
                    evaluator = components['evaluator']

                    # Prepare data
                    predictions = [predictions_text]
                    references = [references_text]
                    sources = [sources_text] if sources_text.strip() else ["No source provided"]

                    # Run evaluation
                    task_eval = evaluator.evaluate_task(task_type, predictions, references, sources)

                    # Display results
                    st.markdown("### 📊 Evaluation Results")

                    # Overall score
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(
                            "Overall Score",
                            f"{task_eval.overall_score:.3f}",
                            help="Weighted average of all metrics"
                        )

                    with col2:
                        st.metric(
                            "Metrics Evaluated",
                            len(task_eval.results),
                            help="Number of evaluation metrics applied"
                        )

                    with col3:
                        performance_level = "Excellent" if task_eval.overall_score > 0.8 else "Good" if task_eval.overall_score > 0.6 else "Fair" if task_eval.overall_score > 0.4 else "Poor"
                        st.metric(
                            "Performance Level",
                            performance_level,
                            help="Qualitative assessment based on score"
                        )

                    # Detailed metrics
                    st.markdown("### 📈 Detailed Metrics")

                    # Create metrics dataframe for better visualization
                    import pandas as pd

                    metrics_data = []
                    for result in task_eval.results:
                        metric_name = result.metric.value
                        details_str = str(result.details) if isinstance(result.details, dict) else result.details
                        metrics_data.append({
                            "Metric": metric_name.replace('_', ' ').title(),
                            "Score": f"{result.score:.3f}",
                            "Confidence": f"{result.confidence:.2f}",
                            "Details": details_str[:100] + "..." if len(str(details_str)) > 100 else details_str
                        })

                    df = pd.DataFrame(metrics_data)
                    st.dataframe(df, use_container_width=True)

                    # Metric breakdown
                    st.markdown("### 🔍 Metric Breakdown")

                    for result in task_eval.results:
                        metric_name = result.metric.value
                        with st.expander(f"📊 {metric_name.replace('_', ' ').title()} - Score: {result.score:.3f}"):
                            col1, col2 = st.columns([1, 2])

                            with col1:
                                st.metric("Score", f"{result.score:.3f}")
                                st.metric("Confidence", f"{result.confidence:.2f}")

                            with col2:
                                st.markdown("**Details:**")
                                if isinstance(result.details, dict):
                                    for key, value in result.details.items():
                                        st.markdown(f"- **{key}**: {value}")
                                else:
                                    st.markdown(str(result.details))

                    # Generate comprehensive report
                    if evaluation_mode == "Comprehensive Analysis":
                        st.markdown("### 📋 Comprehensive Report")

                        report = evaluator.generate_comprehensive_report([task_eval])

                        # System performance
                        st.markdown("#### 🎯 System Performance")
                        perf_col1, perf_col2 = st.columns(2)

                        with perf_col1:
                            st.metric("System Score", f"{report.overall_performance['system_score']:.3f}")

                        with perf_col2:
                            st.metric("Tasks Evaluated", len(report.task_evaluations))

                        # Performance summary
                        st.markdown("#### 📝 Performance Summary")
                        for summary_point in report.performance_summary:
                            st.markdown(f"• {summary_point}")

                        # Recommendations
                        st.markdown("#### 💡 Recommendations")
                        for recommendation in report.recommendations:
                            st.info(f"💡 {recommendation}")

                    # Visualization
                    if len(task_eval.results) > 1:
                        st.markdown("### 📊 Performance Visualization")

                        import plotly.graph_objects as go
                        import plotly.express as px

                        # Radar chart
                        metrics = [result.metric.value for result in task_eval.results]
                        scores = [result.score for result in task_eval.results]

                        fig = go.Figure()

                        fig.add_trace(go.Scatterpolar(
                            r=scores,
                            theta=[m.replace('_', ' ').title() for m in metrics],
                            fill='toself',
                            name='Performance Scores'
                        ))

                        fig.update_layout(
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                    range=[0, 1]
                                )
                            ),
                            showlegend=True,
                            title="Performance Radar Chart"
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        # Bar chart
                        fig_bar = px.bar(
                            x=[m.replace('_', ' ').title() for m in metrics],
                            y=scores,
                            title="Metric Scores Comparison",
                            labels={'x': 'Metrics', 'y': 'Scores'}
                        )
                        fig_bar.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig_bar, use_container_width=True)

                except Exception as e:
                    st.error(f"Error during evaluation: {e}")
                    import traceback
                    st.text(traceback.format_exc())
        else:
            st.warning("Please provide both predictions and references for evaluation.")

def show_enhanced_chat(components):
    """Display enhanced chat interface."""
    st.markdown("## 💬 Enhanced Legal Chat")

    st.markdown("""
    🎆 **Unified AI Assistant** combining all 6 novel components:
    - 🧠 **Chain-of-Thought reasoning** for structured analysis
    - 🔍 **Bias detection** for fair responses
    - 🌐 **Multilingual support** for Indian languages
    - ⚖️ **Section prediction** for legal guidance
    - 📊 **Performance evaluation** for quality assurance
    - 📈 **Dual-view summaries** for accessibility
    """)

    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if 'enhanced_mode' not in st.session_state:
        st.session_state.enhanced_mode = True

    # Chat configuration
    st.markdown("### ⚙️ Chat Configuration")

    col1, col2, col3 = st.columns(3)

    with col1:
        enhanced_mode = st.checkbox(
            "🎆 Enhanced Mode",
            value=st.session_state.enhanced_mode,
            help="Enable all 6 novel components for comprehensive analysis"
        )
        st.session_state.enhanced_mode = enhanced_mode

    with col2:
        auto_detect_language = st.checkbox(
            "🌐 Auto-detect Language",
            value=True,
            help="Automatically detect and respond in user's language"
        )

    with col3:
        bias_checking = st.checkbox(
            "🔍 Real-time Bias Check",
            value=True,
            help="Check responses for bias and hallucinations"
        )

    # Chat interface
    st.markdown("### 💬 Legal AI Assistant")

    # Display chat history
    chat_container = st.container()

    with chat_container:
        for i, message in enumerate(st.session_state.chat_history):
            if message['role'] == 'user':
                st.markdown(f"👤 **You:** {message['content']}")
                if 'language' in message:
                    st.caption(f"🌐 Language: {message['language']}")
            else:
                st.markdown(f"🤖 **Legal AI:** {message['content']}")

                # Show enhanced analysis if available
                if 'analysis' in message:
                    with st.expander("🔍 View Enhanced Analysis"):
                        analysis = message['analysis']

                        if 'cot_reasoning' in analysis:
                            st.markdown("**🧠 Chain-of-Thought Analysis:**")
                            cot = analysis['cot_reasoning']
                            st.markdown(f"- **Verdict:** {cot.get('verdict', 'N/A')}")
                            st.markdown(f"- **Confidence:** {cot.get('confidence', 'N/A')}")

                        if 'bias_report' in analysis:
                            st.markdown("**🔍 Bias Analysis:**")
                            bias = analysis['bias_report']
                            st.markdown(f"- **Bias Score:** {bias.get('score', 'N/A')}")
                            st.markdown(f"- **Issues Found:** {bias.get('issues', 'None')}")

                        if 'sections' in analysis:
                            st.markdown("**⚖️ Relevant Sections:**")
                            for section in analysis['sections']:
                                st.markdown(f"- {section}")

                        if 'language_info' in analysis:
                            st.markdown("**🌐 Language Info:**")
                            lang = analysis['language_info']
                            st.markdown(f"- **Detected:** {lang.get('detected', 'English')}")
                            st.markdown(f"- **Confidence:** {lang.get('confidence', 'N/A')}")

    # Input section
    st.markdown("---")

    # Sample questions
    st.markdown("**💡 Try these sample questions:**")

    sample_questions = {
        "Legal Analysis": "Analyze a murder case under Section 302 IPC with Chain-of-Thought reasoning",
        "Multilingual Query": "जमानत क्या है और कैसे मिलती है?",
        "Section Prediction": "What sections apply when someone cheats in a business transaction?",
        "Bias Check": "Check this text for bias: 'The emotional woman clearly provoked the incident'",
        "Case Summary": "Summarize this case in both legal and simple language"
    }

    cols = st.columns(len(sample_questions))
    for i, (q_type, question) in enumerate(sample_questions.items()):
        with cols[i]:
            if st.button(q_type, key=f"sample_{i}"):
                st.session_state.current_question = question
                st.rerun()

    # User input
    user_input = st.text_area(
        "Ask your legal question:",
        height=100,
        placeholder="Ask anything legal - I'll analyze it with Chain-of-Thought reasoning, check for bias, predict relevant sections, and respond in your language!",
        value=st.session_state.get('current_question', '')
    )

    # Clear current question after displaying
    if 'current_question' in st.session_state:
        del st.session_state.current_question

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        send_button = st.button("🚀 Send Message", type="primary")

    with col2:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

    with col3:
        if st.button("💾 Export Chat"):
            chat_export = "\n".join([
                f"{msg['role'].title()}: {msg['content']}"
                for msg in st.session_state.chat_history
            ])
            st.download_button(
                "Download Chat",
                chat_export,
                file_name="legal_chat_history.txt",
                mime="text/plain"
            )

    # Process user input
    if send_button and user_input.strip():
        # Add user message to history
        user_message = {'role': 'user', 'content': user_input}

        # Detect language if enabled
        if auto_detect_language and components.get('multilingual_chat'):
            try:
                ml_response = components['multilingual_chat'].process_query(user_input)
                user_message['language'] = ml_response.detected_language.value
            except:
                user_message['language'] = 'English'

        st.session_state.chat_history.append(user_message)

        # Generate enhanced response
        with st.spinner("🤖 Generating enhanced legal analysis..."):
            try:
                response_content = ""
                analysis = {}

                # 1. Chain-of-Thought Analysis (if applicable)
                if enhanced_mode and components.get('cot_reasoner') and any(keyword in user_input.lower() for keyword in ['case', 'analyze', 'verdict', 'guilty', 'murder', 'section']):
                    try:
                        cot_result = components['cot_reasoner'].analyze_case(user_input)
                        analysis['cot_reasoning'] = {
                            'verdict': cot_result.final_verdict.value,
                            'confidence': f"{cot_result.overall_confidence:.2f}"
                        }
                        response_content += f"🧠 **Chain-of-Thought Analysis:** Based on structured legal reasoning, the predicted outcome is {cot_result.final_verdict.value} with {cot_result.overall_confidence:.2f} confidence.\n\n"
                    except Exception as e:
                        st.warning(f"CoT analysis failed: {str(e)[:50]}...")

                # 2. Section Prediction (if applicable)
                if enhanced_mode and components.get('section_predictor') and any(keyword in user_input.lower() for keyword in ['section', 'law', 'ipc', 'crpc', 'apply', 'relevant']):
                    try:
                        section_rec = components['section_predictor'].predict_sections(user_input)
                        if section_rec.primary_sections:
                            sections = [f"Section {s.section.section_number} ({s.section.title})" for s in section_rec.primary_sections[:3]]
                            analysis['sections'] = sections
                            response_content += f"⚖️ **Relevant Legal Sections:** {', '.join(sections)}\n\n"
                    except Exception as e:
                        st.warning(f"Section prediction failed: {str(e)[:50]}...")

                # 3. Multilingual Response
                if components.get('multilingual_chat'):
                    try:
                        ml_response = components['multilingual_chat'].process_query(user_input)
                        analysis['language_info'] = {
                            'detected': ml_response.detected_language.value,
                            'confidence': f"{ml_response.translation_confidence:.2f}"
                        }

                        # Use the multilingual response as base
                        if not response_content:
                            response_content = ml_response.english_response

                        # Add native language response if different
                        if ml_response.native_response != ml_response.english_response:
                            response_content += f"\n\n🌐 **In your language:** {ml_response.native_response}"
                    except Exception as e:
                        st.warning(f"Multilingual processing failed: {str(e)[:50]}...")

                # 4. Bias Detection
                if bias_checking and components.get('bias_detector'):
                    try:
                        bias_report = components['bias_detector'].analyze_text(response_content or user_input)
                        analysis['bias_report'] = {
                            'score': f"{bias_report.overall_bias_score:.3f}",
                            'issues': len(bias_report.bias_instances)
                        }

                        if bias_report.overall_bias_score > 0.3:
                            response_content += f"\n\n🔍 **Bias Alert:** Potential bias detected (score: {bias_report.overall_bias_score:.3f}). Please review the response carefully."
                    except Exception as e:
                        st.warning(f"Bias detection failed: {str(e)[:50]}...")

                # 5. Dual-view Summary (if requested)
                if enhanced_mode and components.get('visualizer') and any(keyword in user_input.lower() for keyword in ['summarize', 'summary', 'explain', 'simple']):
                    try:
                        dual_summary = components['visualizer'].create_dual_view_summary(user_input)
                        response_content += f"\n\n📈 **Professional View:** {dual_summary.legal_view.content[:200]}...\n"
                        response_content += f"📈 **Simplified View:** {dual_summary.simplified_view.content[:200]}..."
                    except Exception as e:
                        st.warning(f"Dual-view summary failed: {str(e)[:50]}...")

                # Default response if no specific analysis
                if not response_content:
                    response_content = f"I understand you're asking about: '{user_input}'. This appears to be a legal query. While I can provide general information, please consult with a qualified legal professional for specific legal advice."

                # Add assistant message to history
                assistant_message = {
                    'role': 'assistant',
                    'content': response_content,
                    'analysis': analysis
                }

                st.session_state.chat_history.append(assistant_message)

                # Rerun to show the new message
                st.rerun()

            except Exception as e:
                st.error(f"Error generating response: {e}")
                import traceback
                st.text(traceback.format_exc())

def show_system_analytics(components):
    """Display system analytics interface."""
    st.markdown("## 🔬 System Analytics")

    st.markdown("""
    System performance and usage analytics:
    - Component status monitoring
    - Performance metrics
    - Usage statistics
    - Error tracking
    """)

    # Component status
    st.markdown("### 🔧 Component Status")

    for comp_name, comp in components.items():
        if comp is not None:
            st.success(f"✅ {comp_name}: Active")
        else:
            st.error(f"❌ {comp_name}: Inactive")

    # System metrics
    st.markdown("### 📊 System Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        active_components = sum(1 for comp in components.values() if comp is not None)
        st.metric("Active Components", f"{active_components}/{len(components)}")

    with col2:
        st.metric("Supported Languages", "10")

    with col3:
        st.metric("Legal Sections", "50+")

    with col4:
        st.metric("Evaluation Metrics", "10+")

if __name__ == "__main__":
    main()
