"""
Streamlit web application for Legal LLM system.
"""

import streamlit as st
import sys
from pathlib import Path
import logging
import json
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.chatbot.legal_chatbot import LegalChatbot
from src.models.summarizer import LegalSummarizer
from src.rag.legal_rag import LegalRAG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Legal AI Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .source-box {
        background-color: #f5f5f5;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_chatbot():
    """Load the legal chatbot with caching."""
    try:
        # Try to load fine-tuned model and RAG system
        chatbot = LegalChatbot(
            summarizer_model_path="./results",  # Adjust path as needed
            rag_index_path="./data/rag_index"   # Adjust path as needed
        )
        return chatbot, "Full system loaded (Fine-tuned model + RAG)"
    except Exception as e:
        logger.warning(f"Failed to load full system: {e}")
        try:
            # Fallback to base model only
            chatbot = LegalChatbot()
            chatbot.load_summarizer(None)  # This will load base model
            return chatbot, "Base model loaded (Limited functionality)"
        except Exception as e2:
            logger.error(f"Failed to load base model: {e2}")
            return None, f"Failed to load system: {e2}"

def main():
    # Header
    st.markdown('<h1 class="main-header">⚖️ Legal AI Assistant</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 System Status")
        
        # Load chatbot
        chatbot, status_message = load_chatbot()
        
        if chatbot:
            st.success(status_message)
        else:
            st.error(status_message)
            st.stop()
        
        st.header("📋 Features")
        st.markdown("""
        - **Document Summarization**: Upload legal documents for AI-powered summaries
        - **Legal Research**: Search through legal precedents and case law
        - **Case Analysis**: Get detailed analysis of legal issues
        - **Interactive Chat**: Ask questions about legal concepts
        """)
        
        st.header("⚙️ Settings")
        include_sources = st.checkbox("Include sources in responses", value=True)
        max_sources = st.slider("Maximum sources to show", 1, 10, 5)
        
        if st.button("Clear Chat History"):
            chatbot.clear_history()
            st.rerun()
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📄 Document Summarization", "🔍 Legal Search", "📊 Analytics"])
    
    with tab1:
        st.header("Legal AI Chat Assistant")
        
        # Initialize chat history in session state
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Show sources if available
                if "sources" in message and message["sources"] and include_sources:
                    with st.expander("📚 Sources"):
                        for i, source in enumerate(message["sources"][:max_sources], 1):
                            st.markdown(f"""
                            <div class="source-box">
                                <strong>{i}. Case ID:</strong> {source.get('doc_id', 'N/A')}<br>
                                <strong>Jurisdiction:</strong> {source.get('jurisdiction', 'N/A')}<br>
                                <strong>Relevance Score:</strong> {source.get('score', 0):.3f}
                            </div>
                            """, unsafe_allow_html=True)
        
        # Chat input
        if prompt := st.chat_input("Ask me about legal matters..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get bot response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = chatbot.chat(prompt, include_sources=include_sources)
                
                st.markdown(response["answer"])
                
                # Add assistant message to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response["answer"],
                    "sources": response.get("sources", [])
                })
                
                # Show confidence if available
                if "confidence" in response:
                    st.info(f"Confidence: {response['confidence']:.2%}")
    
    with tab2:
        st.header("📄 Legal Document Summarization")
        
        # Document input methods
        input_method = st.radio("Choose input method:", ["Text Input", "File Upload"])
        
        if input_method == "Text Input":
            document_text = st.text_area(
                "Paste your legal document here:",
                height=300,
                placeholder="Enter the full text of the legal document you want to summarize..."
            )
        else:
            uploaded_file = st.file_uploader(
                "Upload a legal document",
                type=['txt', 'pdf', 'docx'],
                help="Supported formats: TXT, PDF, DOCX"
            )
            
            document_text = ""
            if uploaded_file is not None:
                if uploaded_file.type == "text/plain":
                    document_text = str(uploaded_file.read(), "utf-8")
                else:
                    st.warning("PDF and DOCX support coming soon. Please use TXT files for now.")
        
        # Summarization options
        col1, col2 = st.columns(2)
        with col1:
            summary_length = st.selectbox("Summary Length:", ["Short", "Medium", "Detailed"])
        with col2:
            focus_area = st.selectbox("Focus Area:", ["General", "Facts", "Legal Issues", "Conclusion"])
        
        if st.button("Generate Summary", type="primary"):
            if document_text.strip():
                with st.spinner("Generating summary..."):
                    try:
                        # Create focused prompt based on user selection
                        if focus_area != "General":
                            focused_prompt = f"Focus on the {focus_area.lower()} when summarizing this document:\n\n{document_text}"
                        else:
                            focused_prompt = document_text
                        
                        summary = chatbot.summarizer.summarize(focused_prompt) if chatbot.summarizer else "Summarizer not available"
                        
                        st.success("Summary generated successfully!")
                        st.markdown("### 📋 Summary")
                        st.markdown(summary)
                        
                        # Download option
                        st.download_button(
                            label="Download Summary",
                            data=summary,
                            file_name=f"legal_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
                        
                    except Exception as e:
                        st.error(f"Error generating summary: {e}")
            else:
                st.warning("Please provide a document to summarize.")
    
    with tab3:
        st.header("🔍 Legal Research & Case Search")
        
        # Search interface
        search_query = st.text_input(
            "Enter your legal research query:",
            placeholder="e.g., property tax deduction cases, income tax appeals, contract disputes"
        )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            search_type = st.selectbox("Search Type:", ["General Search", "Similar Cases", "Precedent Analysis"])
        with col2:
            jurisdiction_filter = st.selectbox("Jurisdiction:", ["All", "Indian", "UK"])
        with col3:
            max_results = st.slider("Max Results:", 1, 20, 10)
        
        if st.button("Search Legal Database", type="primary"):
            if search_query.strip():
                with st.spinner("Searching legal database..."):
                    try:
                        if search_type == "Similar Cases":
                            results = chatbot.get_similar_cases(search_query, max_results)
                            
                            if results:
                                st.success(f"Found {len(results)} similar cases")
                                
                                for i, case in enumerate(results, 1):
                                    with st.expander(f"📁 Case {i}: {case['doc_id']} (Score: {case['score']:.3f})"):
                                        st.markdown(f"**Jurisdiction:** {case['jurisdiction']}")
                                        st.markdown(f"**Case Type:** {case['case_type']}")
                                        st.markdown(f"**Summary:** {case['summary'][:500]}...")
                                        st.markdown(f"**Relevant Content:** {case['relevant_content'][:300]}...")
                            else:
                                st.info("No similar cases found. Try different search terms.")
                        
                        else:
                            # General search using RAG
                            if chatbot.rag_system:
                                search_results = chatbot.rag_system.search(search_query, max_results)
                                
                                if search_results:
                                    st.success(f"Found {len(search_results)} relevant documents")
                                    
                                    for i, result in enumerate(search_results, 1):
                                        with st.expander(f"📄 Document {i}: {result['metadata']['doc_id']} (Relevance: {result['score']:.3f})"):
                                            st.markdown(f"**Jurisdiction:** {result['metadata']['jurisdiction']}")
                                            st.markdown(f"**Source:** {result['metadata']['source']}")
                                            st.markdown(f"**Content:** {result['content'][:500]}...")
                                else:
                                    st.info("No relevant documents found. Try different search terms.")
                            else:
                                st.warning("RAG system not available. Please ensure the system is properly configured.")
                    
                    except Exception as e:
                        st.error(f"Search error: {e}")
            else:
                st.warning("Please enter a search query.")
    
    with tab4:
        st.header("📊 System Analytics")
        
        # System statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Chat Sessions", len(st.session_state.get("messages", [])) // 2)
        
        with col2:
            if chatbot.rag_system and hasattr(chatbot.rag_system, 'vector_store'):
                total_docs = chatbot.rag_system.vector_store.ntotal if chatbot.rag_system.vector_store else 0
                st.metric("Indexed Documents", total_docs)
            else:
                st.metric("Indexed Documents", "N/A")
        
        with col3:
            st.metric("System Status", "🟢 Online" if chatbot else "🔴 Offline")
        
        # Conversation history
        if st.button("Show Conversation History"):
            history = chatbot.get_conversation_history()
            if history:
                st.json(history)
            else:
                st.info("No conversation history available.")
        
        # Export options
        st.subheader("Export Options")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export Chat History"):
                history = chatbot.get_conversation_history()
                if history:
                    st.download_button(
                        label="Download Chat History",
                        data=json.dumps(history, indent=2),
                        file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
        
        with col2:
            if st.button("Export System Stats"):
                stats = {
                    "timestamp": datetime.now().isoformat(),
                    "total_messages": len(st.session_state.get("messages", [])),
                    "system_status": status_message
                }
                st.download_button(
                    label="Download System Stats",
                    data=json.dumps(stats, indent=2),
                    file_name=f"system_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

if __name__ == "__main__":
    main()
