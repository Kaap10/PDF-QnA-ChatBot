"""
📚 Intelligent PDF QnA Chatbot
Main Streamlit application with beautiful UI and RAG functionality.
"""

import os
import streamlit as st
from dotenv import load_dotenv
from loader import PDFProcessor
from vectorstore import VectorStore
from qa_chain import QAChatbot

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="📚 PDF QnA Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    
    .user-message {
        background-color: #f0f2f6;
        border-left-color: #667eea;
    }
    
    .assistant-message {
        background-color: #e8f4fd;
        border-left-color: #28a745;
    }
    
    .sidebar-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .status-box {
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #ddd;
        background-color: #f9f9f9;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'qa_bot' not in st.session_state:
        st.session_state.qa_bot = None
    if 'pdf_processor' not in st.session_state:
        st.session_state.pdf_processor = None

def check_api_key():
    """Check if OpenAI API key is available."""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        st.error("❌ OpenAI API key not found! Please add your API key to the .env file.")
        st.info("Create a .env file in the project root and add: OPENAI_API_KEY=your_api_key_here")
        return False
    return True

def display_header():
    """Display the main header."""
    st.markdown("""
    <div class="main-header">
        <h1>📚 Intelligent PDF QnA Chatbot</h1>
        <p>Upload your PDF documents and ask intelligent questions using AI-powered RAG technology</p>
    </div>
    """, unsafe_allow_html=True)

def display_sidebar():
    """Display the sidebar with upload and settings."""
    with st.sidebar:
        st.markdown('<div class="sidebar-header"><h3>⚙️ Settings</h3></div>', unsafe_allow_html=True)
        
        # File upload
        st.subheader("📄 Upload PDF")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload any PDF document to start asking questions"
        )
        
        if uploaded_file is not None:
            if st.button("🔄 Process PDF", type="primary"):
                process_pdf(uploaded_file)
        
        # Status information
        st.subheader("📊 Status")
        if st.session_state.vector_store:
            store_info = st.session_state.vector_store.get_store_info()
            st.markdown(f"""
            <div class="status-box">
                <strong>Vector Store:</strong> {store_info.get('status', 'Unknown')}<br>
                <strong>Documents:</strong> {store_info.get('document_count', 0)}<br>
                <strong>Embeddings:</strong> {store_info.get('embedding_dimension', 0)}D
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No PDF processed yet")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
        
        # Clear vector store button
        if st.button("🗑️ Clear Vector Store"):
            if st.session_state.vector_store:
                st.session_state.vector_store.clear_store()
            st.session_state.vector_store = None
            st.rerun()

def process_pdf(uploaded_file):
    """Process the uploaded PDF file."""
    try:
        # Initialize components
        if not st.session_state.pdf_processor:
            st.session_state.pdf_processor = PDFProcessor()
        if not st.session_state.vector_store:
            st.session_state.vector_store = VectorStore()
        if not st.session_state.qa_bot:
            st.session_state.qa_bot = QAChatbot()
        
        # Process PDF
        documents = st.session_state.pdf_processor.process_pdf(uploaded_file)
        
        if documents:
            # Create vector store
            success = st.session_state.vector_store.create_vector_store(documents)
            if success:
                st.session_state.vector_store.save_vector_store()
                st.success("🎉 PDF processed successfully! You can now ask questions.")
                st.rerun()
        
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")

def display_chat_interface():
    """Display the main chat interface."""
    st.subheader("💬 Chat with your PDF")
    
    # Chat input
    if st.session_state.vector_store and st.session_state.vector_store.vector_store:
        user_question = st.chat_input("Ask a question about your PDF...")
        
        if user_question:
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            
            # Get relevant documents
            relevant_docs = st.session_state.vector_store.similarity_search(user_question)
            
            # Generate answer
            if st.session_state.qa_bot:
                answer = st.session_state.qa_bot.answer_question(user_question, relevant_docs)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
            
            st.rerun()
    else:
        st.info("📄 Please upload a PDF document to start chatting!")
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])

def display_features():
    """Display feature highlights."""
    st.subheader("🚀 Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **📄 Smart PDF Processing**
        - Automatic text extraction
        - Intelligent chunking
        - Metadata preservation
        """)
    
    with col2:
        st.markdown("""
        **🔍 Advanced Search**
        - Semantic similarity search
        - FAISS vector database
        - Context-aware retrieval
        """)
    
    with col3:
        st.markdown("""
        **🤖 AI-Powered QnA**
        - GPT-3.5/4 integration
        - RAG technology
        - Accurate, contextual answers
        """)

def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()
    
    # Check API key
    if not check_api_key():
        return
    
    # Display header
    display_header()
    
    # Create two columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Main chat area
        display_chat_interface()
    
    with col2:
        # Sidebar
        display_sidebar()
    
    # Features section
    st.markdown("---")
    display_features()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>Built with ❤️ using Streamlit, LangChain, OpenAI, and FAISS</p>
        <p>Powered by Retrieval Augmented Generation (RAG) Technology</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 