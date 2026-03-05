# =========================================
# Streamlit App: Ebook RAG Chat with Duplicate Handling
# =========================================

import warnings
warnings.filterwarnings("ignore", message="missing ScriptRunContext")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # suppress TensorFlow warnings

import streamlit as st
import tempfile
from dotenv import load_dotenv
from rag_engine import EbookRAG  # Make sure this version includes hash-based deduplication

import glob
load_dotenv()

# -----------------------
# Page config & CSS
# -----------------------
st.set_page_config(
    page_title="📚 Ebook RAG Chat",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: bold; color: #1f77b4; margin-bottom: 1rem;}
    .sub-header {font-size: 1.2rem; color: #666; margin-bottom: 2rem;}
    .chat-message {padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;}
    .user-message {background-color: #e3f2fd; border-left: 4px solid #2196f3;}
    .bot-message {background-color: #f3e5f5; border-left: 4px solid #9c27b0;}
    .source-badge {background-color: #4caf50; color: white; padding: 0.2rem 0.5rem; border-radius: 0.3rem; font-size: 0.8rem; margin-right: 0.3rem;}
    .stats-card {background-color: #f5f5f5; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;}
</style>
""", unsafe_allow_html=True)

# -----------------------
# Session state initialization
# -----------------------
if 'rag' not in st.session_state:
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        st.session_state.rag = EbookRAG(groq_key)
    else:
        st.session_state.rag = None

if 'messages' not in st.session_state:
    st.session_state.messages = []

# -----------------------
# Sidebar: Book upload & management
# -----------------------
with st.sidebar:
    st.markdown("## 📖 Book Upload")

    # API Key input if missing
    if not st.session_state.rag:
        groq_key = st.text_input("Enter Groq API Key:", type="password")
        if groq_key:
            st.session_state.rag = EbookRAG(groq_key)
            st.success("✅ API Key set!")

    # File uploader
    uploaded_file = st.file_uploader("Upload PDF Book", type=['pdf'], help="Upload a PDF to start chatting")

    if uploaded_file:
        save_name = st.text_input(
            "Save as (optional):",
            value=uploaded_file.name.replace('.pdf', '')
        )

        if st.button("📤 Process Book", type="primary"):
            with st.spinner("Processing PDF... This may take a minute"):
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                # Process PDF with hash deduplication
                result = st.session_state.rag.process_pdf(tmp_path, save_name)

                # Cleanup temp file
                os.unlink(tmp_path)

                if result['success']:
                    st.success(f"✅ Processed {result['pages']} pages into {result['chunks']} chunks!")
                else:
                    st.error(f"❌ {result['error']}")

    # Load existing saved books
    st.markdown("---")
    st.markdown("## 💾 Saved Books")

    saved_books = [f.replace('faiss_index_', '') for f in glob.glob('faiss_index_*') if os.path.isdir(f)]
    
    if saved_books:
        selected_book = st.selectbox("Load existing book:", saved_books)
        if st.button("📂 Load Book"):
            with st.spinner("Loading..."):
                if st.session_state.rag.load_existing_index(selected_book):
                    st.success(f"✅ Loaded {selected_book}")
                else:
                    st.error("❌ Failed to load")
    else:
        st.info("No saved books found")

    # Stats & clear memory
    st.markdown("---")
    if st.session_state.rag:
        stats = st.session_state.rag.get_stats()
        st.markdown("### 📊 Stats")
        for key, value in stats.items():
            st.markdown(f"**{key}:** {value}")

        if st.button("🗑️ Clear Chat History"):
            st.session_state.rag.clear_memory()
            st.session_state.messages = []
            st.success("Memory cleared!")

# -----------------------
# Main header
# -----------------------
st.markdown('<div class="main-header">📚 Ebook RAG Chat</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Chat with your books using AI. Upload a PDF and start asking questions!</div>', unsafe_allow_html=True)

# Check if a book is loaded
if not st.session_state.rag or not st.session_state.rag.current_book:
    st.info("👈 Please upload a PDF book from the sidebar to get started!")
    st.stop()

# -----------------------
# Display chat messages
# -----------------------
for message in st.session_state.messages:
    if message['role'] == 'user':
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>🧑 You:</strong> {message['content']}
        </div>
        """, unsafe_allow_html=True)
    else:
        sources_html = "".join([f'<span class="source-badge">Page {p}</span>' for p in message.get('pages', [])])
        st.markdown(f"""
        <div class="chat-message bot-message">
            <strong>🤖 Bot:</strong> {message['content']}
            <br><br>
            <strong>Sources:</strong> {sources_html}
        </div>
        """, unsafe_allow_html=True)

# -----------------------
# Chat input
# -----------------------
if prompt := st.chat_input("Ask a question about your book..."):
    # Add user message
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    # Get AI response
    with st.spinner("Thinking..."):
        answer, pages = st.session_state.rag.ask(prompt)

    # Add bot message
    st.session_state.messages.append({'role': 'assistant', 'content': answer, 'pages': pages})

    # Rerun to display new message
    st.rerun()

# -----------------------
# Footer
# -----------------------
st.markdown("---")
st.markdown("Made with ❤️ using LangChain, Groq, and Streamlit")
