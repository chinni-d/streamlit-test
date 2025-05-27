import os
import streamlit as st
import tempfile
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Set OpenRouter credentials (for LLM only)
os.environ["OPENAI_API_KEY"] = "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"  # 🔑 Replace with your actual key
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

# --- UI Enhancements ---
st.set_page_config(page_title="📄 Document QA Chatbot", layout="wide", page_icon="📄")

# Custom CSS for a modern and clean look
st.markdown(
    """
    <style>
    /* General Styles */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', Roboto, Arial, sans-serif;
        color: #000000 !important; /* FORCE BLACK TEXT GLOBALLY */
    }
    .main {
        background-color: #f8f9fa !important; /* Light gray background */
        padding: 2rem;
        color: #000000 !important; /* Ensure text in main is black */
    }
    .stApp {
        background-color: #f8f9fa !important; /* Consistent light gray background */
        color: #000000 !important; /* Ensure text in stApp is black */
    }

    /* Ensure all markdown text is black */
    .stMarkdown, .stMarkdown p, .stMarkdown li {
        color: #000000 !important;
    }

    /* Title */
    h1 {
        color: #2c3e50; /* Dark blue-gray */
        text-align: center;
        padding-bottom: 1rem;
    }

    /* Sidebar */
    .stSidebar {
        background-color: #ffffff !important; /* FORCE WHITE sidebar background */
        padding: 1rem;
        border-right: 1px solid #e0e0e0; /* Subtle border */
    }
    /* Style the sidebar collapse/expand arrow - trying more general and specific selectors */
    /* Attempt 1: Targeting common Streamlit button patterns with SVGs */
    .stSidebar button[kind="icon"] svg {
        fill: #000000 !important;
    }
    /* Attempt 2: Using data-testid which are sometimes stable */
    button[data-testid="stSidebarNavCollapseButton"] svg,
    button[data-testid="stSidebarCollapseButton"] svg {
        fill: #000000 !important;
    }
    /* Attempt 3: A more direct path if the structure is consistent */
    .stSidebar > div:first-child > div:first-child > button svg {
        fill: #000000 !important;
    }
    /* Attempt 4: Broadest attempt for any SVG in a button in the sidebar header area */
    div[data-testid="stSidebarHeader"] button svg {
        fill: #000000 !important;
    }
    /* Attempt 5: Broadest for any SVG in any button directly in stSidebar first div */
    .stSidebar > div > button svg {
         fill: #000000 !important;
    }

    .stSidebar .stMarkdown h2 {
        color: #2c3e50 !important; /* Dark blue-gray for sidebar titles */
        font-size: 1.5rem;
    }
    .stSidebar .stMarkdown p, .stSidebar .stInfo, .stSidebar .stInfo > div {
        color: #000000 !important; /* FORCE BLACK text in sidebar paragraphs and info boxes */
        font-size: 0.95rem;
    }
    .stSidebar .stImage>img {
        border-radius: 8px;
        margin-bottom: 1rem;
    }

    /* File Uploader and Text Input */
    .stFileUploader > div > div {
        border: 1.5px dashed #3498db; /* Blue dashed border - can be changed if desired */
        border-radius: 8px;
        background-color: #d2e6f9 ; /* New background color D2E6F9 */
    }
    .stFileUploader label { /* Target the label of the file uploader */
        color: #000000 !important; /* Ensure label text is black */
    }

    .stTextInput > div > input {
        border: 1.5px solid #bdc3c7; /* Light gray border - can be changed if desired */
        border-radius: 8px;
        padding: 0.75rem 1rem;
        background-color: #d2e6f9 !important; /* New background color D2E6F9 */
        color: #000000 !important; /* Ensure input text is black */
    }
    .stTextInput > div > input:focus {
        border-color: #3498db; /* Blue border on focus */
        box-shadow: 0 0 0 0.2rem rgba(52, 152, 219, 0.25);
    }

    /* Buttons (if any, for future use) */
    .stButton>button {
        color: white;
        background-color: #3498db; /* Primary blue */
        border-radius: 8px;
        font-weight: 600;
        padding: 0.6em 1.5em;
        border: none;
        transition: background-color 0.2s;
    }
    .stButton>button:hover {
        background-color: #2980b9; /* Darker blue on hover */
    }

    /* Expander for Source Documents */
    .stExpander {
        border: 1px solid #e0e0e0;
        border-radius: 8px !important;
        background-color: #ffffff !important; /* White background for expander */
        margin-top: 1.5rem;
    }
    .stExpanderHeader {
        font-weight: 600;
        color: #2c3e50; /* Dark blue-gray for expander headers - this is fine */
        font-size: 1.1rem;
    }
    .stExpander code { /* Style for code within expander if any */
        background-color: #f0f0f0;
        padding: 0.2em 0.4em;
        border-radius: 3px;
    }

    /* Answer and Messages */
    .stAlert, .stSuccess, .stInfo, .stError { /* General alert styling */
        border-radius: 8px !important;
        font-size: 1rem;
        padding: 1rem;
    }
    /* Custom styling for the answer block */
    .answer-block {
        background-color: #eafaf1; /* Light green background */
        padding:1em 1.2em;
        border-radius:8px;
        border-left:5px solid #2ecc71; /* Green left border */
        margin-top: 1rem;
        color: #000000 !important; /* Black text for answer block */
    }
    .answer-block b {
        color: #1e8449; /* Darker green for "Answer:" label, stands out from black text */
    }
    /* Styling for source document items */
    .source-doc-item {
        font-size:0.9em;
        color:#000000 !important; /* Black text for source documents */
        max-height:150px;
        overflow-y:auto;
        padding:8px;
        border:1px solid #dddddd;
        border-radius:5px;
        background-color: #f0f0f0 !important; /* Distinct light gray background */
        margin-bottom: 8px;
    }

    /* Ensure Streamlit's native alerts have dark text if their background is light */
    .stAlert > div { /* Target the inner div where text often resides */
        color: #000000 !important;
    }
    /* If specific alert types still have issues, they can be targeted: */
    .stSuccess > div, .stInfo > div, .stWarning > div, .stError > div {
        color: #000000 !important; /* Assuming their default backgrounds are light */
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar for branding and instructions
with st.sidebar:
    st.markdown("## 📄 Document QA Chatbot")
    st.markdown(
        """
        **How to use:**
        1. **Upload** your PDF or DOCX file.
        2. **Ask** a question related to the document.
        3. Get an **instant answer** with cited sources!
        """
    )
    st.info("Powered by Langchain, OpenRouter, and HuggingFace Embeddings.")

# Main Page Layout
st.title("📄 Document QA Chatbot ")

# Using columns for a more organized layout
col1, col2 = st.columns([2, 3]) # Adjust ratio as needed

with col1:
    uploaded_file = st.file_uploader(
        "Upload your PDF or DOCX document",
        type=["pdf", "docx"],
        help="Drag and drop or click to upload your document."
    )

with col2:
    query = st.text_input(
        "Ask a question about the content of your document:",
        placeholder="E.g., What are the main findings?",
        help="Type your question here and press Enter."
    )

if uploaded_file and query:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Load the document
    if uploaded_file.name.endswith(".pdf"):
        loader = PyPDFLoader(tmp_path)
    else:
        loader = Docx2txtLoader(tmp_path)

    documents = loader.load()

    # Use HuggingFace local embeddings instead of OpenAI
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(documents, embeddings)

    # Create retriever and QA chain with OpenRouter model
    retriever = vectordb.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(
            temperature=0,
            model_name="openai/gpt-3.5-turbo", # or try another model from OpenRouter
        ),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )

    with st.spinner("🧠 Thinking & Analyzing Document..."):
        try:
            result = qa_chain(query)

            # Display answer in a styled block
            st.markdown(
                f"<div class='answer-block'><b>Answer:</b><br>{result['result']}</div>",
                unsafe_allow_html=True
            )

            # Show source documents
            with st.expander("🔍 View Source Documents"):
                for i, doc in enumerate(result["source_documents"], 1):
                    st.markdown(f"**Source {i}:**")
                    # Apply the new class for better styling and visibility
                    st.markdown(f"<div class='source-doc-item'>{doc.page_content}</div>", unsafe_allow_html=True)
                    if i < len(result["source_documents"]): # Add separator if not the last document
                        st.markdown("---")
        except Exception as e:
            st.error(f"❌ Oops! An error occurred: {e}")
        finally:
            # Clean up temp file
            os.remove(tmp_path)
elif uploaded_file and not query:
    st.info("✨ Great! Now please enter a question about the uploaded document.")
elif query and not uploaded_file:
    st.warning("☝️ Please upload a document first to ask questions about it.")
else:
    st.info("👋 Welcome! Upload a document and ask a question to get started.")