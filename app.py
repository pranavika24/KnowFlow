"""
AI Knowledge Base Application
=============================
A complete Streamlit app that allows users to upload PDFs, ask questions,
and get answers based on the document content using free AI models.

Features:
- PDF text extraction
- Text chunking for better retrieval
- Sentence embeddings using HuggingFace
- FAISS vector search
- Question answering based on retrieved content
"""

import streamlit as st
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from pypdf import PdfReader
import os
import traceback

# ============================================================
# SECTION 1: PDF TEXT EXTRACTION
# ============================================================
def extract_text_from_pdf(pdf_file):
    """
    Extract all text from a PDF file.
    
    Args:
        pdf_file: Uploaded Streamlit file object
        
    Returns:
        str: Extracted text from all pages
    """
    pdf_reader = PdfReader(pdf_file)
    text = ""
    
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    
    return text


# ============================================================
# SECTION 2: TEXT CHUNKING
# ============================================================
def chunk_text(text, chunk_size=300, overlap=50):
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]
        chunks.append(chunk)

        start += chunk_size - overlap

        # Prevent too many chunks (memory safety)
        if len(chunks) > 1000:
            break

    return chunks


# ============================================================
# SECTION 3: CREATE EMBEDDINGS
# ============================================================
@st.cache_resource
def load_embedding_model():
    """
    Load a free sentence transformer model for embeddings.
    Uses 'BAAI/bge-base-en-v1.5' - a powerful model for semantic search.
    This model provides better answer quality than all-MiniLM-L6-v2.
    
    Returns:
        SentenceTransformer: The embedding model
    """
    return SentenceTransformer('BAAI/bge-base-en-v1.5')


def create_embeddings(chunks, model):
    """
    Convert text chunks into vector embeddings.
    
    Args:
        chunks: List of text chunks
        model: SentenceTransformer model
        
    Returns:
        numpy array: Embeddings matrix
    """
    embeddings = model.encode(chunks, show_progress_bar=True)
    return np.array(embeddings, dtype='float32')


# ============================================================
# SECTION 4: FAISS VECTOR STORE
# ============================================================
def create_faiss_index(embeddings):
    """
    Create a FAISS index for fast similarity search.
    
    Args:
        embeddings: numpy array of text embeddings
        
    Returns:
        FAISS index: Index for searching
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


def retrieve_relevant_chunks(query, chunks, model, index, top_k=3):
    """
    Find the most relevant text chunks for a query.
    
    Args:
        query: User's question
        chunks: List of text chunks
        model: Embedding model
        index: FAISS index
        top_k: Number of chunks to retrieve
        
    Returns:
        list: Top-k relevant text chunks
    """
    # Create query embedding
    query_embedding = model.encode([query], show_progress_bar=False)
    query_embedding = np.array(query_embedding, dtype='float32')
    
    # Search in FAISS index
    distances, indices = index.search(query_embedding, top_k)
    
    # Return relevant chunks
    relevant_chunks = [chunks[i] for i in indices[0]]
    return relevant_chunks


# ============================================================
# SECTION 4B: PERSISTENCE (Save/Load)
# ============================================================
import pickle
import os

def save_knowledge_base(chunks, index, processed_files, folder="knowledge_base"):
    """
    Save the knowledge base to disk for persistence.
    
    Args:
        chunks: List of text chunks
        index: FAISS index
        processed_files: List of processed file names
        folder: Folder to save to
    """
    os.makedirs(folder, exist_ok=True)
    
    # Save chunks
    with open(f"{folder}/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    
    # Save FAISS index
    faiss.write_index(index, f"{folder}/index.faiss")
    
    # Save processed files list
    with open(f"{folder}/files.pkl", "wb") as f:
        pickle.dump(processed_files, f)
    
    return True


def load_knowledge_base(folder="knowledge_base"):
    """
    Load the knowledge base from disk.
    
    Args:
        folder: Folder to load from
        
    Returns:
        tuple: (chunks, index, processed_files) or None if not found
    """
    try:
        # Load chunks
        with open(f"{folder}/chunks.pkl", "rb") as f:
            chunks = pickle.load(f)
        
        # Load FAISS index
        index = faiss.read_index(f"{folder}/index.faiss")
        
        # Load processed files
        with open(f"{folder}/files.pkl", "rb") as f:
            processed_files = pickle.load(f)
        
        return chunks, index, processed_files
    except FileNotFoundError:
        return None


# ============================================================
# SECTION 5: QUESTION ANSWERING
# ============================================================
@st.cache_resource
def load_llm():
    """
    Load a lightweight text generation model for QA.
    Uses distilgpt2 for stable on-device generation.
    
    Returns:
        pipeline: Text generation pipeline
    """
    return pipeline(
        "text-generation",
        model="distilgpt2",
        max_new_tokens=100
    )


def generate_answer(question, context_chunks, llm):
    """
    Generate an answer based on the retrieved context.
    
    Args:
        question: User's question
        context_chunks: Retrieved relevant text
        llm: Text generation pipeline
        
    Returns:
        str: Generated answer
    """
    # Combine top chunks into context
    context = " ".join(context_chunks[:3])

    prompt = f"""
Answer the question based on the context below:

Context:
{context}

Question:
{question}

Answer:
"""

    result = llm(prompt)
    answer = result[0]["generated_text"]
    answer = answer.split("Answer:")[-1].strip()

    return answer


# ============================================================
# SECTION 6: STREAMLIT UI
# ============================================================
def main():
    """
    Main Streamlit application UI.
    """
    # Page configuration
    st.set_page_config(
        page_title="AI Knowledge Base",
        page_icon="📚",
        layout="wide"
    )
    
    # Title and description
    st.title("📚 AI Knowledge Base")
    st.markdown("""
    Upload a PDF document and ask questions about its content.
    This app uses **free** AI models to understand your document and answer your questions.
    """)
    
    st.divider()
    
    # Initialize session state
    if 'chunks' not in st.session_state:
        st.session_state.chunks = None
    if 'index' not in st.session_state:
        st.session_state.index = None
    if 'embedding_model' not in st.session_state:
        st.session_state.embedding_model = None
    if 'llm' not in st.session_state:
        st.session_state.llm = None
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = []  # Store list of processed file names
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []  # Store (question, answer, sources) tuples
    
    # ============================================================
    # SIDEBAR: PDF UPLOAD
    # ============================================================
    with st.sidebar:
        st.header("📄 Upload PDFs")
        st.markdown("Upload one or more PDF documents to create your knowledge base.")

        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True
        )
        
        # Check for saved knowledge base
        saved_kb = load_knowledge_base()
        if saved_kb is not None and st.session_state.chunks is None:
            st.divider()
            if st.button("📂 Load Previous Knowledge Base"):
                with st.spinner("Loading saved knowledge base..."):
                    chunks, index, processed_files = saved_kb
                    st.session_state.chunks = chunks
                    st.session_state.index = index
                    st.session_state.processed_files = processed_files
                    
                    # Load models
                    embedding_model = load_embedding_model()
                    st.session_state.embedding_model = embedding_model
                    llm = load_llm()
                    st.session_state.llm = llm
                    
                    st.success(f"✅ Loaded {len(processed_files)} documents with {len(chunks)} chunks!")
                    st.rerun()
        
        # Process button
        if uploaded_files is not None and len(uploaded_files) > 0:
            if st.button("Process PDFs", type="primary"):
                with st.spinner("Processing PDFs... This may take a minute."):
                    try:
                        # Step 1: Extract text from all PDFs
                        st.info("📄 Extracting text from PDFs...")
                        all_text = ""
                        processed_files = []
                        
                        for uploaded_file in uploaded_files:
                            text = extract_text_from_pdf(uploaded_file)
                            if text.strip():
                                all_text += f"\n\n=== Document: {uploaded_file.name} ===\n\n"
                                all_text += text
                                processed_files.append(uploaded_file.name)
                        
                        if not all_text.strip():
                            st.error("Could not extract text from any PDF. Files might be image-based.")
                            st.stop()
                        
                        # Limit text size for memory safety
                        all_text = all_text[:50000]
                        
                        st.success(f"Extracted text from {len(processed_files)} files ({len(all_text)} characters)")
                        
                        # Step 2: Chunk text
                        st.info("✂️ Splitting text into chunks...")
                        chunks = chunk_text(all_text)
                        st.session_state.chunks = chunks
                        st.session_state.processed_files = processed_files  # Store file names
                        st.success(f"Created {len(chunks)} chunks")
                        
                        # Step 3: Load embedding model
                        st.info("🔄 Loading embedding model...")
                        embedding_model = load_embedding_model()
                        st.session_state.embedding_model = embedding_model
                        
                        # Step 4: Create embeddings
                        st.info("📊 Creating embeddings...")
                        embeddings = create_embeddings(chunks, embedding_model)
                        
                        # Step 5: Create FAISS index
                        st.info("🔍 Building search index...")
                        index = create_faiss_index(embeddings)
                        st.session_state.index = index
                        
                        # Step 5B: Save to disk
                        st.info("💾 Saving knowledge base...")
                        save_knowledge_base(chunks, index, processed_files)
                        
                        # Step 6: Load LLM
                        st.info("🤖 Loading LLM...")
                        llm = load_llm()
                        st.session_state.llm = llm
                        
                        st.success("✅ PDFs processed successfully!")
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        st.text(traceback.format_exc())
        
        # Show processing status
        if st.session_state.processed_files:
            st.divider()
            st.success(f"✅ Loaded {len(st.session_state.processed_files)} files:")
            for f in st.session_state.processed_files:
                st.text(f"• {f}")
    
    # ============================================================
    # MAIN AREA: QUESTION ANSWERING
    # ============================================================
    
    # Check if PDF is processed
    if st.session_state.chunks is None:
        st.info("👈 Please upload and process a PDF in the sidebar first!")
        
        # Show example instructions
        st.markdown("""
        ### How to use:
        1. **Upload a PDF** using the sidebar
        2. Click **Process PDF** to analyze the document
        3. **Ask questions** about the document content
        4. Get AI-powered answers based on the PDF!
        
        ### What it does:
        - Extracts text from your PDF
        - Creates searchable chunks
        - Uses AI to find relevant information
        - Answers your questions accurately
        """)
    else:
        # Show document info
        file_count = len(st.session_state.processed_files)
        st.info(f"📄 {file_count} document(s) loaded with {len(st.session_state.chunks)} chunks")
        
        # ============================================================
        # DISPLAY CHAT HISTORY
        # ============================================================
        if st.session_state.chat_history:
            st.subheader("💬 Chat History")
            for i, chat in enumerate(st.session_state.chat_history):
                with st.expander(f"Q{len(st.session_state.chat_history)}: {chat['question'][:50]}..."):
                    st.markdown(f"**You:** {chat['question']}")
                    st.markdown(f"**AI:** {chat['answer']}")
            st.divider()
        
        # Question input at bottom
        with st.form(key="chat_form", clear_on_submit=True):
            question = st.text_input("Type your question...", key="chat_input")
            submit = st.form_submit_button("Send", type="primary")
        
        if submit:
            if not question.strip():
                st.warning("Please enter a question!")
            else:
                llm = st.session_state.llm
                if llm is None:
                    st.error("QA model failed to load.")
                    return
                with st.spinner("Finding relevant information..."):
                    try:
                        relevant_chunks = retrieve_relevant_chunks(
                            question,
                            st.session_state.chunks,
                            st.session_state.embedding_model,
                            st.session_state.index,
                            top_k=3
                        )
                        with st.spinner("Generating answer..."):
                            answer = generate_answer(
                                question,
                                relevant_chunks,
                                llm
                            )
                        st.session_state.chat_history.append({
                            'question': question,
                            'answer': answer,
                            'sources': relevant_chunks
                        })
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error generating answer: {str(e)}")
        st.divider()


# Run the app
if __name__ == "__main__":
    main()