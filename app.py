import streamlit as st
import tempfile
import uuid
from pathlib import Path
import os
from utils.config import PINECONE_API_KEY, GROQ_API_KEY
from rag.ingestion import extract_multimodal_documents
from rag.indexing import index_documents
from rag.pipeline import answer_question
from rag.evaluation import render_evaluation_dashboard
from PIL import Image

st.set_page_config(
    page_title="Multimodal RAG Assistant",
    page_icon="📄",
    layout="wide"
)

app_mode = st.sidebar.radio("App Mode", ["Chat Assistant", "RAG Evaluation Dashboard"])
st.sidebar.divider()

if app_mode == "RAG Evaluation Dashboard":
    render_evaluation_dashboard()
else:
    st.title("Multimodal RAG Assistant")
    st.subheader("Ask questions about text, tables, graphs and images inside your document.")

    # Ensure keys are loaded
    if not GROQ_API_KEY or not PINECONE_API_KEY:
        st.error("API keys missing! Please configure .env file.")
        st.stop()

# Initialize session state
if "document_id" not in st.session_state:
    st.session_state.document_id = None
if "namespace" not in st.session_state:
    st.session_state.namespace = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "doc_stats" not in st.session_state:
    st.session_state.doc_stats = None

with st.sidebar:
    st.header("📄 Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    
    if st.button("Process Document") and uploaded_file is not None:
        with st.spinner("Processing document (extracting text, tables, and images)..."):
            # 1. Reset state for new document
            new_doc_id = str(uuid.uuid4())
            st.session_state.document_id = new_doc_id
            st.session_state.namespace = f"doc-{new_doc_id}"
            st.session_state.chat_history = []
            
            # 2. Save PDF temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_pdf_path = Path(tmp_file.name)
            
            # 3. Process and Ingest
            try:
                documents, stats = extract_multimodal_documents(tmp_pdf_path, new_doc_id)
                st.session_state.doc_stats = stats
                
                # 4. Index in Pinecone
                with st.spinner("Indexing documents in Pinecone..."):
                    index_documents(documents, st.session_state.namespace)
                    
                st.success("✅ Document processed successfully!")
            except Exception as e:
                st.error(f"Error processing document: {e}")
            finally:
                os.unlink(tmp_pdf_path)

if st.session_state.doc_stats:
    stats = st.session_state.doc_stats
    cols = st.columns(4)
    cols[0].metric("Text chunks", stats["text_chunks"])
    cols[1].metric("Tables", stats["tables"])
    cols[2].metric("Visuals", stats["visuals"])
    total = stats["text_chunks"] + stats["tables"] + stats["visuals"]
    cols[3].metric("Total indexed items", total)

st.divider()

if st.session_state.document_id:
    st.header("💬 Ask a question")
    
    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            if msg["role"] == "assistant" and "sources" in msg:
                with st.expander("📚 Retrieved Sources"):
                    for src in msg["sources"]:
                        st.markdown(f"**Page {src['page']} | Type: {src['modality'].capitalize()}**")
                        if src['modality'] == 'visual' and src['image_path']:
                            try:
                                img = Image.open(src['image_path'])
                                st.image(img, caption=f"Page {src['page']} Visual")
                            except Exception:
                                st.write("(Image file not found)")
                        else:
                            st.text(src['content'][:500] + ("..." if len(src['content']) > 500 else ""))
                        st.divider()
                if "metrics" in msg and msg["metrics"]:
                    m = msg["metrics"]
                    st.caption(f"⏱️ Retrieval: {m['retrieval_latency_ms']:.0f}ms | Context: {m['context_building_latency_ms']:.0f}ms | LLM: {m['llm_latency_ms']:.0f}ms | Total: {m['total_latency_ms']:.0f}ms | Prompt Tokens: {m['estimated_final_prompt_tokens']}")
    
    # Chat Input
    if question := st.chat_input("Type your question here..."):
        # Add user question to history
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
            
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = answer_question(question, st.session_state.namespace)
                    answer = result["answer"]
                    sources = result["sources"]
                    
                    st.markdown(answer)
                    
                    with st.expander("📚 Retrieved Sources"):
                        for src in sources:
                            st.markdown(f"**Page {src['page']} | Type: {src['modality'].capitalize()}**")
                            if src['modality'] == 'visual' and src['image_path']:
                                try:
                                    img = Image.open(src['image_path'])
                                    st.image(img, caption=f"Page {src['page']} Visual")
                                except Exception:
                                    st.write("(Image file not found)")
                            else:
                                st.text(src['content'][:500] + ("..." if len(src['content']) > 500 else ""))
                            st.divider()
                    
                    if result.get("metrics"):
                        m = result["metrics"]
                        st.caption(f"⏱️ Retrieval: {m['retrieval_latency_ms']:.0f}ms | Context: {m['context_building_latency_ms']:.0f}ms | LLM: {m['llm_latency_ms']:.0f}ms | Total: {m['total_latency_ms']:.0f}ms | Prompt Tokens: {m['estimated_final_prompt_tokens']}")
                            
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": answer,
                        "sources": sources,
                        "metrics": result.get("metrics")
                    })
                except Exception as e:
                    st.error(f"Error answering question: {e}")
else:
    st.info("Please upload and process a document from the sidebar to start asking questions.")
