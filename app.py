import streamlit as st
from pathlib import Path
from PIL import Image
from src.graph import build_graph
from src.config import PDFS_DIR, PAGE_IMAGES_DIR
from src.ingest import ingest_pdf
from src.vector_store import ensure_collection, close_client

st.set_page_config(page_title="VisionRAG", layout="wide")
st.title("VisionRAG: Multimodal PDF Chatbot")

# Ensure directories exist
PDFS_DIR.mkdir(exist_ok=True)

# Sidebar: Upload PDF
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded_file:
        pdf_path = PDFS_DIR / uploaded_file.name
        # Only ingest if it's a new file
        if not pdf_path.exists():
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            with st.spinner("Processing PDF with ColQwen2 (runs on CPU/GPU)..."):
                ensure_collection(reset=True)
                ingest_pdf(pdf_path, 0)
                close_client()
            st.success("Indexing complete!")
        else:
            st.info("Document already indexed.")

# Main area: Chat
question = st.text_input("Ask a question about your document:")
if question:
    with st.spinner("Searching and generating answer..."):
        graph = build_graph()
        try:
            result = graph.invoke({"question": question})
        finally:
            close_client()
        
    st.subheader("Answer:")
    parts = result["answer"].split("\n\nCitations (with bounding boxes):")
    st.markdown(parts[0])
    
    # Display highlighted page if present
    if len(parts) > 1:
        st.subheader("Citations:")
        for line in parts[1].strip().split("\n"):
            if "Highlighted file:" in line:
                filename = line.split("Highlighted file:")[-1].strip()
                img_path = PAGE_IMAGES_DIR / filename
                if img_path.exists():
                    st.image(Image.open(img_path), caption=line.replace("- ", ""), use_container_width=True)
