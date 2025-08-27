import streamlit as st
from utils.document_loader import load_document
from utils.text_processing import chunk_text
from utils.qa_engine import build_qa_engine

st.set_page_config(page_title="Document Q&A with GROQ + Gemini", layout="wide")

st.title("📄 Document Q&A With GROQ API and Gemini")

uploaded_file = st.file_uploader("Upload a document (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

if uploaded_file:
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.read())
    
    st.success("✅ File uploaded successfully!")
    
    # Load and chunk text
    text = load_document(uploaded_file.name)
    chunks = chunk_text(text)
    
    # Build QA engine
    qa = build_qa_engine(chunks)
    
    st.subheader("Ask your document anything 👇")
    query = st.text_input("Enter your question")
    
    if query:
        response = qa.run(query)
        st.write("### Answer:")
        st.success(response)
