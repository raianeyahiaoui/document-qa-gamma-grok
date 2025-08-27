from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

def build_qa_engine(chunks):
    """Builds a Q&A pipeline using Groq + Gemini (Gamma Models)."""
    
    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    
    # LLM
    llm = ChatGroq(model="gemini-pro")  # You can adjust with GROQ API
    
    # Prompt
    template = """
    You are an assistant that answers questions based on documents.
    Context: {context}
    Question: {question}
    Answer:"""
    prompt = PromptTemplate(
        input_variables=["context", "question"], 
        template=template
    )
    
    # QA Chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": prompt}
    )
    
    return qa
