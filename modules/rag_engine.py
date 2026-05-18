import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def setup_rag():
    # Check if the file exists to avoid errors during initial testing
    pdf_path = "data/course_policies.pdf"
    if not os.path.exists(pdf_path):
        print("Warning: PDF file not found. Please add 'course_policies.pdf' to the 'data' folder.")
        return None

    loader = DirectoryLoader(
                        "data/",
                        glob="**/*.pdf",
                        loader_cls=PyPDFLoader
                        )
    docs = loader.load()
    
    
    # Split the document into smaller manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)
    
    # Create embeddings and store them in ChromaDB
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    
    # Return the retriever to be used by the agent
    return vectorstore.as_retriever()
