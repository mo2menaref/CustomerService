import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def setup_rag():
    """Setup RAG engine with PDF documents and return a retriever."""
    # Check if the data directory exists
    data_dir = "data"
    if not os.path.exists(data_dir):
        print(f"Warning: '{data_dir}' directory not found. Please create it and add PDF files.")
        return None
    
    try:
        # Load all PDF documents from the data directory
        loader = DirectoryLoader(
            data_dir,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )
        docs = loader.load()
        
        if not docs:
            print("Warning: No PDF files found in the 'data' folder.")
            return None
        
        print(f"✓ Loaded {len(docs)} documents from PDFs")
        
        # Split the documents into smaller manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = text_splitter.split_documents(docs)
        
        print(f"✓ Split documents into {len(splits)} chunks")
        
        # Create embeddings and store them in ChromaDB
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        
        print("✓ RAG engine initialized successfully")
        
        # Return the retriever to be used by the agent
        return vectorstore.as_retriever()
    
    except Exception as e:
        print(f"Error setting up RAG engine: {e}")
        return None
