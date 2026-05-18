import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def setup_rag():
    """Setup RAG engine with PDF and TXT documents and return a retriever."""
    # Check if the data directory exists
    data_dir = "data"
    if not os.path.exists(data_dir):
        print(f"Warning: '{data_dir}' directory not found. Please create it and add PDF or TXT files.")
        return None
    
    try:
        # Load all PDF documents from the data directory
        pdf_loader = DirectoryLoader(
            data_dir,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )
        pdf_docs = pdf_loader.load()
        
        # Load all TXT documents from the data directory
        txt_loader = DirectoryLoader(
            data_dir,
            glob="**/*.txt",
            loader_cls=TextLoader
        )
        txt_docs = txt_loader.load()
        
        # Combine both document types
        docs = pdf_docs + txt_docs
        
        if not docs:
            print("Warning: No PDF or TXT files found in the 'data' folder.")
            return None
        
        pdf_count = len(pdf_docs)
        txt_count = len(txt_docs)
        print(f"✓ Loaded {pdf_count} PDF documents and {txt_count} TXT documents")
        
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
