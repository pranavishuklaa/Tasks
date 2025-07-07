from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
import torch

# Check if CUDA is available
device = 0 if torch.cuda.is_available() else -1
print(f"[INFO] Using device: {'GPU' if device == 0 else 'CPU'}")

# Embedding model - Fixed model name consistency
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
)

def split_text(text):
    """Split text into chunks for processing"""
    if not text or not text.strip():
        raise ValueError("Input text is empty or None")
    
    print("[INFO] Splitting text into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512, 
        chunk_overlap=128,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_text(text)
    print(f"[INFO] Total Chunks Created: {len(chunks)}")
    
    if not chunks:
        raise ValueError("No chunks created from input text")
    
    return [Document(page_content=chunk) for chunk in chunks]

def create_vectorstore(docs):
    """Create FAISS vectorstore from documents"""
    if not docs:
        raise ValueError("No documents provided for vectorstore creation")
    
    print("[INFO] Creating vectorstore using FAISS and HuggingFace embeddings...")
    print(f"[INFO] Embedding Model Used: sentence-transformers/all-mpnet-base-v2")
    
    try:
        vectorstore = FAISS.from_documents(docs, embedding_model)
        print("[INFO] Vectorstore created successfully.")
        return vectorstore
    except Exception as e:
        print(f"[ERROR] Failed to create vectorstore: {e}")
        raise

def create_retriever_chain(vectorstore):
    """Create the retrieval QA chain"""
    print("[INFO] Loading FLAN-T5 model for generation...")
    
    try:
        # Create pipeline with proper error handling
        pipe = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            tokenizer="google/flan-t5-base",
            device=device,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            temperature=0.7,
            max_length=128,
            do_sample=True,
            # TRANSFORMERS_VERBOSITY = info
            pad_token_id=0  # Add padding token
        )
        print("[INFO] Model Loaded: google/flan-t5-base")
        
        # Create LLM with proper parameters
        llm = HuggingFacePipeline(
            pipeline=pipe,
            model_kwargs={
                "temperature": 0.7,
                "max_length": 256,
                "do_sample": True
            }
        )
        
        print("[INFO] Building RetrievalQA Chain...")
        
        # Create retriever with proper parameters
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}  # Return top 3 most relevant chunks
        )
        
        # Create RetrievalQA chain
        retriever_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # Specify chain type
            retriever=retriever,
            verbose=True,
            return_source_documents=False  # Return source documents for debugging
        )
        
        print("[INFO] RAG pipeline ready! 🚀")
        return retriever_chain
        
    except Exception as e:
        print(f"[ERROR] Failed to create retriever chain: {e}")
        raise