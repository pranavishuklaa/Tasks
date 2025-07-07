from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

# Embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def split_text(text):
    print("[INFO] Splitting text into chunks...")  # Notification
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    print(f"[INFO] Total Chunks Created: {len(chunks)}")  # See how many chunks
    return [Document(page_content=chunk) for chunk in chunks]


def create_vectorstore(docs):
    print("[INFO] Creating vectorstore using FAISS and HuggingFace embeddings...")
    print(f"[INFO] Embedding Model Used: all-MiniLM-L6-v2")  # ✅ Model name
    vectorstore = FAISS.from_documents(docs, embedding_model)
    print("[INFO] Vectorstore created successfully.")
    return vectorstore


def create_retriever_chain(vectorstore):
    print("[INFO] Loading FLAN-T5 model for generation...")
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        tokenizer="google/flan-t5-base",
        max_length=256
    )
    print("[INFO] Model Loaded: google/flan-t5-base")  # ✅ LLM name
    llm = HuggingFacePipeline(pipeline=pipe)
    print("[INFO] Building RetrievalQA Chain...")
    retriever_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
    print("[INFO] RAG pipeline ready! 🚀")
    return retriever_chain

