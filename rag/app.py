import streamlit as st
from utils.file_loader import extract_text
from utils.rag_pipeline import split_text, create_vectorstore, create_retriever_chain

st.set_page_config(page_title="🧠 RAG Chatbot (Hugging Face)", layout="wide")
st.title("🗃️ Chat With Your Documents — With the RAG provided solution!")

uploaded_files = st.file_uploader("📄 Upload PDF, DOCX, or TXT files", type=['pdf', 'docx', 'txt'], accept_multiple_files=True)

if uploaded_files:
    full_text = ""
    for file in uploaded_files:
        full_text += extract_text(file) + "\n"

    with st.spinner("📚 Processing documents..."):
        docs = split_text(full_text)
        vectorstore = create_vectorstore(docs)
        qa_chain = create_retriever_chain(vectorstore)

    st.success("✅ Documents processed! Ask me anything 👇")

    user_input = st.text_input("❓ Your Question:")
    if user_input:
        retrieved_docs = vectorstore.similarity_search(user_input, k=3)
        # st.write("**Retrieved chunks:**")
        # for i, doc in enumerate(retrieved_docs):
        #     st.write(f"**Chunk {i+1}:** {doc.page_content[:300]}...")
        #     st.write(f"**Score:** {doc.metadata if hasattr(doc, 'metadata') else 'N/A'}")
        #     # st.write("---")
        result = qa_chain.run(user_input)
        st.markdown(f"**🤖 Answer:** {result}")
        
