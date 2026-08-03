import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import ConversationalRetrievalChain

st.set_page_config(page_title="Cloud AI Summariser", layout="wide")
st.title("📄 Cloud AI Document Summariser & QA Bot")

# Securely read token from Streamlit Secrets
hf_token = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("Upload Zone")
    uploaded_file = st.file_uploader("Upload your PDF document:", type=["pdf"])

if uploaded_file:
    try:
        with st.spinner("Processing your document in the cloud..."):
            reader = PdfReader(uploaded_file)
            raw_text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
            
            if not raw_text.strip():
                st.error("This PDF seems to be empty or scanned.")
            else:
                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
                chunks = splitter.split_text(raw_text)
                
                # Cloud Embeddings
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                db = Chroma.from_texts(chunks, embedding=embeddings)
                
                # Cloud LLM Model
                llm = HuggingFaceEndpoint(
                    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
                    huggingfacehub_api_token=hf_token,
                    temperature=0.3
                )
                
                qa = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=db.as_retriever(search_kwargs={"k": 2}),
                    return_source_documents=True
                )
                
                st.success("Ready! Ask your questions below.")
                
                query = st.text_input("Ask something about the document:")
                if query:
                    with st.spinner("AI is analyzing..."):
                        res = qa({"question": query, "chat_history": st.session_state.history})
                        st.session_state.history.append((query, res["answer"]))
                    st.markdown(f"**Answer:** {res['answer']}")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
