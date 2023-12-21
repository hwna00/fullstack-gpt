import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.storage import LocalFileStore
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda

st.set_page_config(
    page_title="Document GPT",
    page_icon="📜",
)


@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"

    with open(file_path, "wb") as f:
        f.write(file_content)

    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )

    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()

    cache_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cache_embeddings)
    retriver = vectorstore.as_retriever()

    return retriver


def send_message(msg, role, save=True):
    with st.chat_message(role):
        st.markdown(msg)
    if save:
        st.session_state["messages"].append({"msg": msg, "role": role})


def paint_history():
    for msg in st.session_state["messages"]:
        send_message(msg["msg"], msg["role"], False)


st.title("Document GPT")

st.markdown(
    """
### Welcome!
            
Use this chatbot to ask questions to an AI about your files!

Upload your file on the sidebar.
"""
)

with st.sidebar:
    file = st.file_uploader(
        "Upload any file do you want!",
        type=["pdf", "txt", "docx"],
    )

if file:
    retriever = embed_file(file)
    send_message("I'm ready! Ask away!", "ai", False)
    paint_history()

    message = st.chat_input("Ask anything about your file")

    if message:
        send_message(message, "human")
else:
    st.session_state["messages"] = []
