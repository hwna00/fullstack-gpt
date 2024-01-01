import streamlit as st
from langchain.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document

from .chat_handler import memory


def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")

    if header:
        header.decompose()
    if footer:
        footer.decompose()

    return (
        str(soup.get_text())
        .replace("\n", " ")
        .replace("\xa0", " ")
        .replace("CloseSearch Submit Blog", "")
    )


def find_history(query):
    histories = memory.load_memory_variables({})["chat_history"]
    temp = [
        {
            "input": histories[idx * 2].content,
            "output": histories[idx * 2 + 1].content,
        }
        for idx in range(len(histories) // 2)
    ]

    docs = [
        Document(page_content=f"input:{item['input']}\noutput:{item['output']}")
        for item in temp
    ]
    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
    found_docs = vector_store.similarity_search(query)

    if not found_docs:
        return None

    candidate = found_docs[0].page_content.split("\n")[1]
    return candidate.replace("output:", "")


@st.cache_data(show_spinner="Loading website...")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )

    loader = SitemapLoader(
        url,
        parsing_function=parse_page,
    )
    loader.requests_per_second = 1
    docs = loader.load_and_split(splitter)
    vector_store = FAISS.from_documents(
        docs,
        OpenAIEmbeddings(),
    )

    return vector_store.as_retriever()
