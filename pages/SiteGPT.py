import streamlit as st
from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import Html2TextTransformer

st.set_page_config(
    page_title="Quiz GPT",
    page_icon="❓",
)

st.title("Quiz GPT")

st.markdown(
    """
    Ask questions about the content of a website.
    Start by writing the URl of the website on the sidebar.
    """
)

html2text_transformer = Html2TextTransformer()

with st.sidebar:
    url = st.text_input(
        "Write down a URL",
        placeholder="https://example.com",
    )

if url:
    loader = AsyncChromiumLoader([url])
    docs = loader.load()

    # Use transformer, transform html to text
    transformed = html2text_transformer.transform_documents(docs)
