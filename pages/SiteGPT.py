import streamlit as st
from langchain.document_loaders import SitemapLoader

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


@st.cache_data(show_spinner="Loading website...")
def load_website(url):
    loader = SitemapLoader(url)
    loader.requests_per_second = 1
    docs = loader.load()

    return docs


with st.sidebar:
    url = st.text_input(
        "Write down a URL",
        placeholder="https://example.com",
    )

if url:
    if ".xml" not in url:
        with st.sidebar:
            st.error("Please write down a Sitemap URL.")
    else:
        load_website(url)
