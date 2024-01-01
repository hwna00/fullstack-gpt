import streamlit as st
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda


from pages.SiteGPT.chat_handler import get_answers, choose_answer, memory
from pages.SiteGPT.data_loader import load_website
from pages.SiteGPT.utils import paint_history, send_message, save_message


st.set_page_config(page_title="Quiz GPT", page_icon="❓")
st.title("Quiz GPT")
st.markdown(
    """
    Ask questions about the content of a website.
    Start by writing the URl of the website on the sidebar.
"""
)


def load_memory(_):
    return memory.load_memory_variables({})["chat_history"]


def find_history(query):
    histories = load_memory("")
    temp = []
    for idx in range(len(histories) // 2):
        temp.append(
            {
                "input": histories[idx * 2].content,
                "output": histories[idx * 2 + 1].content,
            }
        )
    print(temp)
    for item in temp:
        if item["input"] == query:
            return item["output"]
    return None


with st.sidebar:
    url = st.text_input("Write down a URL", placeholder="https://example.com")

if url:
    if ".xml" not in url:
        with st.sidebar:
            st.error("Please write down a Sitemap URL.")
    else:
        retriever = load_website(url)
        send_message("I'm ready! Ask away!", "ai", False)
        paint_history()

        query = st.chat_input("Ask a question to the website.")
        if query:
            send_message(query, "human")

            found = find_history(query)
            if found:
                send_message(found, "ai")
            else:
                chain = (
                    {
                        "docs": retriever,
                        "question": RunnablePassthrough(),
                    }
                    | RunnablePassthrough.assign(chat_history=load_memory)
                    | RunnableLambda(get_answers)
                    | RunnableLambda(choose_answer)
                )

                with st.chat_message("ai"):
                    result = chain.invoke(query)
                memory.save_context(
                    {"input": query},
                    {"output": result.content},
                )
else:
    st.session_state["messages"] = []
