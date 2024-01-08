import streamlit as st
import subprocess
import math
from pydub import AudioSegment
import glob
import openai
import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import StrOutputParser
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda

llm = ChatOpenAI(
    temperature=0.1,
)

st.set_page_config(page_title="Meeting GPT", page_icon="💼")
st.title("Meeting GPT")
st.markdown(
    """
    Welcome to MeetingGPT, upload a video and I will give you a transcript,
    a summary and a chat bot to ask any question about it.

    Get started by uploading a video file in the side bar.
    """
)

has_transcript = os.path.exists("./.cache/conan.txt")


splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=800,
    chunk_overlap=100,
)


@st.cache_data()
def embed_file(file_path, file_name):
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file_name}")
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=800,
        chunk_overlap=100,
    )

    loader = TextLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()

    cache_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cache_embeddings)
    retriever = vectorstore.as_retriever()

    return retriever


@st.cache_data()
def extract_audio_from_video(video_path):
    if has_transcript:
        return
    audio_path = video_path.replace("mp4", "mp3")
    command = ["ffmpeg", "-y", "-i", video_path, "-vn", audio_path]

    subprocess.run(command)


@st.cache_data()
def cut_audio_in_chunks(audio_path, chunk_size, chunks_folder):
    if has_transcript:
        return
    track = AudioSegment.from_mp3(audio_path)
    chunk_len = chunk_size * 60 * 1000
    chunks = math.ceil(len(track) / chunk_len)

    for i in range(chunks):
        start_time = i * chunk_len
        end_time = (i + 1) * chunk_len
        separated = track[start_time:end_time]
        separated.export(f"{chunks_folder}/chunk_{i}.mp3", format="mp3")


@st.cache_data()
def transcribe_chunks(chunk_folder, destination):
    if has_transcript:
        return

    files = glob.glob(f"{chunk_folder}/*.mp3")
    files.sort()

    for file in files:
        with open(file, "rb") as audio_file, open(destination, "a") as text_file:
            transcript = openai.Audio.transcribe("whisper-1", audio_file)
            text_file.write(transcript["text"])


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


with st.sidebar:
    video = st.file_uploader("Video", type=["mp4", "avi", "mkv", "mov"])

if video:
    chunks_path = "./.cache/chunks"
    with st.status("Loading video...") as status:
        video_content = video.read()
        video_path = f".cache/{video.name}"
        audio_path = video_path.replace("mp4", "mp3")
        transcript_path = video_path.replace("mp4", "txt")
        with open(video_path, "wb") as f:
            f.write(video_content)

    status.update(label="Extracting audio...")
    extract_audio_from_video(video_path)

    status.update(label="Cutting audio segments...")
    cut_audio_in_chunks(audio_path, 6, chunks_path)

    status.update(label="Transcribing audio...")
    transcribe_chunks(chunks_path, transcript_path)

    transcript_tab, summary_tab, qa_tab = st.tabs(["Transcript", "Summary", "Q&A"])

    with transcript_tab:
        with open(transcript_path, "r") as file:
            st.write(file.read())

    with summary_tab:
        start = st.button("Generate summary")

        if start:
            loader = TextLoader(transcript_path)
            docs = loader.load_and_split(text_splitter=splitter)

            first_summary_prompt = ChatPromptTemplate.from_template(
                """
                    Write a concise summary of the following:
                    "{text}"
                    CONCISE SUMMARY:
                    """
            )

            first_summary_chain = first_summary_prompt | llm | StrOutputParser()
            summary = first_summary_chain.invoke({"text": docs[0].page_content})

            refine_prompt = ChatPromptTemplate.from_template(
                """
                    Your job is to provide a final summary.
                    We have provided an existing summary up to a certain point: {existing_summary}
                    We have the opportunity to refine the existing summary (only if needed) with some more context below.
                    ----------
                    {context}
                    ----------
                    Given the new context, refine the original summary.
                    If the context isn't useful, RETURN the original summary.
                    """
            )

            refine_chain = refine_prompt | llm | StrOutputParser()

            with st.status("Summarizing...") as status:
                for idx, doc in enumerate(docs[1:]):
                    status.update(label=f"Processing document {idx+1}/{len(docs)-1}")
                    summary = refine_chain.invoke(
                        {
                            "existing_summary": summary,
                            "context": doc.page_content,
                        }
                    )
            summary

    with qa_tab:
        input = st.text_input("Your question here")
        start = st.button("Generate Answer")
        if input and start:
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """
                        Answer the question using ONLY the following context.
                        If you don't know the answer just say you don't know. DON'T make anything up.
                        ---
                        Context: {context}
                        """,
                    ),
                    ("human", "{question}"),
                ]
            )
            retriever = embed_file(transcript_path, video.name)

            qa_chain = (
                {
                    "context": retriever | RunnableLambda(format_docs),
                    "question": RunnablePassthrough(),
                }
                | prompt
                | llm
                | StrOutputParser()
            )

            answer = qa_chain.invoke(input)
            answer
