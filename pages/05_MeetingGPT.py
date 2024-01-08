import streamlit as st
import subprocess
import math
from pydub import AudioSegment
import glob
import openai
import os

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


with st.sidebar:
    video = st.file_uploader("Video", type=["mp4", "avi", "mkv", "mov"])

if video:
    chunks_path = "./.cache/chunks"
    with st.status("Loading video..."):
        video_content = video.read()
        video_path = f".cache/{video.name}"
        audio_path = video_path.replace("mp4", "mp3")
        transcript_path = video_path.replace("mp4", "txt")
        with open(video_path, "wb") as f:
            f.write(video_content)

    with st.status("Extracting audio..."):
        extract_audio_from_video(video_path)

    with st.status("Cutting audio segments..."):
        cut_audio_in_chunks(audio_path, 6, chunks_path)

    with st.status("Transcribe audio..."):
        transcribe_chunks(chunks_path, transcript_path)
