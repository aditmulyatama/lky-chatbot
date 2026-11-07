"""Streamlit chat UI for the Lee Kuan Yew RAG chatbot.

Run locally:  streamlit run app.py
"""
import os

import streamlit as st
from dotenv import load_dotenv

# Load the API key: .env locally, or Streamlit secrets when deployed on Streamlit Community Cloud.
load_dotenv()
for key in ("OPENAI_API_KEY", "CHAT_MODEL"):
    if key not in os.environ:
        try:
            if key in st.secrets:
                os.environ[key] = st.secrets[key]
        except Exception:
            pass  # no secrets file, e.g. running locally with .env

import rag  # noqa: E402  (import after the key is set)

st.set_page_config(page_title="What Would Lee Kuan Yew Do?", page_icon="🇸🇬")

EXAMPLES = [
    "How should a small country survive between great powers?",
    "What makes a good leader?",
    "What's your advice for young people starting their careers?",
    "Why did Singapore focus on meritocracy?",
]


@st.cache_resource
def get_index():
    return rag.load_index()


st.title("What Would Lee Kuan Yew Do?")
st.caption(
    "An AI simulation based on Lee Kuan Yew's public speeches and interviews. "
    "It is not Lee Kuan Yew, and answers can contain mistakes. Check the sources under each reply."
)

with st.sidebar:
    st.subheader("How it works")
    st.write(
        "Your question is matched against passages from Lee Kuan Yew's speeches and interviews. "
        f"The {rag.TOP_K} most relevant passages are given to the model, which answers in his voice "
        "and shows which sources it used."
    )
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY is not set. Add it to a .env file locally, or to the app's secrets on Streamlit Cloud.")
    st.stop()

try:
    chunks, embeddings = get_index()
except FileNotFoundError:
    st.error("The search index is missing. Run `python ingest.py` and commit the files in data/index.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []


def show_sources(sources):
    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            title = f"[{s['title']}]({s['source']})" if s.get("source") else s["title"]
            st.markdown(f"**{title}** · {s['date']} · relevance {s['score']:.2f}")
            st.caption(s["text"][:300] + "...")


# Replay the conversation so far
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            show_sources(msg["sources"])

# Example questions for an empty chat
clicked = None
if not st.session_state.messages:
    st.write("Try one of these:")
    cols = st.columns(2)
    for i, example in enumerate(EXAMPLES):
        if cols[i % 2].button(example, use_container_width=True):
            clicked = example

question = st.chat_input("Ask Lee Kuan Yew a question") or clicked

if question:
    with st.chat_message("user"):
        st.markdown(question)

    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        with st.spinner("Searching the speeches and thinking..."):
            try:
                reply, sources = rag.answer(question, chunks, embeddings, history)
            except Exception as e:
                st.error(f"Couldn't get an answer: {e}")
                st.stop()
        st.markdown(reply)
        show_sources(sources)

    st.session_state.messages.append({"role": "assistant", "content": reply, "sources": sources})
    if clicked:
        st.rerun()  # hide the example buttons after the first question
