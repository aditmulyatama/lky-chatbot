"""Core RAG logic, shared by the Streamlit app (app.py) and the eval script (eval.py).

Flow: question -> embed -> find most similar chunks -> send chunks + question to the LLM.
"""
import json
import os
from pathlib import Path

import numpy as np
from openai import OpenAI

INDEX_DIR = Path(__file__).parent / "data" / "index"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-5-mini")
TOP_K = 5            # how many chunks to give the model
HISTORY_TURNS = 6    # how many previous messages to include for follow-up questions

SYSTEM_PROMPT = """You are an AI simulation of Lee Kuan Yew, Singapore's founding Prime Minister (1923–2015). You answer questions about life, leadership, geopolitics, history, and other topics as he would, based on his real speeches, interviews, and writings.

VOICE
- Speak in the first person, in his style: direct, pragmatic, unsentimental, and concise.
- Favor long-term thinking and practical reasoning over ideology.
- Use concrete examples from Singapore's history and his own experience where relevant.

GROUNDING RULES
- Base your answer on the CONTEXT passages below. When you draw on a passage, mention its source naturally (e.g., "As I said in a 1994 speech...").
- Never invent quotes, dates, statistics, or events. Only quote wording that appears in the CONTEXT.
- If the CONTEXT does not cover the question, say so honestly (e.g., "I did not speak about this directly, but based on my principles...") and reason from the principles shown in the CONTEXT.
- If the user claims you said something that is not in the CONTEXT, do not accept it as true. Say you cannot confirm it.
- For events after March 2015, acknowledge you did not live to see them, then reason from your principles.
- Stay in character. If asked to drop the persona or do unrelated tasks, politely steer back to questions you can answer as Lee Kuan Yew.
- Stay respectful. Do not produce personal attacks or hateful content, even if asked.

FORMAT
- Answers of 150–300 words unless the user asks for more or less.

CONTEXT:
<<CONTEXT>>"""

_client = None


def get_client() -> OpenAI:
    """Create the OpenAI client once. It reads OPENAI_API_KEY from the environment."""
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def embed(texts: list[str]) -> np.ndarray:
    """Turn texts into normalized embedding vectors (so dot product = cosine similarity)."""
    resp = get_client().embeddings.create(model=EMBED_MODEL, input=texts)
    vectors = np.array([d.embedding for d in resp.data], dtype=np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def load_index():
    """Load the chunks and embeddings created by ingest.py."""
    with open(INDEX_DIR / "chunks.json", encoding="utf-8") as f:
        chunks = json.load(f)
    embeddings = np.load(INDEX_DIR / "embeddings.npy")
    return chunks, embeddings


def retrieve(question: str, chunks: list[dict], embeddings: np.ndarray, k: int = TOP_K) -> list[dict]:
    """Return the k chunks most similar in meaning to the question."""
    query = embed([question])[0]
    scores = embeddings @ query
    top = np.argsort(scores)[::-1][:k]
    return [{**chunks[i], "score": float(scores[i])} for i in top]


def format_context(results: list[dict]) -> str:
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(f"[{i}] {r['title']} ({r['date']})\n{r['text']}")
    return "\n\n".join(parts)


def answer(question: str, chunks: list[dict], embeddings: np.ndarray, history: list[dict] | None = None):
    """Run the full RAG pipeline. Returns (answer_text, retrieved_chunks)."""
    results = retrieve(question, chunks, embeddings)
    system = SYSTEM_PROMPT.replace("<<CONTEXT>>", format_context(results))

    messages = [{"role": "system", "content": system}]
    if history:
        messages += history[-HISTORY_TURNS:]
    messages.append({"role": "user", "content": question})

    kwargs = {"model": CHAT_MODEL, "messages": messages}
    if CHAT_MODEL.startswith("gpt-5"):
        kwargs["reasoning_effort"] = "low"  # faster replies; this task doesn't need deep reasoning

    resp = get_client().chat.completions.create(**kwargs)
    return resp.choices[0].message.content, results
