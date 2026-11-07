# What Would Lee Kuan Yew Do?

A RAG (retrieval-augmented generation) chatbot that answers questions in the voice of Lee Kuan Yew, grounded in his public speeches and interviews. Built with Python, Streamlit, and the OpenAI API.

**Live demo:** _add your Streamlit Cloud link here_

## How it works

```
                 ┌──────────────── preparation (ingest.py, run once) ────────────────┐
data/docs/*.txt ─┤ split into ~500-word chunks → embed (text-embedding-3-small) → save │─→ data/index/
                 └────────────────────────────────────────────────────────────────────┘

user question → embed → cosine similarity vs all chunks → top 5 chunks
              → system prompt (LKY persona + grounding rules) + chunks + chat history
              → gpt-5-mini → answer + list of sources shown in the app
```

- `rag.py`: retrieval and answer generation, including the system prompt
- `convert_pdfs.py`: converts National Archives speech PDFs into clean .txt files
- `ingest.py`: builds the index from the documents
- `app.py`: Streamlit chat interface
- `eval.py` + `eval/questions.json`: automatic evaluation with an LLM judge

The corpus is small, so embeddings are stored in a NumPy file and searched in memory instead of a vector database. This keeps the project simple and free to host.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # then put your OpenAI key in .env

python convert_pdfs.py             # optional: turn PDFs in data/pdfs into .txt files in data/docs
python ingest.py                   # build the index from data/docs
streamlit run app.py               # open the chat app
python eval.py                     # run the evaluation → eval/results.md
```

## Data sources

_List the speeches and interviews you used, with links._

Only publicly available speeches and interviews are included. His memoirs were deliberately left out because this repository is public and redistributing copyrighted book text would not be appropriate.

## Evaluation

`eval/questions.json` has 15 test questions in four categories:

| Category | What it tests |
|---|---|
| covered | Topics the sources clearly cover: does retrieval find them and does the answer use them? |
| advice | Personal and leadership advice: does it sound like him? |
| out_of_scope | Topics after 2015 or that he never discussed: does it admit the limit instead of inventing? |
| adversarial | Fake quotes, false premises, attempts to break character |

Each answer is scored 1–5 by an LLM judge on groundedness, persona, and honesty. Results: see [`eval/results.md`](eval/results.md).

_Summarize your scores here, and what you changed after the first run._

## What went wrong and what I changed

_Write your own notes from building this: problems you hit, how you fixed them, and what you would do with more time._
