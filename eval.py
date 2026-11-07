"""Evaluate the chatbot on the test questions in eval/questions.json.

Run:  python eval.py
Output: eval/results.md (summary table + full answers) and eval/results.json

Each answer is scored 1-5 by an LLM judge on:
  groundedness - are its facts and quotes supported by the retrieved passages?
  persona      - does it sound like Lee Kuan Yew?
  honesty      - does it admit when the sources don't cover something, instead of inventing?
For questions with an "expected_source", we also check whether retrieval found that document.
"""
import json
import statistics
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
import rag  # noqa: E402

EVAL_DIR = Path(__file__).parent / "eval"
JUDGE_MODEL = rag.CHAT_MODEL
CRITERIA = ["groundedness", "persona", "honesty"]

JUDGE_PROMPT = """You are evaluating an AI chatbot that simulates Lee Kuan Yew using passages retrieved from his speeches and interviews.

Score the ANSWER from 1 to 5 on each criterion:
- groundedness: Are the answer's factual claims and any quotes supported by the CONTEXT? 5 = fully supported, 1 = invents facts or quotes.
- persona: Does it sound like Lee Kuan Yew (first person, direct, pragmatic, long-term thinking)? 5 = convincing, 1 = sounds like a generic assistant.
- honesty: When the CONTEXT does not cover the question, or the question contains a false claim, does the answer say so instead of inventing? If the CONTEXT fully covers the question, score 5.

Also check the answer against the EXPECTED BEHAVIOR for this test case.

Respond only with JSON: {"groundedness": int, "persona": int, "honesty": int, "reason": "one short sentence"}"""


def judge(case: dict, reply: str, results: list[dict]) -> dict:
    content = (
        f"QUESTION:\n{case['question']}\n\n"
        f"EXPECTED BEHAVIOR:\n{case['expected_behavior']}\n\n"
        f"CONTEXT:\n{rag.format_context(results)}\n\n"
        f"ANSWER:\n{reply}"
    )
    kwargs = {
        "model": JUDGE_MODEL,
        "messages": [{"role": "system", "content": JUDGE_PROMPT}, {"role": "user", "content": content}],
        "response_format": {"type": "json_object"},
    }
    if JUDGE_MODEL.startswith("gpt-5"):
        kwargs["reasoning_effort"] = "low"
    resp = rag.get_client().chat.completions.create(**kwargs)
    try:
        return json.loads(resp.choices[0].message.content)
    except json.JSONDecodeError:
        return {c: 0 for c in CRITERIA} | {"reason": "Judge returned invalid JSON"}


def avg(rows, key):
    values = [r[key] for r in rows if isinstance(r.get(key), (int, float)) and r[key] > 0]
    return round(statistics.mean(values), 2) if values else "-"


def clean(text: str) -> str:
    return str(text).replace("|", "/").replace("\n", " ")


def main():
    cases = json.loads((EVAL_DIR / "questions.json").read_text(encoding="utf-8"))
    chunks, embeddings = rag.load_index()
    rows = []

    for case in cases:
        print(f"[{case['id']}] {case['question']}")
        reply, results = rag.answer(case["question"], chunks, embeddings)
        scores = judge(case, reply, results)

        hit = "-"
        if case.get("expected_source"):
            target = case["expected_source"].lower()
            hit = "yes" if any(target in r["title"].lower() for r in results) else "no"

        rows.append({
            **case, **scores,
            "retrieval_hit": hit,
            "answer": reply,
            "sources": [f"{r['title']} ({r['score']:.2f})" for r in results],
        })

    # Markdown report
    lines = [
        "# Evaluation results",
        "",
        f"Run on {datetime.now():%Y-%m-%d %H:%M} · chat model `{rag.CHAT_MODEL}` · judge `{JUDGE_MODEL}` · top-k {rag.TOP_K}",
        "",
        "## Averages",
        "",
        "| Category | Groundedness | Persona | Honesty |",
        "|---|---|---|---|",
    ]
    for cat in sorted({r["category"] for r in rows}):
        group = [r for r in rows if r["category"] == cat]
        lines.append(f"| {cat} | " + " | ".join(str(avg(group, c)) for c in CRITERIA) + " |")
    lines.append("| **Overall** | " + " | ".join(f"**{avg(rows, c)}**" for c in CRITERIA) + " |")

    hits = [r for r in rows if r["retrieval_hit"] != "-"]
    if hits:
        rate = sum(r["retrieval_hit"] == "yes" for r in hits) / len(hits)
        lines += ["", f"Retrieval hit rate (expected source in top {rag.TOP_K}): **{rate:.0%}** ({len(hits)} questions)"]

    lines += [
        "", "## Per question", "",
        "| ID | Category | Question | G | P | H | Retrieval | Judge's reason |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['id']} | {r['category']} | {clean(r['question'])} | {r.get('groundedness')} | "
            f"{r.get('persona')} | {r.get('honesty')} | {r['retrieval_hit']} | {clean(r.get('reason', ''))} |"
        )

    lines += ["", "## Full answers", ""]
    for r in rows:
        lines += [f"### {r['id']}: {r['question']}", "", r["answer"], "", "Sources: " + "; ".join(r["sources"]), ""]

    (EVAL_DIR / "results.md").write_text("\n".join(lines), encoding="utf-8")
    (EVAL_DIR / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nOverall: " + ", ".join(f"{c} {avg(rows, c)}" for c in CRITERIA))
    print(f"Saved eval/results.md")


if __name__ == "__main__":
    main()
