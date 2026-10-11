"""Streamlit front end for the legal-rag API.

    uv run --group ui streamlit run ui/app.py

Settings come from Streamlit secrets (Streamlit Community Cloud) or environment variables:
API_URL (default http://127.0.0.1:8000) and API_KEY (an issued key, when the API requires one).
"""

from __future__ import annotations

import html
import os
from itertools import chain
from typing import Any

import streamlit as st
from rag_client import Answer, RagClient, is_arabic

st.set_page_config(page_title="Egyptian Civil Code Q&A", page_icon="⚖️", layout="centered")

EXAMPLES = [
    "ما هي مدة التقادم في الالتزامات؟",
    "When does a person's legal personality begin?",
    "هل يجوز طلب إبطال العقد إذا وقعت في غلط جوهري؟",
    "What is a lease contract?",
]

st.markdown(
    """<style>
    .answer { line-height: 1.8; }
    .rtl { direction: rtl; text-align: right; }
    .chips span { display: inline-block; margin: 0 .3rem .3rem 0; padding: .1rem .55rem;
                  border-radius: 999px; font-size: .8rem; background: rgba(128,128,128,.15); }
    </style>""",
    unsafe_allow_html=True,
)


def setting(name: str, default: str | None = None) -> str | None:
    try:
        if name in st.secrets:
            return st.secrets[name]
    except FileNotFoundError:  # no secrets file when running locally
        pass
    return os.environ.get(name, default)


@st.cache_resource
def get_client() -> RagClient:
    return RagClient(setting("API_URL", "http://127.0.0.1:8000"), setting("API_KEY"))


def show_text(text: str, target: Any = st) -> None:
    css = "answer rtl" if is_arabic(text) else "answer"
    body = html.escape(text).replace("\n", "<br>")
    target.markdown(f'<div class="{css}">{body}</div>', unsafe_allow_html=True)


def send_feedback(request_id: str) -> None:
    rating = st.session_state.get(f"fb-{request_id}")
    if rating is not None:
        get_client().feedback(request_id, "up" if rating == 1 else "down")


def show_details(answer: Answer, question: str) -> None:
    if answer.error:
        st.error(answer.error)
        return
    if answer.sources:
        arabic = is_arabic(question)
        with st.expander(f"Sources ({len(answer.sources)})", expanded=True):
            for s in answer.sources:
                cite = s.get("citation_ar") if arabic else s.get("citation")
                st.markdown(f"**Art. {s['article_number']}** · {cite or ''}")
    chips = []
    if answer.refused:
        chips.append("refused: no article answers this")
    chips += [f"guard: {g}" for g in answer.guardrails]
    if total := answer.timings_ms.get("total"):
        chips.append(f"{total / 1000:.1f} s")
    if chips:
        spans = "".join(f"<span>{html.escape(c)}</span>" for c in chips)
        st.markdown(f'<div class="chips">{spans}</div>', unsafe_allow_html=True)
    if answer.request_id and not answer.refused:
        st.feedback(
            "thumbs",
            key=f"fb-{answer.request_id}",
            on_change=send_feedback,
            args=(answer.request_id,),
        )


def sidebar() -> None:
    client = get_client()
    st.sidebar.header("Egyptian Civil Code Q&A")
    st.sidebar.caption(
        "Answers come only from the articles of the Civil Code and cite them. "
        "Questions the code does not cover get a refusal. Not legal advice."
    )
    try:
        health = client.health()
        meta = client.metadata()
        st.sidebar.success(f"Connected · {health.get('documents_indexed', '?')} articles indexed")
        st.sidebar.caption(
            f"Model `{meta.get('llm_model')}` · prompt {meta.get('prompt_version')} · "
            f"decider {meta.get('decider_backend')}"
        )
    except Exception:  # noqa: BLE001 - any failure here just means "offline"
        st.sidebar.error("The API is not reachable right now.")
    st.sidebar.subheader("Try")
    for example in EXAMPLES:
        if st.sidebar.button(example, width="stretch"):
            st.session_state.pending = example
    if st.sidebar.button("Clear chat", type="secondary"):
        st.session_state.messages = []


def main() -> None:
    st.session_state.setdefault("messages", [])
    sidebar()
    st.title("⚖️ Egyptian Civil Code Q&A")
    st.caption("Ask in Arabic or English. اسأل بالعربي أو بالإنجليزي.")

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            show_text(m["text"])
            if m["role"] == "assistant":
                show_details(m["answer"], m["question"])

    question = st.chat_input("Ask about the Egyptian Civil Code") or st.session_state.pop(
        "pending", None
    )
    if not question:
        return
    st.session_state.messages.append({"role": "user", "text": question})
    with st.chat_message("user"):
        show_text(question)
    with st.chat_message("assistant"):
        placeholder = st.empty()
        answer = Answer()
        with st.spinner("Searching the Civil Code..."):
            events = get_client().stream(question)
            first = next(events, None)
        for event in chain([first], events) if first else []:
            answer.apply(event)
            show_text(answer.text + ("" if answer.done else " ▌"), placeholder)
        if answer.error:
            placeholder.empty()
        show_details(answer, question)
    st.session_state.messages.append(
        {"role": "assistant", "text": answer.text, "answer": answer, "question": question}
    )


main()
