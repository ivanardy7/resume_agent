from __future__ import annotations
import pandas as pd
import streamlit as st
from agents import run_hr_agent

st.set_page_config(
    page_title="HR Candidate Screening Assistant",
    page_icon="🧑‍💼",
    layout="wide",
)

st.title("🧑‍💼 HR Candidate Screening Assistant")
st.caption(
    "Multi-Agent RAG chatbot untuk mencari, menganalisis, dan ranking kandidat berdasarkan database resume."
)

with st.sidebar:
    st.header("Tentang Aplikasi")
    st.markdown(
        """
        Aplikasi ini menggunakan:
        - Resume dataset sebagai knowledge base
        - OpenAI Embedding untuk vectorization
        - Qdrant Cloud sebagai vector database
        - LangGraph untuk multi-agent workflow
        - Streamlit sebagai UI
        """
    )

    st.header("Contoh Pertanyaan")
    st.markdown(
        """
        - Cari kandidat yang punya pengalaman Python dan SQL.
        - Kandidat mana yang cocok untuk role Data Analyst?
        - Cari kandidat dengan pengalaman sales dan business development.
        - Apa kandidat terbaik untuk posisi HR Manager?
        - Tampilkan kandidat engineering yang punya pengalaman project management.
        """
    )

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Tanya kebutuhan kandidat / resume di sini...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    recent_history = st.session_state.messages[-15:]

    with st.chat_message("assistant"):
        with st.spinner("Mencari dan menganalisis resume..."):
            try:
                result = run_hr_agent(prompt, recent_history)
                answer = result.get("answer", "Maaf, jawaban tidak berhasil dibuat.")
            except Exception as exc:
                answer = (
                    "Terjadi error saat menjalankan agent. "
                    "Cek API key, Qdrant collection, dan apakah create_vector_db.py sudah dijalankan.\n\n"
                    f"Error detail: `{exc}`"
                )
                result = {}

        st.markdown(answer)

        if result:
            sources = result.get("sources", [])
            if sources:
                with st.expander("📚 Retrieved Resume Sources"):
                    source_table = pd.DataFrame(
                        [
                            {
                                "resume_id": item.get("resume_id"),
                                "category": item.get("category"),
                                "chunk_index": item.get("chunk_index"),
                                "score": item.get("score"),
                                "snippet": item.get("content", "")[:300] + "...",
                            }
                            for item in sources
                        ]
                    )
                    st.dataframe(source_table, use_container_width=True)

            with st.expander("🧠 Agent Workflow Details"):
                st.markdown("**Search query yang dibuat Query Understanding Agent:**")
                st.code(result.get("search_query", "-"))

                st.markdown("**Resume Analyst Agent output:**")
                st.code(result.get("analysis", "-"))

                st.markdown("**Candidate Ranking Agent output:**")
                st.code(result.get("ranking", "-"))

            with st.expander("💬 Chat History Used"):
                st.json(recent_history)

            with st.expander("💰 Token Usage"):
                st.code(
                    f"Input tokens  : {result.get('total_input_tokens', 0)}\n"
                    f"Output tokens : {result.get('total_output_tokens', 0)}\n"
                    f"Total tokens  : {result.get('total_tokens', 0)}\n"
                    f"Est. cost IDR : {result.get('estimated_cost_idr', 0):.4f}"
                )

    st.session_state.messages.append({"role": "assistant", "content": answer})