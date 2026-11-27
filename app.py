"""Streamlit chat UI.  Run:  streamlit run app.py"""
import streamlit as st

from rag import RAGChat

st.set_page_config(page_title="RAG Chatbot", page_icon="💬")
st.title("💬 RAG Chatbot")

if "chat" not in st.session_state:
    st.session_state.chat = RAGChat()
    st.session_state.messages = []  # display transcript: {"role", "content", "sources"}

with st.sidebar:
    if st.button("New conversation"):
        st.session_state.chat.reset()
        st.session_state.messages = []
        st.rerun()

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("Sources"):
                st.write("\n".join(f"- {s}" for s in m["sources"]))

if question := st.chat_input("Ask about your documents..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        answer = st.write_stream(st.session_state.chat.stream(question))
        sources = sorted(set(st.session_state.chat.last_sources))
        with st.expander("Sources"):
            st.write("\n".join(f"- {s}" for s in sources))

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
