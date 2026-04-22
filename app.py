import streamlit as st
from rag_engine import get_pdf_text, get_chunks, get_vectorstore, get_answer

st.set_page_config(
    page_title="PDF Q&A Chatbot",
    layout="centered"
)

st.title("PDF Q&A Chatbot")
st.write("Upload a PDF and ask questions about it.")

uploaded_file = st.file_uploader(
    "Upload your PDF file",
    type="pdf"
)

if uploaded_file:
    with st.spinner("Reading and indexing your PDF..."):
        raw_text = get_pdf_text(uploaded_file)
        chunks = get_chunks(raw_text)
        if "vectorstore" not in st.session_state:
            st.session_state.vectorstore = get_vectorstore(chunks)
            st.session_state.messages = []

    st.success("PDF loaded! Ask me anything about it.")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    question = st.chat_input("Ask a question about your PDF...")

    if question:
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = get_answer(
                    st.session_state.vectorstore,
                    question
                )
            st.write(answer)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )