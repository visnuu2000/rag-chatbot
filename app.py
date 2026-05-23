import streamlit as st
from rag_engine import get_pdf_text, get_chunks, get_vectorstore, get_answer

st.set_page_config(
    page_title="PDF Q&A Chatbot",
    layout="centered"
)

st.title("PDF Q&A Chatbot")
st.write("Upload a PDF and ask questions about it.")

# Initialize session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None

uploaded_file = st.file_uploader(
    "Upload your PDF file",
    type="pdf"
)

# Only re-index if a new file is uploaded
if uploaded_file is not None:
    if uploaded_file.name != st.session_state.last_uploaded_file:
        with st.spinner("Reading and indexing your PDF..."):
            raw_text = get_pdf_text(uploaded_file)

            if len(raw_text) < 50:
                st.error(
                    "Could not read text from this PDF. "
                    "Try a different PDF file."
                )
                st.session_state.vectorstore = None
            else:
                chunks = get_chunks(raw_text)
                st.session_state.vectorstore = get_vectorstore(chunks)
                st.session_state.messages = []  # Reset chat on new PDF
                st.session_state.last_uploaded_file = uploaded_file.name
                st.success(
                    f"PDF loaded! {len(chunks)} chunks indexed. "
                    "Ask me anything about it."
                )

# Show chat UI only when vectorstore is ready — outside the upload block
if st.session_state.vectorstore is not None:
    st.divider()

    # Render chat history
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