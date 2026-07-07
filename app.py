"""Streamlit interface for uploading a PDF and chatting with the RAG assistant."""

from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.graph import build_graph
from rag.ingest import process_pdf_bytes

load_dotenv()

st.set_page_config(page_title="AI Engineering Knowledge Assistant", page_icon="📚")
st.title("AI Engineering Knowledge Assistant")
st.caption("Ask questions about RAG, LangGraph, Tool Calling and MCP.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_name" not in st.session_state:
    st.session_state.processed_name = None
if "graph" not in st.session_state:
    st.session_state.graph = build_graph()

with st.sidebar:
    st.subheader("Knowledge base PDF")
    uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])
    if st.button("Process Document", disabled=uploaded_pdf is None, use_container_width=True):
        try:
            with st.spinner("Reading, splitting, and embedding the PDF..."):
                count = process_pdf_bytes(uploaded_pdf.getvalue(), uploaded_pdf.name)
            st.session_state.processed_name = uploaded_pdf.name
            st.session_state.chat_history = []
            st.success(f"Document processed successfully ({count} chunks). You can now ask questions.")
        except Exception as error:
            st.error(f"Could not process this PDF: {error}")

    # Offer the included demo PDF without requiring the user to copy or re-upload it.
    demo_pdf = Path(__file__).parent / "data" / "AI_Engineering_Fundamentals_Knowledge_Base.pdf"
    if demo_pdf.exists() and st.button("Load bundled demo PDF", use_container_width=True):
        try:
            with st.spinner("Processing the demo PDF..."):
                count = process_pdf_bytes(demo_pdf.read_bytes(), demo_pdf.name)
            st.session_state.processed_name = demo_pdf.name
            st.session_state.chat_history = []
            st.success(f"Document processed successfully ({count} chunks). You can now ask questions.")
        except Exception as error:
            st.error(f"Could not process the demo PDF: {error}")
    if st.session_state.processed_name:
        st.caption(f"Active document: {st.session_state.processed_name}")

for item in st.session_state.chat_history:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])
        if item.get("sources"):
            with st.expander("Retrieved sources"):
                for source in item["sources"]:
                    st.markdown(source)

prompt = st.chat_input("Ask a question about your processed PDF")
if prompt:
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        try:
            history = []
            for item in st.session_state.chat_history[:-1]:
                history.append(HumanMessage(content=item["content"]) if item["role"] == "user" else AIMessage(content=item["content"]))
            result = st.session_state.graph.invoke({"messages": [*history, HumanMessage(content=prompt)]})
            answer = result["messages"][-1].content
            if isinstance(answer, list):
                answer = "\n".join(part.get("text", "") for part in answer if isinstance(part, dict))
            st.markdown(answer)

            # Show passages returned during this graph run, so displayed sources
            # match the evidence the assistant actually received.
            tool_results = [message for message in result["messages"] if isinstance(message, ToolMessage)]
            sources = []
            for message in tool_results:
                content = message.content
                if isinstance(content, str) and content.startswith("Source:"):
                    sources.extend(content.split("\n\n---\n\n"))
            if sources:
                with st.expander("Retrieved sources"):
                    for source in sources:
                        st.markdown(source)
            st.session_state.chat_history.append({"role": "assistant", "content": answer, "sources": sources})
        except Exception as error:
            st.error(f"The assistant could not complete this request: {error}")
