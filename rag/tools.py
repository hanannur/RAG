"""LangChain retrieval tool definition."""

from langchain_core.tools import tool

from rag.retriever import search_knowledge_base as retrieve


@tool
def search_knowledge_base(query: str) -> str:
    """Search the processed PDF knowledge base before answering factual questions. Use it for any question that needs facts, especially about RAG, embeddings, vector search, tool calling, LangGraph, and MCP. Returns relevant excerpts with source and page metadata; if no indexed content is available, says so."""
    try:
        documents = retrieve(query)
    except Exception as error:
        return f"Knowledge base search is unavailable: {error}"
    if not documents:
        return "No relevant text was found in the processed knowledge base."

    excerpts = []
    for document in documents:
        source = document.metadata.get("source", "uploaded PDF")
        page = document.metadata.get("page")
        page_label = f", page {page + 1}" if isinstance(page, int) else ""
        excerpts.append(f"Source: {source}{page_label}\n{document.page_content}")
    return "\n\n---\n\n".join(excerpts)
