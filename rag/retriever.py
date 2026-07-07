"""Small retrieval wrapper shared by the UI, agent, and MCP server."""

from langchain_core.documents import Document

from rag.store import open_vectorstore


def search_knowledge_base(query: str, k: int = 4) -> list[Document]:
    """Return the most relevant chunks for a query, or an empty list if unindexed."""
    if not query.strip():
        return []
    store = open_vectorstore()
    if store._collection.count() == 0:
        return []
    return store.similarity_search(query, k=k)
