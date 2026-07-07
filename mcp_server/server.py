"""MCP server exposing the same Chroma retrieval capability as an MCP tool."""

from mcp.server.fastmcp import FastMCP

from rag.retriever import search_knowledge_base as retrieve

mcp = FastMCP("AI Engineering Knowledge Base")


@mcp.tool()
def search_knowledge_base(query: str) -> str:
    """Search the processed PDF knowledge base and return relevant excerpts with source and page information."""
    documents = retrieve(query)
    if not documents:
        return "No relevant text was found in the processed knowledge base."
    results = []
    for document in documents:
        page = document.metadata.get("page")
        page_label = f", page {page + 1}" if isinstance(page, int) else ""
        results.append(
            f"Source: {document.metadata.get('source', 'uploaded PDF')}{page_label}\n{document.page_content}"
        )
    return "\n\n---\n\n".join(results)


if __name__ == "__main__":
    mcp.run(transport="stdio")
