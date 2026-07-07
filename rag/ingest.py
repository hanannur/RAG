"""Load PDF files, split them, embed chunks, and replace the local collection."""

from pathlib import Path
from tempfile import NamedTemporaryFile

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from rag.store import COLLECTION_NAME, VECTORSTORE_DIR, get_embeddings

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120


def split_pdf(pdf_path: str | Path):
    """Read a PDF and return useful, page-labelled text chunks."""
    reader = PdfReader(str(pdf_path))
    pages = [
        Document(page_content=page.extract_text() or "", metadata={"source": str(pdf_path), "page": index})
        for index, page in enumerate(reader.pages)
    ]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
    chunks = splitter.split_documents(pages)
    return [chunk for chunk in chunks if chunk.page_content.strip()]


def replace_knowledge_base(chunks) -> int:
    """Replace all old vectors so unrelated uploads are not mixed together."""
    if not chunks:
        raise ValueError("The PDF did not contain extractable text.")

    VECTORSTORE_DIR.parent.mkdir(parents=True, exist_ok=True)
    # Delete only this app's collection; the persistent directory can be reused.
    store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(VECTORSTORE_DIR),
    )
    store.delete_collection()
    Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=COLLECTION_NAME,
        persist_directory=str(VECTORSTORE_DIR),
    )
    return len(chunks)


def process_pdf_bytes(pdf_bytes: bytes, filename: str) -> int:
    """Process uploaded PDF bytes using a temporary file for the PDF loader."""
    if not filename.lower().endswith(".pdf") or not pdf_bytes.startswith(b"%PDF"):
        raise ValueError("Please upload a valid PDF file.")
    with NamedTemporaryFile(suffix=".pdf") as temporary_pdf:
        temporary_pdf.write(pdf_bytes)
        temporary_pdf.flush()
        chunks = split_pdf(temporary_pdf.name)
    for chunk in chunks:
        chunk.metadata["source"] = filename
    return replace_knowledge_base(chunks)


def ingest_pdf(pdf_path: str | Path) -> int:
    """Ingest a local PDF from the command line."""
    chunks = split_pdf(pdf_path)
    return replace_knowledge_base(chunks)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Index a PDF into local Chroma.")
    parser.add_argument("pdf", nargs="?", default="data/AI_Engineering_Fundamentals_Knowledge_Base.pdf")
    args = parser.parse_args()
    count = ingest_pdf(args.pdf)
    print(f"Indexed {count} chunks from {args.pdf}")
