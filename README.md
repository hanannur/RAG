# AI Engineering Knowledge Assistant

A beginner-friendly RAG demo that answers questions from a PDF. It demonstrates document loading, chunking, Gemini embeddings, Chroma retrieval, LangChain tool calling, a LangGraph workflow, a Streamlit chat UI, and a separate MCP server.

## Learning objectives

- **RAG:** retrieve relevant PDF passages before generating document-grounded answers.
- **Embeddings:** use Gemini to represent text as vectors that can be compared by meaning.
- **Chroma:** persist those vectors locally and search them by similarity.
- **Tool calling:** let the model request `search_knowledge_base` when it needs document evidence.
- **LangGraph:** represent the model/tool loop as a small stateful graph.
- **MCP:** expose the same retrieval function through a real MCP server tool using the MCP Python SDK.

## Architecture

```text
PDF upload (or bundled PDF)
        |
        v
PDF loader -> text splitter -> Gemini embeddings -> local Chroma
                                                   ^
                                                   |
User -> Streamlit -> LangGraph assistant -> search_knowledge_base
                    ^                    <- retrieved passages + sources
                    |                                  |
                    +----------- Gemini ---------------+

MCP client -> MCP server -> search_knowledge_base -> same Chroma store
```

## Setup (Windows PowerShell)

From the project directory:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
notepad .env
```

Put your Gemini API key in `.env` as `GEMINI_API_KEY=...`. The application loads it through `python-dotenv`; never commit `.env`. The checked-in `.gitignore` excludes it.

The bundled demo PDF is in `data/AI_Engineering_Fundamentals_Knowledge_Base.pdf`. To index it from PowerShell:

```powershell
python -m rag.ingest
```

Then start the interface:

```powershell
streamlit run app.py
```

You can also upload another PDF in the sidebar and click **Process Document**. Processing replaces the active Chroma collection, so chunks from separate PDFs are not combined. The UI stays open and updates the retriever immediately. The “Load bundled demo PDF” button is an alternate way to load the included document.

Start the MCP server in a second activated terminal from the project directory:

```powershell
python -m mcp_server.server
```

It uses standard input/output transport, which is intended for an MCP host to launch and communicate with the server. A plain terminal may appear idle while it waits for an MCP client. The MCP server searches the same local Chroma store as the app, so process a document first. Stop it with `Ctrl+C`.

## How RAG works

The ingestion pipeline is **PDF → chunks → embeddings → Chroma**. PyPDF extracts text and page metadata. `RecursiveCharacterTextSplitter` uses 800-character chunks with 120 characters of overlap: this keeps each piece small enough for focused retrieval while overlap helps preserve sentences that cross a boundary. Gemini's `models/gemini-embedding-001` creates vectors, stored in a local persistent Chroma collection. At question time, similarity search returns up to four relevant chunks, including source and page metadata.

## How LangGraph works

The graph state contains the conversation's messages. The `assistant` node sends them, with a system instruction, to Gemini. A conditional edge checks if Gemini requested a tool. If so, the `tools` node runs the retrieval tool and puts its result in the message history; control returns to the assistant so it can write a grounded answer. If there is no tool call, the graph ends. This is the simple model → optional tool → model loop.

## How tool calling works

The LangChain tool has a name, description, and a `query` input. Its description tells Gemini to use it for questions that depend on the PDF's topics. Gemini decides when a tool is needed; the Python tool performs the actual Chroma search and returns excerpts with source/page information. The model then uses that evidence to answer. A question unrelated to the PDF may be answered conversationally, and the assistant is instructed to distinguish that from document evidence.

## How MCP works

The Model Context Protocol (MCP) is a standard interface that lets an AI application discover and call tools provided by another process or service. An **MCP server** publishes capabilities; an **MCP tool** is a named operation with documented inputs and outputs. This project uses the MCP Python SDK and registers `search_knowledge_base` with its tool decorator. The MCP host can discover the tool and call it with a query; the server searches Chroma and returns matching passages.

The LangChain tool is invoked inside this app's LangGraph by Gemini's tool-call workflow. The MCP tool is exposed over the MCP protocol for a separate compatible host to discover and call. Both call the same retrieval helper, but MCP is not just a Python function renamed as a tool: the SDK registers it and serves it over the MCP stdio transport. To avoid unsupported factual answers, the assistant searches the PDF for factual questions and reports when it has no relevant evidence; greetings do not need a search.

## Example questions

- What is RAG?
- What are embeddings?
- What is tool calling?
- What is LangGraph?
- What is MCP?
- How do LangGraph, MCP, and RAG work together?
- What is the population of Addis Ababa? (A test of the assistant's response when the PDF has no relevant evidence.)

## Project limitations

This is a learning demo, not a production RAG platform. It uses one local Chroma collection and replaces its contents on each upload. PDF extraction is text based and may not read scanned/image-only PDFs. Retrieval quality depends on the source document, chunking, and embeddings. API calls require a valid Gemini key and internet access. There is no authentication, multi-user isolation, upload-size management, or production monitoring.
