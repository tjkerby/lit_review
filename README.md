## Project Overview

The **Knowledge Graph & RAG Pipeline** builds a semantic layer over academic papers by ingesting PDFs, chunking text, and embedding fragments into a Neo4j-based vector store. It then spins up a LangGraph agent powered by an Ollama LLM (e.g., `llama3.1:8b`), enabling Retrieval-Augmented Generation (RAG) with stateful memory persisted in SQLite.

### Purpose and Goals

- **Centralize Paper Metadata**: Harvest titles, authors, and citations via Semantic Scholar API. 
- **Fine-Grained Retrieval**: Split full-text PDFs into manageable chunks linked to their paper node for precise RAG searches. 
- **Interactive Exploration**: Provide a Gradio interface for chat-style querying, combining LLM reasoning with graph-based context. 

### High-Level Architecture

1. **Data Ingestion**:  
   - Semantic Scholar scraper → JSON of paper metadata.  
   - PDF loader & chunker → sequential “chunk nodes” linked to paper nodes.

2. **Vector Store Population**:  
   - Generate embeddings for abstracts and chunks.  
   - Store vectors in Neo4j as properties on chunk nodes for efficient kNN search. 

3. **Agent & Tools**:  
   - LangGraph state machine orchestrates streaming LLM calls, tool detection, and execution.  
   - Custom LangGraph tool wraps Neo4j queries for RAG retrieval.  

4. **User Interface**:  
   - Gradio app ties the entire flow into a web-based chat with memory, vector search, and LLM responses. 

## Core Components

### Knowledge Graph Construction

- **`neo4j_utils.py`**: Helpers to connect, create nodes/relationships, and manage indexes.  
- **`kg_builder.py`**: Fetches citations from existing nodes and links papers via `CITES` edges. 

### Document Ingestion and Chunking

- **`create_chunks.py`**:  
  - Reads JSON of `{paper_id: pdf_path}`.  
  - Splits text into overlapping chunks.  
  - Creates sequential chunk nodes pointing back to their paper node. 

### Embeddings and Vector Store

- **`add_abstract_embeddings.py`**: Embeds paper abstracts and stores vectors in Neo4j.  
- **`rag_utils.py`**: Wraps various embedding frameworks (e.g., OpenAI, SentenceTransformers). 

### Agent and Tools

- **`agent.py`**: Defines the LangGraph `StateGraph` with nodes for LLM calls and tool execution loops.  
- **`agent_tools.py`**: Implements a Neo4j “search chunks” tool conforming to LangChain’s tool interface. 

### User Interface

- **`gradio_langgraph.py`**:  
  - Initializes the RAG tool, Ollama model binding, and memory checkpointer (`load_memory`).  
  - Launches a Gradio chat interface with custom callbacks. 

## Scripts

- **`build_kg.py`**: Automates Semantic Scholar scraping for title/author/citation ingestion.  
- **`create_chunks.py`**: Generates chunk nodes and embeds them.  
- **`add_abstract_embeddings.py`**: Builds vector index for abstracts.  
- **`gradio_langgraph.py`**: Bootstraps the full RAG-agent demo. 

## Best Practices Followed

### Concise, Modular Documentation  
Each module has a single, clear responsibility, and top-level docstrings explain inputs, outputs, and side effects, ensuring maintainability. 

### Standardized Structure  
- **README First**: Project goals and quickstart are front-and-center, guiding new contributors.
- **Consistent Terminology**: Uses “paper node,” “chunk node,” and “RAG tool” uniformly.

## Usage and Hosting

### Prerequisites

- Ollama server running locally  
- Neo4j instance accessible with proper credentials  
- SQLite for memory persistence  

### Launching the Gradio App

```bash
python gradio_langgraph.py
```

The interface will be available at `http://localhost:7861`, providing chat with live RAG and memory support.
