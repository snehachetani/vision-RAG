# VisionRAG: Multimodal PDF Chatbot

VisionRAG is a visual document retrieval and question-answering system. Unlike traditional text-only RAG pipelines that parse documents into plain text, VisionRAG indexes documents visually using page screenshots. This allows the system to read and answer questions about complex layouts, tables, charts, and mathematical formulas directly.

## Purpose and Rationale

Traditional Retrieval-Augmented Generation (RAG) suffers from parsing loss because converting PDF layouts (with columns, headers, sidebars, charts, and tables) into a flat text stream strips away critical visual relationships. 

VisionRAG addresses this by using a visual retriever (ColQwen2). It treats every page as an image, extracts visual features via patches, and retrieves relevant pages based on their visual content. By preserving layout and formatting, the system can extract information that text parsers typically ignore or corrupt.

## Key Features

* Visual Page Indexing: Converts PDF pages to images and generates patch-level vector embeddings using ColQwen2.
* Memory Optimization: Implements Qdrant Binary Quantization (1-bit vector representations) to compress the database memory footprint by up to 97% for large document scales.
* Structured Reasoning: Uses LangGraph to orchestrate a retrieve-then-answer state machine.
* Visual Citations: Calls Gemini 3.5 Flash to retrieve bounding boxes indicating where the evidence is located, and automatically annotates the source page with a bounding box.
* Streamlit Interface: A simple web UI for uploading PDFs, monitoring ingestion status, and chatting.

## Technical Specifications

* Embedding Model: vidore/colqwen2-v1.0 (runs on CUDA-enabled GPU or falls back to CPU)
* Vector Database: Qdrant (running in a Docker container, port 6333)
* Language Model: gemini-3.5-flash 

## Setup and Installation

### 1. Prerequisites
Ensure you have Docker, Python, Poppler, and uv installed on your machine.

### 2. Environment Configuration
Create a .env file in the root of the project with your API key:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Run Qdrant Vector Database
Launch the Qdrant container with local directory mapping:
```powershell
docker run -p 6333:6333 -p 6334:6334 -v "${PWD}/qdrant_storage:/qdrant/storage:z" qdrant/qdrant
```

### 4. Run the Streamlit Application
Install dependencies and run the server:
```powershell
uv run streamlit run app.py
```
Open http://localhost:8501 in your browser.

### 5. Running Direct Command-Line Interface (CLI) Tests
You can ingest and query directly from the terminal:

* **Generate a Sample PDF**: To test immediately with a sample document containing pure pixels (no text layer, only drawn tables and charts), generate the sample PDF:
  ```powershell
  uv run python scripts/make_sample_pdf.py
  ```
  This creates `sales_report.pdf` in the `pdfs/` directory.

* **Ingest PDFs**: Index the generated PDF (or your own PDFs):
  ```powershell
  uv run python -m src.ingest
  ```

* **Query the System**: Run `main.py` with your question:
  ```powershell
  uv run python main.py "Which region had the highest revenue?"
  ```

