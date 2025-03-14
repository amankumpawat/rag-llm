# RAG-LLM

RAG-LLM is a Python tool that demonstrates retrieval-augmented generation (RAG) with large language models (LLMs) to provide more accurate, context-aware answers. It combines an LLM (via OpenAI's API) with your custom data: the system builds a knowledge index from your documents, retrieves relevant information for a given query, and then generates a response using the LLM grounded in those retrieved facts. This approach enables interactive question-answering on domain-specific content, reducing hallucinations and improving answer reliability.

## Features

- **Custom Knowledge Integration**: Easily incorporate your own text documents or dataset into the LLM’s knowledge. The tool indexes your content so that the LLM can use it when answering questions, ensuring responses are based on up-to-date and domain-specific information.
- **Retrieval-Augmented Generation**: Utilizes RAG to fetch relevant context before generating answers. The system performs semantic search over your data (using text embeddings) to find the most pertinent chunks for each query, then feeds those to the LLM for answer generation.
- **OpenAI API Powered**: Leverages OpenAI’s GPT models for both embedding computation and language generation. You get the benefit of powerful language understanding and generation, with your data guiding the output. (An OpenAI API key is required.)
- **CSV-Based Knowledge Base**: Stores the embeddings index of your documents in a CSV file for persistence. On subsequent runs, the tool can load this file to avoid re-processing documents, saving time and API calls.
- **Simple Interface**: Designed as a command-line application. Run the script, enter your question, and receive an answer. The straightforward workflow makes it easy to test RAG with minimal setup.

## Components

This repository is organized into a few key components:

- **Main Script**: The entry point (e.g. `rag-llm-icd11.py`) orchestrates the RAG pipeline. It loads configuration, processes the document dataset, handles user queries, and prints out answers.
- **Document Processing Module**: Reads your source documents (from a file or directory) and breaks them into chunks suitable for embedding.
- **Embedding and Indexing**: Generates vector embeddings for each document chunk (using OpenAI embeddings) and builds a similarity index stored in a CSV file.
- **Retrieval Mechanism**: Searches the vector index to find the closest matching document pieces when a query is asked.
- **LLM QA Module**: Composes a prompt for the LLM using the retrieved context and the user’s question, then calls the OpenAI API to generate an answer.
- **Configuration File/Variables**: Handles API keys and file paths via a config file (such as a `.env` file or Python config).

## Installation

### Prerequisites

- Python 3.8 or higher
- OpenAI API key 
- Pip (Python package installer)
- Internet connection (for OpenAI API calls)

### Additional Dependencies

To enable advanced document processing and model support, install the following dependencies into a virtual environment:

```bash
pip install -q langchain langchain-community langchain-openai chromadb pypdf pdfplumber transformers torch sentencepiece accelerate line_profiler
```

For Jupyter Notebook users, enable the `line_profiler` extension:

```bash
jupyter nbextension enable --py line_profiler
```

## Configuration

Before running RAG-LLM, configure your OpenAI API key and data source:

```bash
OPENAI_API_KEY=your-openai-key-here
```

Set the data path:

```bash
DATA_PATH="./docs"
```

## Usage

Run the program via the command line:

```bash
rag-llm
```

or directly using Python:

```bash
python -m rag_llm
```

or if cloned from source:

```bash
python main.py
```

## Output

- **Embedded Knowledge Base (CSV)**: Stores document embeddings for efficient retrieval.
- **Answer Logs (optional)**: Logs user queries and responses for later review.
- **Console Output**: Displays progress updates and final responses.

## Troubleshooting

- **API Key Error**: Ensure `OPENAI_API_KEY` is set correctly.
- **No Data Found**: Verify the `DATA_PATH` is correctly configured and accessible.
- **Slow Performance**: First runs require indexing; subsequent runs use cached embeddings.
- **OpenAI API Limits**: Reduce request frequency or upgrade OpenAI subscription.
- **Environment Issues**: Use a virtual environment to isolate dependencies.
- **Rebuilding Index**: Delete the existing embeddings CSV file and rerun the tool.

## Technical Details

### How RAG Works

1. **Document Ingestion & Embedding**: Documents are split into chunks, converted into vector embeddings using OpenAI’s `text-embedding-ada-002` model, and stored in a CSV.
2. **Query Processing & Retrieval**: Queries are transformed into embeddings and matched against stored document vectors using similarity search.
3. **Augmented Answer Generation**: The retrieved chunks are included in the LLM’s prompt, improving factual accuracy.
4. **Result Output**: The LLM-generated answer is displayed to the user.

This approach ensures that the model references external knowledge before producing an answer, reducing misinformation and improving response quality.

## Disclaimer

RAG-LLM is an open-source project developed by independent contributors. It is not officially affiliated with or endorsed by OpenAI. Names like OpenAI and GPT are trademarks of their respective owners; usage here is only to describe the tool’s functionality (i.e., integration with the OpenAI API). Use this project at your own discretion. Always ensure compliance with OpenAI’s terms of service when using the API, and be mindful of the costs and data privacy implications of sending content to third-party services. The maintainers of RAG-LLM provide this tool as-is, without warranty, for educational and research purposes.

---

Enjoy using RAG-LLM! 🚀
