# RAG-LLM for ICD-11

RAG-LLM is a Python tool showcasing retrieval-augmented generation (RAG) with large language models (LLMs). This particular setup focuses on querying the ICD-11 (International Classification of Diseases, 11th Revision) document. It indexes the ICD-11 PDF locally and retrieves relevant sections to answer medical coding questions with higher accuracy and fewer hallucinations. However, the approach and code can be adapted for other large text files or PDFs—simply update the data source and adjust your prompt as needed. This code was optimized to Google Colab using the t4 GPU, so using other environments may need alterations.

## Features

- **ICD-11 Knowledge**: Specifically tuned to parse and index the ICD-11 PDF (or any other large medical/manual-like document). The chunked text ensures queries reference the right classification details.
- **Retrieval-Augmented Generation**: Combines an LLM with a semantic search over the ICD-11 embeddings, returning the most relevant chunks for a given query.
- **OpenAI API–Powered**: Uses OpenAI’s GPT models for both creating embeddings and generating responses. You’ll need an OpenAI API key to use this functionality.
- **Persistent Embeddings**: Stores computed document embeddings in a CSV file so you won’t have to re-embed the entire ICD-11 text on every run.
- **Generalizable**: While focused on ICD-11, this RAG workflow can be customized for other documents—simply supply your own text/PDF and tweak the prompt to reflect the new domain.

## Components

This repository is organized into a few key components:

- **Main Script**: The entry point (e.g., `rag-llm-icd11.py`) orchestrates the RAG pipeline. It loads configuration, processes the document dataset, handles user queries, and prints out answers.
- **Document Processing Module**: Reads the ICD-11 PDF (or other input documents) and breaks them into text chunks suitable for embedding.
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
export OPENAI_API_KEY="your-openai-key-here"
```

Set the data path to your ICD-11 PDF:

```bash
export DATA_PATH="./docs/icd11.pdf"
```

If using a different document, modify `DATA_PATH` accordingly.

The system will retrieve relevant ICD-11 sections, use them to inform the LLM’s response, and return a grounded answer.

## Output

RAG-LLM produces multiple outputs during operation:

- **Embedded Knowledge Base (CSV):**  
  The system generates a structured CSV file containing embeddings of the ICD-11 document. This file allows for efficient retrieval of relevant medical codes and descriptions.

- **Formatted Medical Record CSVs:**  
  Each time the code is executed, two CSV files are generated:
  
  1. **Primary Medical Record CSV:**  
     - This file contains all demographic details along with the doctor's note.
     - The following fields are included:  
       - **Record ID**  
       - **Diagnosis**  
       - **Age**  
       - **Race**  
       - **Gender**  
       - **Doctor's Note**  
     - This ensures that queries return structured, retrievable data aligned with ICD-11 classification.

  2. **Evaluation CSV:**  
     - This file contains all the information from the primary CSV plus an evaluation section.
     - The evaluation section includes:  
       - **Number of documents retrieved per query**  
       - **Content retrieved from the ICD-11 index**  
       - **Relevant pages where the information was pulled from**  
     - This provides insights into how well the system is retrieving and processing ICD-11 data for each query.

- **Console Output:**  
  Displays progress updates and final responses in real time.

These outputs ensure that the system provides both structured patient data and retrieval evaluation, making it useful for tracking and refining ICD-11-based query performance.

## Troubleshooting

- **API Key Error**: Ensure `OPENAI_API_KEY` is set correctly.
- **No Data Found**: Verify the `DATA_PATH` is correctly configured and accessible.
- **Slow Performance**: First runs require indexing; subsequent runs use cached embeddings.
- **OpenAI API Limits**: Reduce request frequency or upgrade OpenAI subscription.
- **Environment Issues**: Use a virtual environment to isolate dependencies.
- **Rebuilding Index**: Delete the existing embeddings CSV file and rerun the tool.

## Technical Details

### How RAG Works

1. **Document Ingestion & Embedding**:
   - The ICD-11 text is split into manageable chunks, converted into vector embeddings using OpenAI’s `text-embedding-ada-002` model, and stored in a CSV.
2. **Query Processing & Retrieval**:
   - Queries are transformed into embeddings and matched against stored document vectors using similarity search.
3. **Augmented Answer Generation**:
   - The retrieved chunks are included in the LLM’s prompt, improving factual accuracy.
4. **Result Output**:
   - The LLM-generated answer is displayed to the user.

This approach ensures that the model references external knowledge before producing an answer, reducing misinformation and improving response quality.

## Disclaimer

RAG-LLM is an open-source project developed by independent contributors. It is not officially affiliated with or endorsed by OpenAI. Names like OpenAI and GPT are trademarks of their respective owners; usage here is only to describe the tool’s functionality (i.e., integration with the OpenAI API). Use this project at your own discretion. Always ensure compliance with OpenAI’s terms of service when using the API, and be mindful of the costs and data privacy implications of sending content to third-party services. The maintainers of RAG-LLM provide this tool as-is, without warranty, for educational and research purposes.
