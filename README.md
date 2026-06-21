# Simple RAG System for PDF Question Answering

This project builds a fundamental RAG (Retrieval Augmented Generation) application designed to answer academic questions. The system leverages scientific papers in PDF format as its knowledge source and uses the LangChain framework to connect the language model with the data.

## 🧠 Techniques Used
This project applies the following natural language processing techniques and tools:
* **Core Framework:** LangChain, specifically designed for deploying LLMs in real-world applications.
* **Language Model (LLM):** Qwen2.5-3B from Hugging Face.
* **Hardware Optimization (Quantization):** Applies 4-bit quantization using `BitsAndBytesConfig` (with the `nf4` standard) to perform inference on low-VRAM GPUs.
* **Document Processing:** Uses `PyPDFLoader` to read PDF files and `RecursiveCharacterTextSplitter` for text chunking.
* **Embedding Model:** Uses `HuggingFaceEmbeddings`.
* **Vector Database:** Utilizes ChromaDB as the default vector store and FAISS to support similarity search.
* **API Deployment:** Uses FastAPI combined with LangServe to initialize the API and provide a Playground interface.

## 📂 Project Structure
The source code is organized within the `rag_langchain/` directory as follows:

```text
rag_langchain/
├── data_source/
│   └── generative_ai/
│       └── download.py             # Script to automatically download scientific PDF papers
├── src/
│   ├── base/
│   │   └── llm_model.py            # Initializes and quantizes the LLM
│   ├── rag/
│   │   ├── file_loader.py          # Functions for loading and processing PDF files
│   │   ├── main.py                 # Initializes and connects chains (build_rag_chain)
│   │   ├── offline_rag.py          # Configures Prompt Template and Output Parser
│   │   ├── utils.py                # Helper functions to extract the model's answer
│   │   └── vectorstore.py          # Initializes the vector database (Chroma/FAISS)
│   └── app.py                      # Initializes FastAPI and LangServe routes
└── requirements.txt                # List of required dependencies


