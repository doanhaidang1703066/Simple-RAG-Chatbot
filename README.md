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
```




## 🛠️ Requirements

This source code is built and optimized to run on the Ubuntu operating system with a GPU (e.g., a 24GB GPU).
You need to install the dependencies listed in `requirements.txt`:

* `torch==2.2.2`
* `transformers==4.39.3`
* `accelerate==0.28.0`
* `bitsandbytes==0.42.0`
* `huggingface-hub==0.22.2`
* `langchain==0.1.14`
* `langchain-core==0.1.43`
* `langchain-community==0.0.31`
* `pypdf==4.2.0`
* `sentence-transformers==2.6.1`
* `beautifulsoup4==4.12.3`
* `langserve[all]`
* `chromadb==0.4.24`
* `langchain-chroma==0.1.0`
* `faiss-cpu==1.8.0`
* `rapidocr-onnxruntime==1.3.16`
* `unstructured==0.13.2`
* `fastapi==0.110.1`
* `uvicorn==0.29.0`

## 🚀 Installation & Usage

**Step 1: Install Dependencies**

```bash
pip install -r requirements.txt

```

**Step 2: Prepare Data**
Run the script to automatically download scientific papers (e.g., Attention Is All You Need, Llama 2...) to the data directory:

```bash
python data_source/generative_ai/download.py

```

**Step 3: Start the API Server**
Open the terminal, navigate to the project root directory (`rag_langchain/`), and run the uvicorn command to start the FastAPI Server:

```bash
uvicorn src.app:app --host "0.0.0.0" --port 5000 --reload

```

*(If port 5000 is in use, you can change it to another available port)*

**Step 4: Use the Chatbot**

* **System Status:** Check at `http://0.0.0.0:5000/check`.
* **Playground Interface:** Access the URL provided by LangServe to interact directly with the model at `http://0.0.0.0:5000/generative_ai/playground`.
* **API Call via cURL:** You can POST JSON-formatted questions to the `/generative_ai` endpoint.
