# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field

from src.rag.file_loader import Loader
from src.rag.offline_rag import Offline_RAG
from src.rag.vectorstore import VectorDB

class InputQA(BaseModel): #inherits BaseModel from pydantic for data validation
    question: str = Field(..., title = "Question to ask the model")

class OutputQA(BaseModel):
    answer: str = Field(..., title = "Answer from the model")

def build_rag_chain(llm, data_dir, file_type):
    """Build a RAG chain from the given LLM, data directory, and data type
    
    Args:
        llm: LLM model
        data_dir: Directory containing the documents
        data_type: Type of data
    
    Returns:
        RAG chain
    """
    loader = Loader(file_type = file_type)
    documents = loader.load_dir(data_dir)
    vectorstore = VectorDB(documents)
    retriever = vectorstore.get_retriever()
    rag_chain = Offline_RAG(llm).get_chain(retriever)
    return rag_chain