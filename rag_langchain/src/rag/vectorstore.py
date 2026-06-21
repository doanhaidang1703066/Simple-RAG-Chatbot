# pyrefly: ignore [missing-import]
from typing import Union
# pyrefly: ignore [missing-import]
from langchain_chroma import Chroma
# pyrefly: ignore [missing-import]
from langchain_community.vectorstores import FAISS
# pyrefly: ignore [missing-import]
from langchain_community.embeddings import HuggingFaceEmbeddings

class VectorDB:
    def __init__(self, 
                documents = None,
                vector_db: Union[Chroma, FAISS] = Chroma,
                embeddings = HuggingFaceEmbeddings()) -> None:
        self.vector_db = vector_db
        self.embeddings = embeddings
        self.db = self._build_db(documents)

    def _build_db(self, documents):
        #return vector store initialized from documents and embeddings
        db = self.vector_db.from_documents(documents, self.embeddings)
        return db
    
    def get_retriever(self, 
                search_type: str = "similarity",
                search_kwargs: dict = {"k": 10}):
        #return vector store retriever
        retriever = self.db.as_retriever(search_type = search_type, search_kwargs = search_kwargs)
        return retriever


