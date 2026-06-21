import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# pyrefly: ignore [missing-import]
from fastapi import FastAPI
# pyrefly: ignore [missing-import]
from fastapi.middleware.cors import CORSMiddleware
# pyrefly: ignore [missing-import]
from langserve import add_routes
from src.base.llm_model import LLMModel
from src.rag.main import build_rag_chain, InputQA, OutputQA

llm_instance = LLMModel(temperature = 0.9)
llm = llm_instance.create_llm()

genai_docs = "./data_source/generative_ai"

rag_chain = build_rag_chain(llm, genai_docs, "pdf")

#------ App - FastAPI ------
app = FastAPI(
    title = "Langchain Server",
    version = "1.0.0",
    description = "A simple api server using Langchain's Runnable Interfaces"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
    expose_headers = ["*"]
)

#------ Routes - FastAPI ------
@app.get("/check")
async def check():
    return {"status": "ok"}

@app.post("/ask")
async def ask_llm(input: InputQA):
    result = rag_chain.invoke({"question": input.question})
    return OutputQA(answer = result.content)

#------Langserve Routes - Playground -------
add_routes(app, 
            rag_chain,
            playground_type="default",
            path="/generative_ai")



