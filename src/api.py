from fastapi import FastAPI
from src.generator import generate
from src.retrieval import dense_retriever, reranker
from src.embedder import index_loader
from src.data_pipeline import chunks_loader
from pathlib import Path
from pydantic import BaseModel

app = FastAPI()

chunks = chunks_loader(Path(r"data/chunks.json"))

index = index_loader(Path(r"data/index.faiss"))


RAGAS_SCORES = {
    "faithfulness": 0.8489,
    "answer_relevancy": 0.5397,
    "context_precision": 0.6575,
    "context_recall": 0.7167,
    "config": "512 chunks, dense retrieval, reranked"
}

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: str
    references: list[str]
    ragas_scores: dict

@app.get("/health")
def home():
    return {"status":"healthy"}

@app.post("/query")
def get_response(query: QueryRequest) -> QueryResponse:
    dense_res = dense_retriever(query.query, index, chunks, top_k = 10)
    reranked_res = reranker(query.query, dense_res, top_k = 5)
    response, references = generate(query.query, reranked_res)

    return {"response": response,
            "references": references,
            "ragas_scores": RAGAS_SCORES}