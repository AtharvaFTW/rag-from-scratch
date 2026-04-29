import faiss
from src.embedder import query_embedder
import numpy as np

def dense_retriever(query:str, index:faiss.IndexFlatL2, chunks:list[dict], top_k:int = 20) -> list[dict]:
    query_embedding = query_embedder(query)

    distances, indices = index.search(query_embedding, top_k)
    distances, indices = distances.flatten(), indices.flatten()

    res = []

    for distance, idx in zip(distances, indices):

        target_chunk = chunks[idx]
        score = 1 / (1 + distance)

        chunk = {
            "text" : target_chunk["text"],
            "source": target_chunk["source"],
            "score" : score
        }

        res.append(chunk)
    
    return res

        
def bm25_retriever(query:str, chunks:list[dict], top_k:int= 20) -> list[dict]:
    pass


def hybrid_retriever(query:str, index:faiss.IndexFlatL2, chunks:list[dict], top_k:int= 20) -> list[dict]:
    pass


def reranker(query:str, candidates:list[dict], top_k:int = 5) -> list[dict]:
    pass


if __name__ == "__main__":
    from src.embedder import index_loader
    from src.data_pipeline import chunks_loader
    from pathlib import Path

    index = index_loader(Path("data/index.faiss"))
    chunks = chunks_loader(Path("data/chunks.json"))

    results = dense_retriever("what are the penelties for animal cruelty", index, chunks, top_k = 5)

    assert len(results) == 5
    assert all("text" in r and "source" in r and "score" in r for r in results)
    for r in results:
        print(r["score"], r["source"], r["text"][:80])