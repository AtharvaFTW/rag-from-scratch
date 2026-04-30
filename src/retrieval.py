import faiss
from src.embedder import query_embedder
import numpy as np
from rank_bm25 import BM25Okapi

def dense_retriever(query:str, index:faiss.IndexFlatL2, chunks:list[dict], top_k:int = 20) -> list[dict]:
    """
    The dense_retriever finds the top_k chunks for the query using Faiss Index L2.

    Args:
        query : The query string
        index : Index created with faiss
        chunks : The chunks of the corpus
        top_k : Number of most relevant chunks to retrieve 
    """
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
    """
    The bm25_retriever uses lexical search where the query is compared against
    a collection of text documents by directly matching the words in the query.

    Args:
        query : The query string
        chunks : Chunks of the corpus
        top_k : Number of most relevant chunks to retrieve
    """

    corpus = [c["text"] for c in chunks]
    tokenized_corpus = [c.split(" ") for c in corpus]


    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = query.split(" ")
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k] # we revert the indices to descending and return the top_k

    res = []

    for idx in top_indices:
        target = chunks[idx]
        
        chunk = {
            "text" : target["text"],
            "source" : target["source"],
            "score" : scores[idx]
        }   
        res.append(chunk)
        
    return res

def hybrid_retriever(query:str, index:faiss.IndexFlatL2, chunks:list[dict], top_k:int= 20) -> list[dict]:
    pass


def reranker(query:str, candidates:list[dict], top_k:int = 5) -> list[dict]:
    pass


if __name__ == "__main__":
    from src.embedder import index_loader
    from src.data_pipeline import chunks_loader
    from pathlib import Path

    chunks = chunks_loader(Path("data/chunks.json"))

    results = bm25_retriever(query = "penelties for animal cruelty", chunks = chunks, top_k = 5)

    assert len(results) == 5
    assert all("text" in r and "source" in r and "score" in r for r in results)
    for r in results:
        print(r["score"], r["source"], r["text"][:80])