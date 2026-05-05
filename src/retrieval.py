import faiss
from src.embedder import query_embedder
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

CE_MODEL = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

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
            "chunk_index" : target_chunk["chunk_index"],
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
        target_chunk = chunks[idx]
        
        chunk = {
            "chunk_index" : target_chunk["chunk_index"],
            "text" : target_chunk["text"],
            "source" : target_chunk["source"],
            "score" : scores[idx]
        }   
        res.append(chunk)
        
    return res

def hybrid_retriever(query:str, index:faiss.IndexFlatL2, chunks:list[dict], top_k:int= 20) -> list[dict]:
    """
    The hybrid_retriever uses the RRF (Reciprocal Rank Fusion). 
    We us ranks to avoid dominance of bm25 over the dense retriever score.

    Args:
        query : The query string
        index : Index created with faiss
        chunks : The chunks of the corpus
        top_k : Number of most relevant chunks to retrieve 
    """
    rrf_scores = {}
    chunk_map = {}

    dense_results = dense_retriever(query, index, chunks, top_k = len(chunks))
    bm25_results = bm25_retriever(query, chunks, top_k = len(chunks))

    for rank, result in enumerate(dense_results, start = 1):
        chunk_index = result["chunk_index"]
        rrf_scores[chunk_index] = rrf_scores.get(chunk_index, 0) + 1 / (rank + 60) # 60 is a tuning constant from the orginal RRF paper, not related to how many results you want.
        chunk_map[chunk_index] = result

    for rank, result in enumerate(bm25_results, start =1):
        chunk_index = result["chunk_index"]
        rrf_scores[chunk_index] = rrf_scores.get(chunk_index, 0) + 1/ (rank + 60) # 60 is a tuning constant from the orginal RRF paper, not related to how many results you want.
        chunk_map[chunk_index] = result

    sorted_indices = sorted(rrf_scores.keys(), key = lambda idx: rrf_scores[idx], reverse = True)

    res = []

    for idx in sorted_indices[:top_k]:
        target_chunk = chunks[idx]

        chunk = {
            "chunk_index" : target_chunk["chunk_index"],
            "text" : target_chunk["text"],
            "source" : target_chunk["source"],
            "score" : rrf_scores[idx]
        }

        res.append(chunk)
    
    return res



def reranker(query:str, candidates:list[dict], top_k:int = 5) -> list[dict]:
    """
    The reranker uses the cross encoder which passes the list[(query + chunk)] for all the available chunks.
    This enables the model to do full attention between query word and every chunk word unlike the bi-encoders where
    the query and chunks are encoded seperately.

    Args:
        query : The query string
        candidates : The top chunks retrieved chunks from the hybrid (dense + bm25) retriever.
        top_k : Number of most relevant chunks to retrieve

    """
    tups = [(query, c["text"]) for c in candidates]

    scores = CE_MODEL.predict(tups)
    top_scores = np.argsort(scores)[::-1][:top_k]
    
    res = []

    for idx in top_scores:
        target_chunk = candidates[idx]
        
        chunk = {
            "chunk_index" : target_chunk["chunk_index"],
            "text" : target_chunk["text"],
            "source" : target_chunk["source"],
            "score" : scores[idx]
        }

        res.append(chunk)

    return res 


if __name__ == "__main__":
    from src.embedder import index_loader
    from src.data_pipeline import chunks_loader
    from pathlib import Path

    chunks = chunks_loader(Path("data/chunks.json"))
    loaded_index = faiss.read_index("data/index.faiss")
    hybrid_results = hybrid_retriever(query = "penelties for animal cruelty", index = loaded_index, chunks = chunks, top_k = 5)

    reranked_results = reranker("penalities for animal cruelty", hybrid_results, top_k = 5)

    assert len(reranked_results) == 5
    assert all("text" in r and "source" in r and "score" in r for r in reranked_results)
    print(f"#########################################")
    for r in reranked_results:
        print(f"Score : {r["score"]:.2f}")
        print ("--")
        print(f"Text : {r["text"][:200]}")
        print ("--")
        print(f"Source : {r["source"]}")
        print(f"#########################################")