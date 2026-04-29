from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def chunk_embedder(chunks: list[dict]) -> np.ndarray:
    """
    The chunk_embedder takes the chunks of the corpus and converts
    them in to the embeddings with the help of numpy.

    Args:
        chunks: The list of chunks_dict with text, chunk_index and source
    """

    text_chunks = [chunk["text"] for chunk in chunks] # The model doesn't take the dict as is.
    embeddings = MODEL.encode(text_chunks)
    
    return embeddings.astype(np.float32)

def query_embedder(query: str) -> np.ndarray:
    """
    query_embedder handles embedding for a single query.

    Args:
        query: The query string
    """
    query_embed = MODEL.encode(query)
    query_embed = np.expand_dims(query_embed, axis = 0)
    return query_embed.astype(np.float32)



def index_builder(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """
    The index_builder takes the embeddings and builds the vector database. 
    Here we are using the brute-force KNN approach to maximize the accuracy.

    Args:
        embeddings: The vector representation of the chunks
    """
    dimensions = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimensions)

    index.add(embeddings)

    return index


def index_saver(index: faiss.IndexFlatL2, path: Path) -> None:
    """
    The index_saver stores the index at a given path

    Args:
        index: the index to store
        path: the location to store the index
    """
    faiss.write_index(index, str(path))


def index_loader(path: Path) -> faiss.IndexFlatL2:
    """
    The index_loader returns the index from the given path.
    Same index is onboarded on to the RAM for querying.

    Args:
        path: the location of index
    """
    index = faiss.read_index(str(path))
    return index


if __name__ == "__main__":
    from src.data_pipeline import corpus_builder
    from pathlib import Path

    chunks = corpus_builder(Path("data/raw"), chunk_size = 512, overlap_size = 50)
    embeddings = chunk_embedder(chunks)
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Dtype : {embeddings.dtype}")

    index = index_builder(embeddings)
    print(f"Index size: {index.ntotal}")

    index_path = Path("data/index.faiss")

    index_saver(index, index_path)
    loaded_index = index_loader(index_path)
    print(f"Loaded index size: {loaded_index.ntotal}")

    assert embeddings.dtype == np.float32
    assert index.ntotal == len(chunks)
    assert loaded_index.ntotal == index.ntotal

