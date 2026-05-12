import pytest
from src.embedder import chunk_embedder, index_builder
from src.retrieval import dense_retriever, bm25_retriever, hybrid_retriever, reranker


@pytest.fixture
def sample_chunks():
    """
    Returns fake 20 chunks for testing.
    """
    return [
        {"text": "Chunk 0: Legal definition of pets.", "chunk_index": 0, "source": "doc_a.pdf"},
        {"text": "Chunk 1: Rules for animal shelters.", "chunk_index": 1, "source": "doc_a.pdf"},
        {"text": "Chunk 2: Penalties for cruelty.", "chunk_index": 2, "source": "doc_a.pdf"},
        {"text": "Chunk 3: Wildlife protection basics.", "chunk_index": 3, "source": "doc_b.pdf"},
        {"text": "Chunk 4: Endangered species list.", "chunk_index": 4, "source": "doc_b.pdf"},
        {"text": "Chunk 5: Bird sanctuary regulations.", "chunk_index": 5, "source": "doc_b.pdf"},
        {"text": "Chunk 6: Marine life protection.", "chunk_index": 6, "source": "doc_c.pdf"},
        {"text": "Chunk 7: Fishing laws in coastal areas.", "chunk_index": 7, "source": "doc_c.pdf"},
        {"text": "Chunk 8: Import/Export of exotic animals.", "chunk_index": 8, "source": "doc_c.pdf"},
        {"text": "Chunk 9: Veterinary standards.", "chunk_index": 9, "source": "doc_d.pdf"},
        {"text": "Chunk 10: Animal testing laws.", "chunk_index": 10, "source": "doc_d.pdf"},
        {"text": "Chunk 11: Transportation of livestock.", "chunk_index": 11, "source": "doc_d.pdf"},
        {"text": "Chunk 12: Zoo management guidelines.", "chunk_index": 12, "source": "doc_e.pdf"},
        {"text": "Chunk 13: Stray dog sterilization programs.", "chunk_index": 13, "source": "doc_e.pdf"},
        {"text": "Chunk 14: Prevention of illegal hunting.", "chunk_index": 14, "source": "doc_e.pdf"},
        {"text": "Chunk 15: National parks boundaries.", "chunk_index": 15, "source": "doc_f.pdf"},
        {"text": "Chunk 16: Rights of elephants in captivity.", "chunk_index": 16, "source": "doc_f.pdf"},
        {"text": "Chunk 17: Forest conservation act summary.", "chunk_index": 17, "source": "doc_f.pdf"},
        {"text": "Chunk 18: Biodiversity board functions.", "chunk_index": 18, "source": "doc_g.pdf"},
        {"text": "Chunk 19: Public awareness for animal rights.", "chunk_index": 19, "source": "doc_g.pdf"}
    ]

@pytest.fixture
def sample_index(sample_chunks):
    index = index_builder(chunk_embedder(sample_chunks))
    return index

@pytest.fixture
def sample_query():
    query = "Animals"
    return query

@pytest.fixture
def sample_top_k():
    return 5


def test_dense_retriever_returns_top_k(sample_chunks, sample_index, sample_query, sample_top_k):

    dense_res = dense_retriever(sample_query, sample_index, sample_chunks, top_k = sample_top_k)

    assert len(dense_res) == sample_top_k


def test_bm25_retriever_sorted_scores(sample_query, sample_chunks, sample_top_k):

    sparse_res = bm25_retriever(sample_query, sample_chunks, top_k = sample_top_k)
    order = [r["score"] for r in sparse_res]
    
    assert order == sorted(order, reverse = True)


def test_hybrid_retriever_no_duplicates(sample_query,sample_index, sample_chunks, sample_top_k):
    
    hybrid_res = hybrid_retriever(sample_query, sample_index, sample_chunks, top_k= sample_top_k)
    non_dup = [r["chunk_index"] for r in hybrid_res]

    assert len(non_dup) == len(set(non_dup))

def test_reranker_output_length(sample_query,sample_index, sample_chunks, sample_top_k):

    hybrid_res = hybrid_retriever(sample_query, sample_index, sample_chunks, top_k= sample_top_k)
    reranker_res = reranker(sample_query, hybrid_res, top_k = sample_top_k)

    assert len(reranker_res) == sample_top_k

def test_reranker_changes_order(sample_query,sample_index, sample_chunks, sample_top_k):

    hybrid_res = hybrid_retriever(sample_query, sample_index, sample_chunks, top_k= sample_top_k)
    reranker_res = reranker(sample_query, hybrid_res, top_k = sample_top_k)

    hyb_ord = [r["chunk_index"] for r in hybrid_res]
    re_ord = [r["chunk_index"] for r in reranker_res]

    assert hyb_ord != re_ord
    
