# Retrieval-Augmented Generation Pipeline with Full Evaluation Suite

**The Corpus consists of:**

- The Prevention of Cruelty to Animals Act, 1960

- The Wildlife Protection Act, 1972


**Why?**

- The narrow domain, verifiable facts and real-world relevance and the punishments/section numbers make ground truth questions very testable.

- The documents contain sections like definitions, penalties, offences, exemptions which makes it suitable for writing ground truth questions on animal welfare.

## Stack

Python · FAISS · SentenceTransformers · Llama 3.1 8b via Ollama · RAGAS · FastAPI · Docker · Streamlit

## Project Phases

- [x] Phase 1 - Corpus Selection
- [x] Phase 2 - Data Pipeline
- [x] Phase 3 - Embedding + Indexing
- [x] Phase 4 - Retrieval Layer
- [x] Phase 5 - Generation
- [ ] Phase 6 - Evaluation Set  &larr;  _Currently here_
- [ ] Phase 7 - RAGAS Evaluation
- [ ] Phase 8 - A/B Testing
- [ ] Phase 9 - FastAPI + Streamlit
- [ ] Phase 10 - Docker

## Architecture

_To be added after the Phase 9 completion_

## Key Findings
### Phase 8 Hypotheses (pre-evaluation predictions)

1. **Chunk boundary arifacts** - The current character-based chunks splits the word, degrading the retrieval quality. 
    - Hypothesis : Larger chunk sizes (512 vs 1024) or word/sentence-based chunks splits will produce more complete   sentences and improve the RAGAS context precision.

2. **Cross-document retrieval imbalance** - Queries explicitly mentioning "Wildlife Protection Act " surface Prevention of Curelty Act chunks in top 5 results due to share vocab. 
    - Hypothesis : BM25 will outperform dense retrieval on cross-document queries because "Wildlife" is a distinctive keyword.

### A/B Test Results

_To be added after the Phase 8 completion_


## Running the Project

_To be added after the Phase 10 completion_



**Updated on** - 5 May 2026