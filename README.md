# Retrieval-Augmented Generation Pipeline with Full Evaluation Suite

**The Corpus consists of:**

- The Prevention of Cruelty to Animals Act, 1960

- The Wildlife Protection Act, 1972


**Why?**

- The narrow domain, verifiable facts and real-world relevance and the punishments/section numbers make ground truth questions very testable.

- The documents contain sections like definitions, penalties, offences, exemptions which makes it suitable for writing ground truth questions on animal welfare.

## Stack

Python · FAISS · SentenceTransformers · Llama 3.1 8b via Ollama(generation) ·gemma4:31b-cloud via Ollama Cloud(evaluation)· RAGAS · FastAPI · Docker · Streamlit

## Project Phases

- [x] Phase 1 - Corpus Selection
- [x] Phase 2 - Data Pipeline
- [x] Phase 3 - Embedding + Indexing
- [x] Phase 4 - Retrieval Layer
- [x] Phase 5 - Generation
- [x] Phase 6 - Evaluation Set  
- [x] Phase 7 - RAGAS Evaluation
- [x] Phase 8 - A/B Testing
- [ ] Phase 9 - FastAPI + Streamlit &larr;  _Currently here_
- [ ] Phase 10 - Pytest
- [ ] Phase 11 - Docker

## Architecture

_To be added after the Phase 9 completion_

## Key Findings

### Phase 8 Hypotheses (pre-evaluation predictions)

1. **Chunk boundary artifacts** - The current character-based chunks splits the word, degrading the retrieval quality. 
    - Hypothesis : Larger chunk sizes (512 vs 1024) or word/sentence-based chunks splits will produce more complete   sentences and improve the RAGAS context precision.

2. **Cross-document retrieval imbalance** - Queries explicitly mentioning "Wildlife Protection Act " surface Prevention of Curelty Act chunks in top 5 results due to share vocab. 
    - Hypothesis : BM25 will outperform dense retrieval on cross-document queries because "Wildlife" is a distinctive keyword.


### Evaluation Infrastructure

Local inference via Ollama on Google Colab T4 GPU tunneled through ngrok proved unreliable for RAGAS evaluation. Primary failure modes:
- T4 GPU throughput was too slow (~230s/job) causing systematic timeouts
- Small models (`Qwen3` 4B, 8B) failed to follow RAGAS's structured JSON output format, producing plain text answers instead of parseable evaluation scores

**Solution**: Ollama Cloud API with `gemma4:31b-cloud` resolved both the issues. Pros like strong instruction following, no GPU setup required, generous free tier(~5M tokens/week). Total evaluation cost: 5.6% of weekly quota for 30 questions across 4 metrics (Faithfulness, Answer Relevancy, Context Precision, Context Recall)


### Retrieval Quality

Dense Retrieval outperformed both sparse and hybrid on this corpus.

Evaluation Metrics:
- Context Precision: 0.66
- Context Recall: 0.72 

Metrics were hightest with dense retrieval + reranking. Hybrid (RRF fusion) underperformed expectations, the combination of dense and BM25 scores hurt further rather than help narrowing the legal domain.

Reranking had a significant impact, removing it dropped context precision from 0.46 -> 0.28, confirming the cross-encoder adds meaningful signal.


### Generator Quality

Faithfulness peaked with smaller chunks (256 tokens: 0.88) signaling focused context reduces hallucination.
Answer Relevancy was highest with larger chunks (1024 tokens: 0.57) proving context is directly proportional to completeness.

To achieve the best of both worlds we set the chunk size to 512

## Evaluation Results

### A/B Testing - RAGAS Scores

| Config | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|--------|-------------|-----------------|-------------------|----------------|
| 512 chunks, hybrid, reranked (baseline) | 0.6345 | 0.4292 | 0.4588 | 0.4667 |
| 256 chunks, hybrid, reranked | 0.8753 | 0.3574 | 0.3621 | 0.3833 |
| 1024 chunks, hybrid, reranked | 0.4568 | 0.5733 | 0.3361 | 0.4000 |
| **512 chunks, dense only, reranked 🏆** | **0.8489** | **0.5397** | **0.6575** | **0.7167** |
| 512 chunks, sparse only, reranked | 0.8201 | 0.6307 | 0.6631 | 0.6167 |
| 512 chunks, hybrid, no reranking | 0.4160 | 0.4492 | 0.2810 | 0.5000 |



## Running the Project

_To be added after the Phase 10 completion_



**Updated on** - 12 May 2026