from ollama import Client
from dotenv import load_dotenv
from src.llama_prompt import prompt
import os
load_dotenv()

HOST = os.environ.get("OLLAMA_NGROK_TUNNEL")

client = Client(host = HOST,  headers={'ngrok-skip-browser-warning': 'true'})


def generate(query: str, chunks: list[dict]) -> str :
    """
    This function is the actual brain of the system. It generates a human readable output for the query.

    Args:
        query : The query string.
        context : The relevant chunks from the documents.
    """

    context = [f"Text - {chunk['text']},Source - {chunk['source']}" for chunk in chunks]
    context_str = "\n\n".join(context)

    filled_prompt = prompt.format(query = query, context = context_str)

    response = client.chat(model = "llama3.1:8b" , messages = [{
        'role': 'user', 'content': f'{filled_prompt}'
    }])

    return response['message']['content']

if __name__ == "__main__":
    from src.embedder import index_loader
    from src.data_pipeline import chunks_loader
    from src.retrieval import hybrid_retriever, reranker
    from pathlib import Path
    from collections import Counter

    chunks = chunks_loader(Path("data/chunks.json"))
    index = index_loader(Path("data/index.faiss"))

    query = "What animals are protected under the Wildlife Protection Act"
    print(f"Question: {query}")
    retrieved = hybrid_retriever(query, index, chunks, top_k = 20)
    reranked = reranker(query, retrieved, top_k = 5)
    answer = generate(query, reranked)
    print(f"Response: {answer}")
