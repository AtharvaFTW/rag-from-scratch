from pathlib import Path
from src.embedder import index_loader
from src.data_pipeline import chunks_loader
from src.retrieval import hybrid_retriever,reranker
from src.generator import generate
from ragas import evaluate, RunConfig
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import GoogleEmbeddings, LangchainEmbeddingsWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from datasets import Dataset
from dotenv import load_dotenv
import faiss
import json
import os


load_dotenv()

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
EVAL_LLM = os.environ.get("EVAL_LLM")

def eval_dataset_builder(eval_path: Path, chunks_path: Path, index_path: Path) -> list[dict]:
    with open(eval_path,"r") as f:
        eval_set = json.load(f)

    chunks = chunks_loader(Path(chunks_path))
    loaded_index = index_loader(Path(index_path))

    res = []
    
    for es in eval_set:
        query = es["question"]
        hybrid_results = hybrid_retriever(query, loaded_index, chunks, 10)
        reranked_results = reranker(query, hybrid_results, 5)
        answer, context = generate(query ,reranked_results)

        dic = {
            "question" : query,
            "answer" : answer,
            "contexts" : context,
            "ground_truth" : es["ground_truth"]
        }

        res.append(dic)
    return res

def run_ragas(eval_data: list[dict]) -> dict:

    llm  = LangchainLLMWrapper(ChatGoogleGenerativeAI(
        model = EVAL_LLM,
        google_api_key = GOOGLE_API_KEY,
        temperature = 0.3,
    ))

    embeddings =GoogleEmbeddings(model = "gemini-embeddings-2-preview",
                                task_type = "retrieval_document")
                                            

    
    eval_dataset = Dataset.from_list(eval_data)

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall ]

    run_config = RunConfig(
        timeout = 120,
        max_retries = 3,
        max_wait = 60,
        max_workers = 2
    )
    
    return evaluate(eval_dataset,
                    metrics = metrics,
                    llm = llm,
                    embeddings = embeddings,
                    run_config = run_config,
                    raise_exceptions = False)


if __name__ == "__main__":
    import torch
    torch.set_num_threads(1)

    eval_path = Path(r"evaluation\eval_set.json")
    chunks_path = Path(r"data\chunks.json")
    index_path = Path(r"data\index.faiss")
    dataset_path = Path(r"evaluation\dataset.json")

    if not dataset_path.exists():
        dataset = eval_dataset_builder(eval_path, chunks_path, index_path)
        with open(dataset_path, 'w',encoding = 'utf-8') as f:
            json.dump(dataset, f, indent = 4)

    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    ragas = run_ragas(dataset)
    print(ragas)