from pathlib import Path
from src.embedder import index_loader
from src.data_pipeline import chunks_loader
from src.retrieval import hybrid_retriever,reranker
from src.generator import generate
from ragas import evaluate, RunConfig
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from datasets import Dataset
from dotenv import load_dotenv
from rich.console import Console
from tqdm import tqdm
from functools import partialmethod
import pandas
import json
import os
import time

tqdm.__init__ = partialmethod(tqdm.__init__, disable = True)

load_dotenv()

OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY")
BASE_URL = os.environ.get("OLLAMA_CLOUD_URL")
EVAL_LLM = os.environ.get("EVAL_LLM")
EMBED_MODEL = os.environ.get("EMBED_MODEL")

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

    console = Console()
    SPINNER_COLOUR = "#D97757"

    llm  = LangchainLLMWrapper(ChatOllama( model = EVAL_LLM, 
                                            base_url = BASE_URL, 
                                            temperature = 0,
                                            format = "json",
                                            client_kwargs = {"headers" : {"Authorization": "Bearer " + OLLAMA_API_KEY}}
    ))

    embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name = EMBED_MODEL))

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    run_config = RunConfig(
        timeout = 300,
        max_retries = 3,
        max_wait = 60,
        max_workers = 1
    )
    
    res = []

    for i, sample in enumerate(eval_data):
        with console.status(f"[{SPINNER_COLOUR}] Evaluating sample {i+1}/{len(eval_data)}", spinner ="star", spinner_style = SPINNER_COLOUR):
        
            dataset = Dataset.from_list([sample])
            result = evaluate(dataset, metrics = metrics, embeddings = embeddings,llm = llm, run_config = run_config, raise_exceptions = False, show_progress = False)
            res.append(result)
            time.sleep(10)

    console.print(f"[bold green] Evaluation Completed Successfully")
    return res

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
    with open("evaluation/results.json", "w" ,encoding = "utf-8") as f:
        json.dump([r.to_pandas().to_dict() for r in ragas], f, indent = 4)
    print(ragas)