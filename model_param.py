import transformers
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline
)

class CFG:
    DEBUG = False
    model_name = 'microsoft/Phi-3-mini-128k-instruct'
    temperature = 0.4
    top_p = 0.90
    repetition_penalty = 1.15
    max_len = 8192
    max_new_tokens = 512
    split_chunk_size = 400
    split_overlap = 200
    embeddings_model_repo = 'BAAI/bge-base-en-v1.5'
    k = 6
    PDFs_path = './New Folder With Items'
    Embeddings_path =  './papers-vectordb/faiss_index_papers'
    Output_folder = './papers-vectordb'


def build_model(model_repo = CFG.model_name):

    print('\nDownloading model: ', model_repo, '\n\n')
    tokenizer = AutoTokenizer.from_pretrained(model_repo)
    model = AutoModelForCausalLM.from_pretrained(
        model_repo,
        device_map = 'auto',
        low_cpu_mem_usage = True,
        trust_remote_code = True,
    )

    return tokenizer, model