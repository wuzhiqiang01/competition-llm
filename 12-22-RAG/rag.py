import gc
import torch
import ctypes
import numpy as np
import pandas as pd
from tqdm import tqdm
from time import time
import torch.functional as F
from torch.utils.data import DataLoader

# FOR RAG
from datasets import Dataset

# FOR LLM
from transformers import AutoModel, AutoTokenizer


def clean_memory():
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)
    torch.cuda.empty_cache()


class SentenceTransformer:
    def __init__(self, checkpoint, device="cuda:0"):
        self.device = device
        self.checkpoint = checkpoint
        self.model = AutoModel.from_pretrained(checkpoint).to(self.device).half()
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    def transform(self, batch):
        tokens = self.tokenizer(batch["text"], truncation=True, padding=True, return_tensors="pt", max_length=MAX_SEQ_LEN)
        return tokens.to(self.device)  

    def get_dataloader(self, sentences, batch_size=32):
        sentences = ["Represent this sentence for searching relevant passages: " + x for x in sentences]
        dataset = Dataset.from_dict({"text": sentences})
        dataset.set_transform(self.transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return dataloader

    def encode(self, sentences, show_progress_bar=False, batch_size=32):
        dataloader = self.get_dataloader(sentences, batch_size=batch_size)
        pbar = tqdm(dataloader) if show_progress_bar else dataloader

        embeddings = []
        for batch in pbar:
            with torch.no_grad():
                e = self.model(**batch).pooler_output
                e = F.normalize(e, p=2, dim=1)
                embeddings.append(e.detach().cpu().numpy())
        embeddings = np.concatenate(embeddings, axis=0)
        return embeddings

class CFG:
    MODEL_PATH = "/mnt/HDD0/wuzhiqiang/competition/FT-Data-Ranker/llm-se/pretrain-model/all-MiniLM-L6-v2"

def main():
    root = "/mnt/HDD0/wuzhiqiang/dataset/kaggle-llm-science-exam"
    df = pd.read_csv(root + "/test.csv", index_col="id")
    # print(df.head())
    # Uncomment this to see results on the train set
    df = pd.read_csv(root + "/train.csv", index_col="id")
    IS_TEST_SET = True
    N_BATCHES = 1
    print(df.head())

    if IS_TEST_SET:
        # Load embedding model
        start = time()
        print(f"Starting prompt embedding, t={time() - start :.1f}s")
        model = SentenceTransformer(CFG.MODEL_PATH, device="cuda:0")

        # Get embeddings of prompts
        f = lambda row : " ".join([row["prompt"], row["A"], row["B"], row["C"], row["D"], row["E"]])
        inputs = df.apply(f, axis=1).values # better results than prompt only
        prompt_embeddings = model.encode(inputs, show_progress_bar=False)

        # Search closest sentences in the wikipedia index 
        print(f"Loading faiss index, t={time() - start :.1f}s")
        faiss_index = faiss.read_index(CFG.MODEL_PATH + '/faiss.index')
        # faiss_index = faiss.index_cpu_to_all_gpus(faiss_index) # causes OOM, and not that long on CPU

        print(f"Starting text search, t={time() - start :.1f}s")
        search_index = faiss_index.search(np.float32(prompt_embeddings), NUM_TITLES)[1]

        print(f"Starting context extraction, t={time() - start :.1f}s")
        dataset = load_from_disk("/kaggle/input/all-paraphs-parsed-expanded")
        for i in range(len(df)):
            df.loc[i, "context"] = "-" + "\n-".join([dataset[int(j)]["text"] for j in search_index[i]])

        # Free memory
        faiss_index.reset()
        del faiss_index, prompt_embeddings, model, dataset
        clean_memory()
        print(f"Context added, t={time() - start :.1f}s")

if __name__ == "__main__":
    main()