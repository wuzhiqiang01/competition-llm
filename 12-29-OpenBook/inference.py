import gc
import torch
import pandas as pd
from faiss import read_index
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings("ignore")
import ctypes
libc = ctypes.CDLL("libc.so.6")


class CFG():
    SIM_MODEL = "../../pretrain-model/all-MiniLM-L6-v2"
    test_path = "datasets/test.csv"
    MAX_LENGTH = 384  # 导入的BertModel的预训练的最大长度为256
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 16
    index_path = "/mnt/HDD0/wuzhiqiang/dataset/wikipedia-2023-07-faiss-index/wikipedia_202307.index"
    index_parquet_path = "/mnt/HDD0/wuzhiqiang/dataset/wikipedia-20230701/wiki_2023_index.parquet"

def main():
    df_test = pd.read_csv(CFG.test_path).drop("id", 1)
    model = SentenceTransformer(CFG.SIM_MODEL, device=CFG.device)
    model.max_seq_length = CFG.MAX_LENGTH  # 输入的最大长度
    prompt_embeddings = model.encode(
        df_test.prompt.values,
        batch_size=CFG.batch_size,
        device=CFG.device,
        show_progress_bar=True,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )
    # (200, 384)
    # 获得prompt的embedding
    prompt_embeddings = prompt_embeddings.detach().cpu().numpy()
    
    # 读取检索的Wiki的index
    sentence_index = read_index(CFG.index_path)
    _ = gc.collect()

    # 获得相似度最近的5个句子和index
    search_score, search_index = sentence_index.search(prompt_embeddings, 5)
    del search_index
    del prompt_embeddings
    _ = gc.collect()
    libc.malloc_trim(0)

    df = pd.read_parquet(CFG.index_parquet_path)
    print(df)


if __name__ == "__main__":
    main()