import pandas as pd 
import torch
import pickle
from pathlib import Path
from tqdm import tqdm
from more_itertools import chunked
from data_preprocessing import tokenize_string
import sys
import faiss

repo_root = Path(__file__).parent.parent
sys.path.append(str(repo_root))
from utils.data_preprocessing import load_word_to_int
from networks import TwoTowers

INTERNET_FILEPATH = Path(__file__).parent.parent / "data/all_docs"

# load word_to_int
word_to_int = load_word_to_int()
vocab_dim = len(word_to_int)
print("vocab size:", vocab_dim)

print("creating model...")
embedding_dim = 64
hidden_dim = embedding_dim * 2
print("hidden dimensions:", hidden_dim)
args_d = (vocab_dim, embedding_dim, hidden_dim)
DModel = TwoTowers.TowerDocument(*args_d)

print("loading model weights")
# DModel.load_state_dict(torch.load(
#     Path(__file__).parent.parent / "weights/DModel_weights.pt", weights_only=True))

print("done")

encoded_docs = []

with open(INTERNET_FILEPATH, "r", encoding="utf-8") as f:
    docs = list(f)
    total_lines = len(docs)

BATCH_SIZE = 1000

with torch.inference_mode():
    for batch in tqdm(chunked(docs, BATCH_SIZE), total=(total_lines // BATCH_SIZE), desc="Encoding Documents"):
        docs_tokenized = [tokenize_string(doc, word_to_int) for doc in batch]
        doc_enc = DModel.encode_docs(docs_tokenized)
        encoded_docs.append(doc_enc)

encoded_docs = torch.cat(encoded_docs)
# dimensions = encoded_docs[0].shape[1]
# index = faiss.IndexFlatIP(dimensions)
# index.add(encoded_docs)

with open(Path(__file__).parent.parent / "data/encoded_docs_list.pkl", "wb") as file:
    pickle.dump(encoded_docs, file)