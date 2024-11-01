import torch
import datasets
import gensim.downloader as api
import numpy as np
import random
import pickle
from tqdm import tqdm
from pathlib import Path

SAVE_DIR = Path(__file__).parent.parent / "data/gensim_marco.pkl"

def words_to_id(tokenizer, words):
    return [tokenizer[word] if word in tokenizer else 0 for word in words]

print("loading gensim")
wrd2vec = api.load("word2vec-google-news-300")
tokenizer = wrd2vec.key_to_index
#txt2vec = lambda txt: torch.tensor(np.array([wrd2vec[x] if x in wrd2vec else [0.0]*300 for x in txt.lower().split()]), dtype=torch.float32, device=device)

print("loading dataset")
ds = datasets.load_dataset("microsoft/ms_marco", "v1.1")
print("done")
ds = ds['train']
passages = ds['passages']
queries = []
positives = []
negatives = []

for row in tqdm(ds, total=len(ds)-1, desc="processing marco"):
    query = words_to_id(tokenizer, row['query'].strip().lower().split(" "))
    #query = row['query']
    num_passages = len(row['passages']['passage_text'])
    # tmp_queries = [query for _ in row['passages']['passage_text']]
    # tmp_pos = [words_to_vec(wrd2vec, pos.strip().lower().split(" ")) for pos in row['passages']['passage_text']]
    # tmp_neg = [None for _ in ]
    
    tmp_queries = [None] * num_passages
    tmp_pos = [None] * num_passages
    tmp_neg = [None] * num_passages
    #tmp_queries = [(query, passage, ds['passages'][random.randint(0, len(ds)-1)]['passage_text'][random.randint(0, len(ds))]) for passage in row['passages']['passage_text']]
    for ind, passage in enumerate(row['passages']['passage_text']):
        tmp_queries[ind] = query
        tmp_pos[ind] = words_to_id(tokenizer, passage.strip().lower().split(" "))
        #queries.append(words_to_vec(wrd2vec, query))
        rnd = random.randint(0, len(ds)-1)
        passages = ds[rnd]['passages']['passage_text']
        tmp_neg[ind] = words_to_id(tokenizer, passages[random.randint(0, len(passages)-1)].strip().lower().split(" "))
    
    queries.extend(tmp_queries), positives.extend(tmp_pos), negatives.extend(tmp_neg)

print("saving data")
triplet = {"query": queries, "pos": positives, "neg": negatives}
with open(SAVE_DIR, "wb") as file:
    pickle.dump(triplet, file)