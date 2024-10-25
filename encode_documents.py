import pandas as pd 
import torch
import pickle
from data_preprocessing import load_word_to_int
from networks import TwoTowers, SkipGramModel

df_train = pd.read_pickle("data/preprocess_bing.pkl")
num_of_rows = len(df_train.index) - 1

# load word_to_int
word_to_int = load_word_to_int()
vocab_dim = len(word_to_int)
print("vocab size:", vocab_dim)

embedding_dim = 64
hidden_dim = embedding_dim * 2
print("hidden dimensions:", hidden_dim)
args_d = (vocab_dim, embedding_dim, hidden_dim)
DModel = TwoTowers.TowerDocument(*args_d)

# load weights TODO

encoded_docs = []

for index, row in df_train.iterrows():
    doc = row['relevant']
    with torch.no_grad():
        doc_enc = DModel(doc)
        encoded_docs.append(doc_enc)

with open("data/encoded_docs_list.pkl", "wb") as file:
    pickle.dump(encoded_docs, file)