import torch
import random
import gensim.downloader as api
import datasets
import numpy as np
from tqdm import tqdm
import wandb
from pathlib import Path
import pickle
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence


print("loading gensim")
wrd2vec = api.load("word2vec-google-news-300")
#txt2vec = lambda txt: torch.tensor(np.array([wrd2vec[x] if x in wrd2vec else [0.0]*300 for x in txt.lower().split()]), dtype=torch.float32, device=device)
print("loading dataset")
with open(Path(__file__).parent.parent / "data/gensim_marco.pkl", "rb") as file:
  ds = pickle.load(file)
#ds = datasets.load_dataset("microsoft/ms_marco", "v1.1")
print("done")

#ds = {"query": ds['query'][:5000], "pos": ds['pos'][:5000], "neg":ds['neg'][:5000]}

#train_list = ds['train'][:3000]

class QryTower(torch.nn.Module):
  def __init__(self, emb_dim, hdn_dim, num_layers=1):
    super(QryTower, self).__init__()
    self.rnn = torch.nn.RNN(input_size=emb_dim, hidden_size=hdn_dim, batch_first=True, num_layers=num_layers)

  def forward(self, queries):
    seq_lens = [len(q) for q in queries]
    padded = pad_sequence(queries, batch_first=True)
    packed = pack_padded_sequence(padded, seq_lens, batch_first=True, enforce_sorted=False)
    _, enc = self.rnn(packed)
    return enc

class DocTower(torch.nn.Module):
  def __init__(self, emb_dim, hdn_dim, num_layers=1):
    super(DocTower, self).__init__()
    self.rnn = torch.nn.RNN(input_size=emb_dim, hidden_size=hdn_dim, batch_first=True, num_layers=num_layers)

  def forward(self, docs):
    seq_lens = [len(d) for d in docs]
    padded = pad_sequence(docs, batch_first=True)
    packed = pack_padded_sequence(padded, seq_lens, batch_first=True, enforce_sorted=False)
    _, enc = self.rnn(packed)
    return enc

class TwoTower(torch.nn.Module):
  def __init__(self):
    super(TwoTower, self).__init__()
    self.qry_tower = QryTower(300, 400)
    self.doc_tower = DocTower(300, 400)

  def forward(self, qry, doc):
    qry_emb = self.qry_tower(qry)
    doc_emb = self.doc_tower(doc)
    return qry_emb, doc_emb
  
  def encode_doc(self, doc):
    return self.doc_tower(doc)
  
# def get_batch(batch_size, num_rows):
#     queries = [None] * batch_size
#     positives = [None] * batch_size
#     negatives = [None] * batch_size
#     for i in range(batch_size):
#         pos_index = random.randint(0, num_rows - 1)
#         query = txt2vec(train_list['query'][pos_index])
#         passages = train_list['passages'][pos_index]['passage_text']
#         pos = txt2vec(passages[random.randint(0, len(passages)-1)])

#         neg_index = random.randint(0, num_rows - 1)
#         neg_passages = train_list['passages'][neg_index]['passage_text']
#         neg = txt2vec(neg_passages[random.randint(0, len(neg_passages)-1)])

#         queries[i] = query
#         positives[i] = pos
#         negatives[i] = neg

def get_batch(batch_size, num_rows):
    index = random.randint(0, num_rows-batch_size-1)
    queries = [torch.tensor(wrd2vec[seq], dtype=torch.float32) for seq in ds['query'][index:index+batch_size]]
    positives = [torch.tensor(wrd2vec[seq], dtype=torch.float32) for seq in ds['pos'][index:index+batch_size]]
    negatives = [torch.tensor(wrd2vec[seq], dtype=torch.float32) for seq in ds['neg'][index:index+batch_size]]
    return queries, positives, negatives

def dist_function(query, sample):
    # Shape [N, 2*H]
    return 1 - torch.nn.functional.cosine_similarity(query, sample, dim=1)

two_tower = TwoTower()
criterion = torch.nn.TripletMarginWithDistanceLoss(
            distance_function=dist_function, reduction="mean"
        )
optimizer = torch.optim.Adam(two_tower.parameters(), lr=0.001)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:", 'cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 1000

num_rows = len(ds['query'])
two_tower.to(device)
name = "avg_pool_rnn"
wandb.init(project='search-engine', name=name)
for epoch in range(10):
  for idx in tqdm(range(0, num_rows // BATCH_SIZE), total=(num_rows // BATCH_SIZE), desc="Training Epoch"):
    qry, pos, neg = get_batch(BATCH_SIZE, num_rows)
    

    optimizer.zero_grad()
    qry_emb, pos_emb = two_tower(qry, pos)
    neg_emb = two_tower.encode_doc(neg)
    loss = criterion(qry_emb, pos_emb, neg_emb)
    pos_dis = dist_function(qry_emb, pos_emb).mean()
    neg_dis = dist_function(qry_emb, neg_emb).mean()

    loss.backward()
    optimizer.step()
    wandb.log({"loss": loss.item(), "pos_dis": pos_dis, "neg_dis": neg_dis})
    #print(tot_loss.item())

print('Done!')
wandb.finish()