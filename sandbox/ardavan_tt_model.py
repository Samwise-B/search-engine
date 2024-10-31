#
#
#
import torch
import random
import gensim.downloader as api
import datasets


#
#
#
wrd2vec = api.load("word2vec-google-news-300")
txt2vec = lambda txt: torch.tensor([wrd2vec[x] if x in wrd2vec else [0.0]*300 for x in txt.lower().split()])
ds = datasets.load_dataset("microsoft/ms_marco", "v1.1")


#
#
#
foo = txt2vec('king of queen hello world besart')
print(foo.shape) # torch.Size([6, 300])


#
#
#
class QryTower(torch.nn.Module):
  def __init__(self):
    super(QryTower, self).__init__()
    self.fc1 = torch.nn.Linear(300, 128)
    self.fc2 = torch.nn.Linear(128, 64)
    self.fc3 = torch.nn.Linear(64, 32)
    self.fc4 = torch.nn.Linear(32, 16)

  def forward(self, x):
    y = self.fc1(x)
    y = torch.nn.functional.relu(y)
    y = self.fc2(y)
    y = torch.nn.functional.relu(y)
    y = self.fc3(y)
    y = torch.nn.functional.relu(y)
    y = self.fc4(y)
    return y


#
#
#
class DocTower(torch.nn.Module):
  def __init__(self):
    super(DocTower, self).__init__()
    self.fc1 = torch.nn.Linear(300, 128)
    self.fc2 = torch.nn.Linear(128, 64)
    self.fc3 = torch.nn.Linear(64, 32)
    self.fc4 = torch.nn.Linear(32, 16)

  def forward(self, x):
    y = self.fc1(x)
    y = torch.nn.functional.relu(y)
    y = self.fc2(y)
    y = torch.nn.functional.relu(y)
    y = self.fc3(y)
    y = torch.nn.functional.relu(y)
    y = self.fc4(y)
    return y


#
#
#
class TwoTower(torch.nn.Module):
  def __init__(self):
    super(TwoTower, self).__init__()
    self.qry_tower = QryTower()
    self.doc_tower = DocTower()

  def forward(self, qry, doc):
    qry_emb = self.qry_tower(qry)
    doc_emb = self.doc_tower(doc)
    return qry_emb, doc_emb


#
#
#
two_tower = TwoTower()
criterion = torch.nn.CosineEmbeddingLoss()
optimizer = torch.optim.Adam(two_tower.parameters(), lr=0.001)


#
#
#
for i in range(100):
  for idx in range(100):
    rnd = random.randint(0, len(ds['train']) - 1)
    qry = txt2vec(ds['train'][idx]['query']).mean(dim=0)
    pos = txt2vec(ds['train'][idx]['passages']['passage_text'][0]).mean(dim=0)
    neg = txt2vec(ds['train'][rnd]['passages']['passage_text'][0]).mean(dim=0)
    optimizer.zero_grad()
    qry_emb, pos_emb = two_tower(qry, pos)
    _______, neg_emb = two_tower(qry, neg)
    pos_loss = criterion(qry_emb, pos_emb, torch.tensor(1))
    neg_loss = criterion(qry_emb, neg_emb, torch.tensor(-1))
    tot_loss = pos_loss + neg_loss
    tot_loss.backward()
    optimizer.step()
    print(tot_loss.item())