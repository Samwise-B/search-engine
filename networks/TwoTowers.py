import torch
import math

class TowerQuery(torch.nn.Module):
  def __init__(self, voc, emb, hddn):
    super().__init__()
    self.emb = torch.nn.Embedding(num_embeddings=voc, embedding_dim=emb)
    self.rnn = torch.nn.RNN(input_size=emb, hidden_size=hddn, batch_first=True) # hidden size math.floor(emb/2)
    self.hidden_size = hddn

  def forward(self, query, batch_size):
    #print(query.shape)
    h0 = torch.zeros(1, batch_size, self.hidden_size)
    #print("hidden", h0.shape)
    emb = self.emb(query)
    #emb = emb.unsqueeze(0)
    #print("emb",emb.shape)
    _, enc = self.rnn(emb, h0)
    #print(enc.shape)
    return enc
  
class TowerDocument(torch.nn.Module):
  def __init__(self, voc, emb, hddn):
    super().__init__()
    self.emb = torch.nn.Embedding(num_embeddings=voc, embedding_dim=emb)
    self.rnn = torch.nn.RNN(input_size=emb, hidden_size=hddn, batch_first=True)
    self.hidden_size = hddn

  def forward(self, relevant, irrelevant, batch_size):
    h0 = torch.zeros(1, batch_size, self.hidden_size)
    emb_rel = self.emb(relevant)
    _, enc_rel = self.rnn(emb_rel, h0)
    #print(enc_rel.shape)

    emb_ir = self.emb(irrelevant)
    _, enc_ir = self.rnn(emb_ir, h0)
    #print(enc_ir.shape)
    return enc_rel, enc_ir