import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TowerQuery(torch.nn.Module):
  def __init__(self, voc, emb, hddn, num_layers=1):
    super().__init__()
    self.emb = torch.nn.Embedding(num_embeddings=voc, embedding_dim=emb)
    self.rnn = torch.nn.RNN(input_size=emb, hidden_size=hddn, batch_first=True, num_layers=num_layers) # hidden size math.floor(emb/2)
    self.hidden_size = hddn
    self.num_layers = num_layers

  def forward(self, query, batch_size):
    query = [torch.tensor(q, dtype=torch.long, device=device) for q in query]
    padded_queries = pad_sequence(query, batch_first=True)
    padded_emb = self.emb(padded_queries)

    q_lens = [len(q) for q in query]
    packed_enc = pack_padded_sequence(padded_emb, q_lens, batch_first=True, enforce_sorted=False)

    _, enc = self.rnn(packed_enc)
    return enc
  
  def encode_query_single(self, query):
     emb = self.emb(query)
     _, enc = self.rnn(emb)
     return enc
  
class TowerDocument(torch.nn.Module):
  def __init__(self, voc, emb, hddn, num_layers=1):
    super().__init__()
    self.emb = torch.nn.Embedding(num_embeddings=voc, embedding_dim=emb)
    self.rnn = torch.nn.RNN(input_size=emb, hidden_size=hddn, batch_first=True, num_layers=num_layers)
    self.hidden_size = hddn
    self.num_layers = num_layers

  def forward(self, relevant, irrelevant, batch_size):
    positives = [torch.tensor(d) for d in relevant]
    pad_rel = pad_sequence(positives, batch_first=True)

    emb_rel = self.emb(pad_rel)
    r_lens = [len(doc) for doc in positives]
    packed_enc_rel = pack_padded_sequence(emb_rel, r_lens, batch_first=True, enforce_sorted=False)

    _, enc_rel = self.rnn(packed_enc_rel)


    negatives = [torch.tensor(d) for d in irrelevant]
    padded_negs = pad_sequence(negatives, batch_first=True)
    emb_ir = self.emb(padded_negs)

    ir_lens = [len(doc) for doc in irrelevant]
    
    packed_enc_ir = pack_padded_sequence(emb_ir, ir_lens, batch_first=True, enforce_sorted=False)

    _, enc_ir = self.rnn(packed_enc_ir)
    return enc_rel, enc_ir
  
  def encode_doc_single(self, doc):
    emb = self.emb(doc)
    _, enc = self.rnn(emb)

    return enc
  
  def encode_docs(self, docs):
    # Shape [N]
        doc_lengths = torch.tensor([len(seq) for seq in docs], dtype=torch.long)

        docs = [torch.tensor(d, dtype=torch.long, device=device) for d in docs]

        # Shape [N, Lmax] (for each)
        padded_docs = pad_sequence(docs, batch_first=True)

        # Shape [N, Lmax, E]
        padded_docs_embeds = self.emb(padded_docs)

        packed_padded_docs = pack_padded_sequence(
            padded_docs_embeds, doc_lengths, batch_first=True, enforce_sorted=False
        )

        _, encoded_docs = self.rnn(packed_padded_docs)
        return encoded_docs.squeeze()