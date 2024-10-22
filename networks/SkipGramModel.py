import torch

class SkipGramFoo(torch.nn.Module):
  def __init__(self, voc, emb, _):
    super().__init__()
    self.emb = torch.nn.Embedding(num_embeddings=voc, embedding_dim=emb)
    self.ffw = torch.nn.Linear(in_features=emb, out_features=voc, bias=False)
    self.sig = torch.nn.Sigmoid()

  def forward(self, inpt, trgs, rand):
    print(inpt.shape)
    emb = self.emb(inpt)
    #print("emb", emb.shape)
    ctx = self.ffw.weight[trgs]
    #print("ctx", ctx.shape)
    rnd = self.ffw.weight[rand]
    out = torch.bmm(ctx, emb.unsqueeze(-1)).squeeze()
    #print("out", out.shape)
    rnd = torch.bmm(rnd, emb.unsqueeze(-1)).squeeze()
    out = self.sig(out)
    rnd = self.sig(rnd)
    pst = -out.log().mean()
    ngt = -(1 - rnd + 10**(-3)).log().mean()
    return pst + ngt