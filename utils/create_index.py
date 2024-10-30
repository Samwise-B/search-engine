import faiss
import torch
import pickle
from pathlib import Path

with open(Path(__file__).parent.parent / "data/encoded_docs_list.pkl", "rb") as f:
    doc_encodings = pickle.load(f)

doc_encodings = torch.cat(doc_encodings.squeeze())
dimensions = doc_encodings.shape[1]
index = faiss.IndexFlatIP(dimensions)

index.add(doc_encodings.detach().numpy())
print("total indexes:", index.ntotal)
print("saving index")
faiss.write_index(index, str(Path(__file__).parent.parent / "data/faiss_index.faiss"))