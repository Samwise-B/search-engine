import torch
import faiss

from pathlib import Path
import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

root_repo = Path(__file__).parent.parent
sys.path.append(str(root_repo))


from networks import TwoTowers
from utils.data_preprocessing import load_word_to_int, tokenize_string
from pathlib import Path
from torch.nn.functional import cosine_similarity

MODEL_WEIGHT_PATH = Path(__file__).parent.parent / "weights/QModel_weights.pt"
FAISS_INDEX_PATH = Path(__file__).parent.parent / "data/faiss_index.faiss"
DOCS_PATH = Path(__file__).parent.parent / "data/all_docs"

VOCAB_SIZE = 112_242
word_to_int = load_word_to_int()
device = "cuda" if torch.cuda.is_available() else "cpu"
map_location = torch.device(device)

print("Loading model")
model = TwoTowers.TowerQuery(
    VOCAB_SIZE, 64, 128
).to(device)

model.load_state_dict(
    torch.load(MODEL_WEIGHT_PATH, weights_only=True, map_location=map_location)
)
print("Model loaded")

faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))

def get_line_from_index(index: int):

    with open(DOCS_PATH, "r",encoding="utf-8") as f:
        for doc_index, line in enumerate(f):
            if index == doc_index:
                return line
        return "LINE NOT FOUND"
        #f.seek(index)
        #return f.readline()


def process_query(query: str):
    tokens = torch.tensor(tokenize_string(query, word_to_int))
    query_encoding = model.encode_query_single(tokens).detach().cpu().numpy()

    _, top_k_indices = faiss_index.search(query_encoding, 10)

    return [get_line_from_index(i) for i in top_k_indices[0]]

if __name__=='__main__':
    for i in range(5):
        print(get_line_from_index(i))
    askar_resp = process_query('King queen')
    same_resp = process_query('Sam')
    pass