from datasets import load_dataset
from networks import TwoTowers, SkipGramModel
from data_preprocessing import load_word_to_int, tokenize, clean_text
import torch
import pandas as pd


torch.seed(42)


def process_batch_tower(row, word_to_int):
    query = row['query']
    passage_col = row['passages']
    passages = passage_col['passage_text']
    relevant_doc = passages[passage_col['is_selected'].index(1)]
    # print(passage_['is_selected'].index(0))
    irrelevant_doc = passages[passage_col['is_selected'].index(0)]
    # print(passage_col['is_selected'].index(1), passage_col['is_selected'].index(0))

    cleaned_query = clean_text(query)
    cleaned_docR = clean_text(relevant_doc)
    cleaned_dorIr = clean_text(irrelevant_doc)

    query_tokens = tokenize(cleaned_query, word_to_int)
    rel_doc_tokens = tokenize(relevant_doc, word_to_int)
    ir_doc_tokens = tokenize(irrelevant_doc, word_to_int)

    return query_tokens, rel_doc_tokens, ir_doc_tokens


def distance_function(enc_one, enc_two):
    return 1 - torch.nn.functional.cosine_similarity(enc_one, enc_two)


def triplet_loss_function(encQ, encR, encIR, margin=1):
    rel_dis = distance_function(encQ, encR)
    ir_dis = distance_function(encQ, encIR)
    return max(0, rel_dis - ir_dis + margin)


ds = load_dataset("microsoft/ms_marco", "v1.1")
df_train = pd.DataFrame(ds['train'])
df_train = df_train[['query', 'passages']]
# train_answers = train_answers
num_of_rows = len(df_train.index) - 1

# load word_to_int
word_to_int = load_word_to_int()
vocab_dim = len(word_to_int) + 1
print("vocab size:", vocab_dim)

# load embeddings
embedding_dim = 64
skip_args = (vocab_dim, embedding_dim, 2)
modelSkip = SkipGramModel.SkipGramFoo(*skip_args)
modelSkip.load_state_dict(torch.load(
    "weights/fine_tuned_weights.pt", weights_only=True))

# initialise two towers
hidden_dim = embedding_dim * 2
print("hidden dimensions:", hidden_dim)
args = (vocab_dim, embedding_dim, hidden_dim)
QModel = TwoTowers.TowerQuery(*args)
DModel = TwoTowers.TowerDocument(*args)

# set new embedding layers to fine-tuned weights
with torch.no_grad():
    QModel.emb.weight.data.copy_(modelSkip.emb.weight.clone())
    DModel.emb.weight.data.copy_(modelSkip.emb.weight.clone())

print('Query Model parameters: ', sum(p.numel() for p in QModel.parameters()))
print('Document Model parameters: ', sum(p.numel()
      for p in DModel.parameters()))

# define optimizers and device
optim_Q = torch.optim.SGD(QModel.parameters(), lr=0.001)
optim_D = torch.optim.SGD(DModel.parameters(), lr=0.001)
device = torch.device('cpu')

QModel.to(device)
DModel.to(device)
for i in range(1):
    for index, row in df_train.iterrows():
        q, r, ir = process_batch_tower(row, word_to_int)
        q_emb = QModel(torch.LongTensor(q))
        r_emb, ir_emb = DModel(torch.LongTensor(r), torch.LongTensor(ir))
        loss = triplet_loss_function(q_emb, r_emb, ir_emb)
        print(f"loss: {loss.item()}")
        loss.backward()
        optim_D.step()
        optim_Q.step()
    # get 1 query, relevant and irrelevant passage
    # tokenize query, passages
    # pass token to Qmodel
    # pass passages tokens to Dmodel
    # calculate loss
    # backprop
    # step optimizer
