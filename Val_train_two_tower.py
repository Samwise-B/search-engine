from datasets import load_dataset
from networks import TwoTowers, SkipGramModel
from data_preprocessing import load_word_to_int, tokenize, clean_text
import torch
import pandas as pd
import wandb
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, log_loss, confusion_matrix

torch.manual_seed(42)


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

def pad_batch(batch):
    max_lengths = batch.apply(lambda row: max(len(row['query']), len(row['relevant']), len(row['irrelevant'])), axis=1)
    batch_max_length = max(max_lengths)

    padded_batch = batch.apply(lambda row: pad_row(row, batch_max_length), axis=1)
    #print(padded_batch.head())
    return padded_batch['query'].tolist(), padded_batch['relevant'].tolist(), padded_batch['irrelevant'].tolist()

# def pad_list(lst, max_length, pad_value=None):
#     """Pads a list with `pad_value` until it reaches `max_length`."""
#     return lst + [pad_value] * (max_length - len(lst))

def pad_row(row, max_length, pad_value=0):
    query = ([pad_value] * (max_length - len(row['query']))) + row['query']
    relevant = ([pad_value] * (max_length - len(row['relevant']))) + row['relevant']
    irrelevant = ([pad_value] * (max_length - len(row['irrelevant']))) + row['irrelevant']
    """Pads a list with `pad_value` until it reaches `max_length`."""
    return pd.Series({'query': query, "relevant": relevant, "irrelevant": irrelevant})

def distance_function(enc_one, enc_two):
    return 1 - torch.nn.functional.cosine_similarity(enc_one, enc_two)

def triplet_loss_function(encQ, encR, encIR, batch_size, margin=1):
    rel_dis = distance_function(encQ, encR)
    ir_dis = distance_function(encQ, encIR)
    #print(f"rel_dis: {rel_dis.shape}, ir_dis: {ir_dis.shape}")
    #return torch.max(torch.zeros(batch_size), rel_dis - ir_dis + margin).mean()
    return torch.nn.functional.relu(rel_dis - ir_dis + margin).mean()



df_val = pd.read_pickle("data/df_val_preprocess_bing.pkl")
num_of_rows = len(df_val.index) - 1
# load word_to_int
word_to_int = load_word_to_int()
vocab_dim = len(word_to_int)
print("vocab size:", vocab_dim)
# load embeddings
embedding_dim = 64
skip_args = (vocab_dim, embedding_dim, 2)
modelSkip = SkipGramModel.SkipGramFoo(*skip_args)
modelSkip.load_state_dict(torch.load("weights/fine_tuned_weights.pt", weights_only=True))

# initialise two towers
hidden_dim = embedding_dim * 2
print("hidden dimensions:", hidden_dim)
args_q = (vocab_dim, embedding_dim, hidden_dim)
args_d = (vocab_dim, embedding_dim, hidden_dim)
QModel = TwoTowers.TowerQuery(*args_q)
DModel = TwoTowers.TowerDocument(*args_d)

# set new embedding layers to fine-tuned weights
with torch.no_grad():
    QModel.emb.weight.data.copy_(modelSkip.emb.weight.clone())
    DModel.emb.weight.data.copy_(modelSkip.emb.weight.clone())

print('Query Model parameters: ', sum(p.numel() for p in QModel.parameters()))
print('Document Model parameters: ', sum(p.numel()for p in DModel.parameters()))

# define optimizers and device
optim_Q = torch.optim.Adam(QModel.parameters(), lr=0.001)
optim_D = torch.optim.Adam(DModel.parameters(), lr=0.001)
# schedular_Q = torch.optim.lr_scheduler.LRScheduler(optim_Q, gamma=0.9)
# schedular_D = torch.optim.lr_scheduler.LRScheduler(optim_D, gamma=0.9)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:", 'cuda' if torch.cuda.is_available() else 'cpu')



BATCH_SIZE = 256
QModel.to(device)
DModel.to(device)

# kfld = KFold(n_splits=5, shuffle=True, random_state=42)
# fold = 1
# res = []                                                                            

# for train_index, val_index in kfld.split(df_val):
#     print(f"Train: {train_index}, Val: {val_index}")
    
print("training...")
name = "two-tower-rnn-20ep-1000batch-lr-0.0001"
wandb.init(project='two-towers', name=name)


for i in range(5):
    wandb.log({"epoch": i+1})
    for j in range(0, len(df_val), BATCH_SIZE):
        #q, r, ir = process_batch_tower(row, word_to_int)
        batch = df_val.iloc[j:j + BATCH_SIZE]
        #print(batch.shape)
        q, r, ir = pad_batch(batch)
        q = torch.LongTensor(q).to(device)
        r = torch.LongTensor(r).to(device)
        ir = torch.LongTensor(ir).to(device)

        optim_D.zero_grad()
        optim_Q.zero_grad()
        q_emb = QModel(q, len(batch.index))
        r_emb, ir_emb = DModel(r, ir, len(batch.index))
        q_emb = q_emb.squeeze()
        r_emb = r_emb.squeeze()
        ir_emb = ir_emb.squeeze()
        #print(f"q_emb shape: {q_emb.shape}, r_emb: {r_emb.shape}, ir_emb: {ir_emb.shape}")
        loss = triplet_loss_function(q_emb, r_emb, ir_emb, len(batch.index))
        #print(f"loss: {loss.item()}")
        wandb.log({"loss": loss.item()})
        loss.backward()
        optim_D.step()
        optim_Q.step()
    # schedular_D.step()
    # schedular_Q.step()
    # get 1 query, relevant and irrelevant passage
    # tokenize query, passages
    # pass token to Qmodel
    # pass passages tokens to Dmodel
    # calculate loss
    # backprop
    # step optimizer
# save model
# print("saving")
# torch.save(QModel.state_dict(), f'./weights/QModel_weights_{name}.pt')
# torch.save(DModel.state_dict(), f'./weights/DModel_weights_{name}.pt')
# print('Uploading...')
# artifact = wandb.Artifact('model-weights', type='model')
# artifact.add_file(f'./weights/QModel_weights_{name}.pt')
# artifact.add_file(f'./weights/DModel_weights_{name}.pt')
# wandb.log_artifact(artifact)
# print('Done!')
# wandb.finish()


with torch.no_grad():
    user_embeddings = QModel(q, len(batch.index))
    item_embeddings = DModel(r, ir, len(batch.index))

# Apply KNN to retrieve the top-K nearest items for each user
K = 5  # Number of neighbors to retrieve

# Fit KNN model on item embeddings
knn = NearestNeighbors(n_neighbors=K, metric='cosine')
knn.fit(item_embeddings)

# Example: Retrieve top-K items for a specific user embedding
user_index = 0
user_query_embedding = user_embeddings[user_index].reshape(1, -1)

# Find K-nearest item neighbors for the user query embedding
distances, indices = knn.kneighbors(user_query_embedding)

# Output results
print(f"Top-{K} item indices for user {user_index}:", indices[0])
print(f"Distances to top-{K} items:", distances[0])