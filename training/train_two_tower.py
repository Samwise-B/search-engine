from datasets import load_dataset
import torch
import pandas as pd
import wandb
from pathlib import Path
import sys

repo_root = Path(__file__).parent.parent
sys.path.append(str(repo_root))
from networks import TwoTowers, SkipGramModel
from utils.data_preprocessing import load_word_to_int, tokenize, clean_text



torch.manual_seed(42)

def pad_batch(batch):
    max_lengths = batch.apply(lambda row: max(len(row['query']), len(row['relevant']), len(row['irrelevant'])), axis=1)
    batch_max_length = max(max_lengths)

    padded_batch = batch.apply(lambda row: pad_row_right(row, batch_max_length), axis=1)
    #print(padded_batch.head())
    return padded_batch['query'].tolist(), padded_batch['relevant'].tolist(), padded_batch['irrelevant'].tolist()

def pad_row_left(row, max_length, pad_value=0):
    query = ([pad_value] * (max_length - len(row['query']))) + row['query']
    relevant = ([pad_value] * (max_length - len(row['relevant']))) + row['relevant']
    irrelevant = ([pad_value] * (max_length - len(row['irrelevant']))) + row['irrelevant']
    """Pads a list with `pad_value` until it reaches `max_length`."""
    return pd.Series({'query': query, "relevant": relevant, "irrelevant": irrelevant})

def pad_row_right(row, max_length, pad_value=0):
    query = row['query'] + ([pad_value] * (max_length - len(row['query'])))
    relevant = row['relevant'] + ([pad_value] * (max_length - len(row['relevant'])))
    irrelevant = row['irrelevant'] + ([pad_value] * (max_length - len(row['irrelevant'])))
    """Pads a list with `pad_value` until it reaches `max_length`."""
    return pd.Series({'query': query, "relevant": relevant, "irrelevant": irrelevant})


def distance_function(enc_one, enc_two):
    return 1 - torch.nn.functional.cosine_similarity(enc_one, enc_two, dim=1)


def triplet_loss_function(encQ, encR, encIR, margin=1):
    rel_dis = distance_function(encQ, encR)
    ir_dis = distance_function(encQ, encIR)
    return torch.nn.functional.relu(rel_dis - ir_dis + margin).mean()


#ds = load_dataset("microsoft/ms_marco", "v1.1")
#df_train = pd.DataFrame(ds['train'])
#df_train = df_train[['query', 'passages']]
# train_answers = train_answers
df_train = pd.read_pickle("../data/preprocess_bing.pkl")
df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)
num_of_rows = len(df_train.index) - 1

# load word_to_int
word_to_int = load_word_to_int()
vocab_dim = len(word_to_int)
print("vocab size:", vocab_dim)

# load embeddings
embedding_dim = 64
skip_args = (vocab_dim, embedding_dim, 2)
modelSkip = SkipGramModel.SkipGram(*skip_args)
modelSkip.load_state_dict(torch.load(
    "../weights/fine_tuned_weights.pt", weights_only=True))

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
print('Document Model parameters: ', sum(p.numel()
      for p in DModel.parameters()))

# define optimizers and device
optim_Q = torch.optim.Adam(QModel.parameters(), lr=0.0001)
optim_D = torch.optim.Adam(DModel.parameters(), lr=0.0001)

# schedular_Q = torch.optim.lr_scheduler.LRScheduler(optim_Q, gamma=0.9)
# schedular_D = torch.optim.lr_scheduler.LRScheduler(optim_D, gamma=0.9)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("device:", 'cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 1000

QModel.to(device)
DModel.to(device)
print("training...")
name = "two-tower-rnn-20ep-1000batch-lr-0.0001-randomshuffle-packed"
wandb.init(project='two-towers', name=name)
for i in range(20):
    wandb.log({"epoch": i+1})
    for j in range(0, len(df_train), BATCH_SIZE):
        batch = df_train.iloc[j:j + BATCH_SIZE]
        q_lens = batch['query_length'].tolist()
        r_lens = batch['r_lens_list'].tolist()
        ir_lens = batch['ir_lens_list'].tolist()
        #print(batch.shape)
        q, r, ir = pad_batch(batch)
        q = torch.LongTensor(q).to(device)
        r = torch.LongTensor(r).to(device)
        ir = torch.LongTensor(ir).to(device)

        optim_D.zero_grad()
        optim_Q.zero_grad()
        q_emb = QModel(q, q_lens, len(batch.index))
        r_emb, ir_emb = DModel(r, ir, r_lens, ir_lens, len(batch.index))
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
print("saving")
torch.save(QModel.state_dict(), f'../weights/QModel_weights_{name}.pt')
torch.save(DModel.state_dict(), f'../weights/DModel_weights_{name}.pt')
print('Uploading...')
artifact = wandb.Artifact('model-weights', type='model')
artifact.add_file(f'../weights/QModel_weights_{name}.pt')
artifact.add_file(f'../weights/DModel_weights_{name}.pt')
wandb.log_artifact(artifact)
print('Done!')
wandb.finish()
