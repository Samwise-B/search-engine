from datasets import load_dataset
import torch
import pandas as pd
import wandb
from pathlib import Path
from tqdm import tqdm
import sys

repo_root = Path(__file__).parent.parent
sys.path.append(str(repo_root))
from networks import TwoTowers, SkipGramModel
from utils.data_preprocessing import load_word_to_int, tokenize, clean_text

VALIDATION_PATH = Path(__file__).parent.parent / "data/validation_preprocess_bing.pkl"
TRAIN_PATH = Path(__file__).parent.parent / "data/preprocess_bing.pkl"
EMBEDDING_WEIGHTS = Path(__file__).parent.parent / "weights/fine_tuned_weights.pt"
WEIGHTS_DIR = Path(__file__).parent.parent / "weights"


torch.manual_seed(42)

def get_validation_loss():
    BATCH_SIZE = 1000
    df= pd.read_pickle(VALIDATION_PATH)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    losses = []
    positive_distances = []
    negative_distances = []
    # loop over batches of dataframe
    with torch.inference_mode():
        for j in tqdm(range(0, len(df), BATCH_SIZE), total=len(df) / BATCH_SIZE, desc="Training Epoch"):
            batch = df.iloc[j:j + BATCH_SIZE]
            q, r, ir = batch['query'].tolist(), batch['relevant'].tolist(), batch['irrelevant'].tolist()
            q, r, ir = torch.LongTensor(q).to(device), torch.LongTensor(r).to(device), torch.LongTensor(ir).to(device)
            q_emb = QModel(q, len(batch.index))
            r_emb, ir_emb = DModel(r, ir, len(batch.index))
            q_emb, r_emb, ir_emb = q_emb.squeeze(), r_emb.squeeze(), ir_emb.squeeze()
            #print(f"q_emb shape: {q_emb.shape}, r_emb: {r_emb.shape}, ir_emb: {ir_emb.shape}")
            loss, pos_dis, neg_dis = triplet_loss_function(q_emb, r_emb, ir_emb, len(batch.index))
            losses.append(loss)
            positive_distances.append(pos_dis)
            negative_distances.append(neg_dis)
            #print(f"loss: {loss.item()}")
        wandb.log({
            "loss": sum(losses) / len(df), 
            "train_positive_distance": sum(positive_distances) / len(df), 
            "train_negative_distance": sum(negative_distances) / len(df)
        })

def distance_function(enc_one, enc_two):
    return 1 - torch.nn.functional.cosine_similarity(enc_one, enc_two, dim=1)

def triplet_loss_function(encQ, encR, encIR, margin=1):
    rel_dis = distance_function(encQ, encR)
    ir_dis = distance_function(encQ, encIR)
    loss_func = torch.nn.TripletMarginWithDistanceLoss(distance_function=distance_function, reduction="mean", margin=margin)
    #return torch.nn.functional.relu(rel_dis - ir_dis + margin).mean(), rel_dis, ir_dis
    return loss_func(encQ, encR, encIR), rel_dis, ir_dis

#ds = load_dataset("microsoft/ms_marco", "v1.1")
#df_train = pd.DataFrame(ds['train'])
#df_train = df_train[['query', 'passages']]
# train_answers = train_answers
df_train = pd.read_pickle(TRAIN_PATH)
df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)
#df_train = df_train[:1000]
num_of_rows = len(df_train.index) - 1

# load word_to_int
word_to_int = load_word_to_int()
vocab_dim = len(word_to_int)
print("vocab size:", vocab_dim)

# load embeddings
embedding_dim = 64
skip_args = (vocab_dim, embedding_dim, 2)
modelSkip = SkipGramModel.SkipGram(*skip_args)
modelSkip.load_state_dict(torch.load(EMBEDDING_WEIGHTS, weights_only=True))

# initialise two towers
hidden_dim = embedding_dim * 2
print("hidden dimensions:", hidden_dim)
args_q = (vocab_dim, embedding_dim, hidden_dim)
args_d = (vocab_dim, embedding_dim, hidden_dim)
QModel = TwoTowers.TowerQuery(*args_q)
DModel = TwoTowers.TowerDocument(*args_d)


# set new embedding layers to fine-tuned weights
# with torch.no_grad():
#     QModel.emb.weight.data = modelSkip.emb.weight.data
#     DModel.emb.weight.data = modelSkip.emb.weight.data

print('Query Model parameters: ', sum(p.numel() for p in QModel.parameters()))
print('Document Model parameters: ', sum(p.numel()
      for p in DModel.parameters()))

# define optimizers and device
optim_Q = torch.optim.Adam(QModel.parameters(), lr=0.001)
optim_D = torch.optim.Adam(DModel.parameters(), lr=0.001)

# schedular_Q = torch.optim.lr_scheduler.LRScheduler(optim_Q, gamma=0.9)
# schedular_D = torch.optim.lr_scheduler.LRScheduler(optim_D, gamma=0.9)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("device:", 'cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 1000

QModel.to(device)
DModel.to(device)
print("training...")
name = "rnn-20ep-1000batch-lr-0.001-randomshuffle-packed"
wandb.init(project='search-engine', name=name)
for i in range(20):
    wandb.log({"epoch": i+1})
    for j in tqdm(range(0, len(df_train), BATCH_SIZE), total=len(df_train) / BATCH_SIZE, desc="Training Epoch"):
        batch = df_train.iloc[j:j + BATCH_SIZE]
        q, r, ir = batch['query'].tolist(), batch['relevant'].tolist(), batch['irrelevant'].tolist()
        #q, r, ir = torch.LongTensor(q).to(device), torch.LongTensor(r).to(device), torch.LongTensor(ir).to(device)

        optim_D.zero_grad()
        optim_Q.zero_grad()
        q_emb = QModel(q, len(batch.index))
        r_emb, ir_emb = DModel(r, ir, len(batch.index))
        q_emb = q_emb.squeeze()
        r_emb = r_emb.squeeze()
        ir_emb = ir_emb.squeeze()
        #print(f"q_emb shape: {q_emb.shape}, r_emb: {r_emb.shape}, ir_emb: {ir_emb.shape}")
        loss, pos_dis, neg_dis = triplet_loss_function(q_emb, r_emb, ir_emb)
        #print(f"loss: {loss.item()}")
        wandb.log({"loss": loss.item(), "train_positive_distance": pos_dis.mean(), "train_negative_distance": neg_dis.mean()})
        loss.backward()
        optim_D.step()
        optim_Q.step()

# save model
print("saving")
torch.save(QModel.state_dict(), WEIGHTS_DIR / f'QModel_weights_{name}.pt')
torch.save(DModel.state_dict(), WEIGHTS_DIR / f'DModel_weights_{name}.pt')
print('Uploading...')
artifact = wandb.Artifact('model-weights', type='model')
artifact.add_file(WEIGHTS_DIR / f'QModel_weights_{name}.pt')
artifact.add_file(WEIGHTS_DIR / f'DModel_weights_{name}.pt')
wandb.log_artifact(artifact)
print('Done!')
wandb.finish()
