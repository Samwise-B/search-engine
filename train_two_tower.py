from datasets import load_dataset
from networks import TwoTowers, SkipGramModel
from data_preprocessing import load_word_to_int, tokenize, clean_text
import torch
import pandas as pd
import wandb


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


#ds = load_dataset("microsoft/ms_marco", "v1.1")
#df_train = pd.DataFrame(ds['train'])
#df_train = df_train[['query', 'passages']]
# train_answers = train_answers
df_train = pd.read_pickle("data/preprocess_bing.pkl").iloc[:100]
num_of_rows = len(df_train.index) - 1

# load word_to_int
word_to_int = load_word_to_int()
vocab_dim = len(word_to_int)
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
optim_Q = torch.optim.Adam(QModel.parameters(), lr=0.001)
optim_D = torch.optim.Adam(DModel.parameters(), lr=0.001)
device = torch.device('cpu')

BATCH_SIZE = 10

QModel.to(device)
DModel.to(device)
print("training...")
wandb.init(project='two-towers', name='two-tower-rnn-small')
for i in range(5):
    for j in range(0, len(df_train), BATCH_SIZE):
        #q, r, ir = process_batch_tower(row, word_to_int)
        batch = df_train.iloc[j:j + BATCH_SIZE]
        #print(batch.shape)
        q, r, ir = pad_batch(batch)
        #print(q.shape)

        optim_D.zero_grad()
        optim_Q.zero_grad()
        q_emb = QModel(torch.LongTensor(q), len(batch.index))
        r_emb, ir_emb = DModel(torch.LongTensor(r), torch.LongTensor(ir), len(batch.index))
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
    # get 1 query, relevant and irrelevant passage
    # tokenize query, passages
    # pass token to Qmodel
    # pass passages tokens to Dmodel
    # calculate loss
    # backprop
    # step optimizer
# save model
print("saving")
torch.save(QModel.state_dict(), './weights/QModel_weights.pt')
torch.save(DModel.state_dict(), './weights/DModel_weights.pt')
print('Uploading...')
artifact = wandb.Artifact('model-weights', type='model')
artifact.add_file('./weights/QModel_weights.pt')
artifact.add_file('./weights/DModel_weights.pt')
wandb.log_artifact(artifact)
print('Done!')
wandb.finish()
