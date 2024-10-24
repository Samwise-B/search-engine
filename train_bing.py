from datasets import load_dataset
import pickle
import pandas as pd
from networks import SkipGramModel
from torch.utils.data import Dataset
import more_itertools
import torch
import wandb
from data_preprocessing import preprocess_wiki, create_lookup_tables_wiki, load_word_to_int, tokenize, clean_text

torch.set_printoptions(threshold=120_000)

ds = load_dataset("microsoft/ms_marco", "v1.1")
# df = ds.to_pandas()
# print(df)
# print(ds['train'].column_names)

df_train = pd.DataFrame(ds['train'])
# print(df_train[:11])

train_answers = df_train[['query', 'passages']]
train_answers = train_answers

num_of_rows = len(train_answers.index) - 1



def get_input_targets(tokens):
    inputs = []
    targets = []
    windows = list(more_itertools.windowed(tokens, 3))
    inputs = [w[1] for w in windows]
    targets = [[w[0], w[2]] for w in windows]
    return inputs, targets

def process_batch(row, word_to_int):
    query = row['query']
    passages = row['passages']['passage_text']
    passage_to_str = ' '.join(passages)

    cleaned_query = clean_text(query)
    cleaned_passages = clean_text(passage_to_str)

    query_tokens = tokenize(cleaned_query, word_to_int)
    passage_tokens = tokenize(cleaned_passages, word_to_int)

    i_passages, t_passages = get_input_targets(passage_tokens)

    if len(query_tokens) < 3:
        return pd.Series({"inputs": i_passages, "targets": t_passages})
    else:
        i_query, t_query = get_input_targets(query_tokens)
        return pd.Series({"inputs":i_query + i_passages, "targets": t_query + t_passages})

def load_dataset(df, word_to_int):
    print("turning dataset into input, output pairs...")
    training_data = df.apply(lambda row: process_batch(row, word_to_int), axis=1)
    print("num rows:", len(training_data.index))

    return training_data

# load weights from smaller wiki model
wiki_vocab = 63642
wiki_args = (wiki_vocab, 64, 2)
wiki_model = SkipGramModel.SkipGramFoo(*wiki_args)
wiki_model.load_state_dict(torch.load("weights/wiki-weights.pt", weights_only=True))

word_to_int = load_word_to_int()
vocab_size = len(word_to_int)

# create larger model and use old weights
args = (vocab_size, 64, 2)
model = SkipGramModel.SkipGramFoo(*args)
with torch.no_grad():
    model.emb.weight[:wiki_vocab] = wiki_model.emb.weight.clone()
print('model parameters: ', sum(p.numel() for p in model.parameters()))#

optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

training_data = load_dataset(train_answers, word_to_int)
BATCH_SIZE = 256
EPOCH_NUM = 1


model.to(device)
print("training...")
wandb.init(project='two-towers', name='bing-skip-gram')
for i in range(EPOCH_NUM):
    print(f"Epoch: {i}")
    for j in range(0, len(training_data), BATCH_SIZE):
        #print(j)
        batch = training_data.iloc[j:j + BATCH_SIZE]
        inputs = batch['inputs'].sum()
        #print(inputs)
        targets = batch['targets'].sum()
        inputs = torch.LongTensor(inputs)
        targets = torch.LongTensor(targets)

        inputs = inputs.to(device)
        targets = targets.to(device)

        rand = torch.randint(0, len(word_to_int), (inputs.size(0), 2)).to(device)
        optimizer.zero_grad()
        loss = model(inputs, targets, rand)
        print(f"loss: {loss.item()}")
        wandb.log({"loss": loss.item()})
        loss.backward()
        optimizer.step()

        if (j+1 % 25) == 0:
            print("saving")
            torch.save(model.state_dict(), f'./weights/bing_tuned_weights-{j}.pt')

# save model
print("saving")
torch.save(model.state_dict(), './weights/fine_tuned_weights.pt')
print('Uploading...')
artifact = wandb.Artifact('model-weights', type='model')
artifact.add_file('./weights/fine_tuned_weights.pt')
wandb.log_artifact(artifact)
print('Done!')
wandb.finish()


