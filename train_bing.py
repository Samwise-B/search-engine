from datasets import load_dataset
import pickle
import pandas as pd
from data import clean_text
from torch.utils.data import Dataset
import more_itertools

ds = load_dataset("microsoft/ms_marco", "v1.1")
# df = ds.to_pandas()
# print(df)
# print(ds['train'].column_names)

df_train = pd.DataFrame(ds['train'])
# print(df_train[:11])

train_answers = df_train[['query', 'passages']]
train_answers = train_answers

num_of_rows = len(train_answers.index) - 1

def load_word_to_int():
    with open("dictionaries/bing_word_to_int.pkl", "rb") as file:
        word_to_int = pickle.load(file)
    return word_to_int

def tokenize(words, word_to_int):
    tokens = []
    for word in words:
        tokens.append(word_to_int.get(word, len(word_to_int)))

    return tokens

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

    i_query, t_query = get_input_targets(query_tokens)
    i_passages, t_passages = get_input_targets(passage_tokens)
    return i_query + i_passages, t_query + t_passages

def loop_dataset(df):
    word_to_int = load_word_to_int()
    training_data = df.apply(lambda row: process_batch(row, word_to_int), axis=1)
    print("input")
    print(len(training_data[0][0]))
    print("target")
    print(len(training_data[0][1]))
    print("num rows:", len(training_data.index))

    return training_data

training_data = loop_dataset(train_answers)
BATCH_SIZE = 256
EPOCH_NUM = 1

for i in range(EPOCH_NUM):

    for j in range(0, len(training_data), BATCH_SIZE):
        batch = training_data.iloc[j:j + BATCH_SIZE]
        inputs = batch[0][:]
        targets = batch[:][1].sum()

