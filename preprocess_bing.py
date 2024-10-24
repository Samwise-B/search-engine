from datasets import load_dataset
from data_preprocessing import load_word_to_int, clean_text, tokenize
import pandas as pd
import random
import pickle
# pd.set_option('display ')

ds = load_dataset("microsoft/ms_marco", "v1.1")
df_train = pd.DataFrame(ds['train'])
df_train = df_train[['query', 'passages']]
# train_answers = train_answers
num_of_rows = len(df_train.index) - 1


# load word_to_int
word_to_int = load_word_to_int()
print(list(word_to_int.items())[:10])
print(dict(word_to_int.items())['rba'])
vocab_dim = len(word_to_int)
print("vocab size:", vocab_dim)

# for index, row in df_train.iterrows():

df = pd.DataFrame()


def process_row(row, word_to_int):
    row_index = row.name
    query = row['query']
    passages = row['passages']

    clean_query = clean_text(query)
    tokenquery = tokenize(clean_query, word_to_int)

    relevant_list = []
    query_list = []
    irrelant_list = []
    for passage in passages['passage_text']:
        clean_passage = clean_text(passage)
        tokenpasage = tokenize(clean_passage, word_to_int)
        relevant_list.append(tokenpasage)
        query_list.append(tokenquery)
        irpassage = get_random_text(row_index)
        clean_ir_passage = clean_text(irpassage)
        token_irr_passage = tokenize(clean_ir_passage, word_to_int)
        irrelant_list.append(token_irr_passage)
    return query_list, relevant_list, irrelant_list


def get_random_text(index):
    random_index = random.randint(0, num_of_rows)
    while (random_index == index):
        random_index = random.randint(0, num_of_rows)

    irrelevant_row = df_train.iloc[random_index]
    irrelevant_item = irrelevant_row['passages']
    irrelevant_passages = irrelevant_item['passage_text']

    random_passage_index = random.randint(0, len(irrelevant_passages)-1)
    random_passage = irrelevant_passages[random_passage_index]
    return random_passage


query_list = []
revalent_list = []
irrelavant_list = []
for index, row in df_train.iterrows():
    q, r, ir = process_row(row, word_to_int)
    # print(index)
    query_list += q
    revalent_list += r
    irrelavant_list += ir
output_dict = pd.DataFrame({
    'query': query_list,
    'relevant': revalent_list,
    'irrelevant': irrelavant_list
})

print(output_dict.head(20))
output_dict.to_pickle("data/preprocess_bing.pkl")
#output_dict.to_csv('data/preprocess_bing.csv')
# create_frame = df_train.apply(lambda row: process_row(row), axis=1)
# print(create_frame.head())
