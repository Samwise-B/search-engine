from datasets import load_dataset
import pandas as pd
from data_preprocessing import load_word_to_int, tokenize, clean_text
import random

ds = load_dataset("microsoft/ms_marco", "v1.1")
df_train = pd.DataFrame(ds['train'])
df_val = pd.DataFrame(ds['validation'])
df_test = pd.DataFrame(ds['test'])
df_all = pd.concat([df_train, df_val])
df_all = df_all[['query', 'passages']]
print(df_all[:11])
num_of_rows = len(df_val.index) - 1 



# start preprocessing the data
word_to_int = load_word_to_int()
print(dict(word_to_int.items())['rba'])
vocab_dim = len(word_to_int)
print("vocab size:", vocab_dim)

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

    irrelevant_row = df_val.iloc[random_index]
    irrelevant_item = irrelevant_row['passages']
    irrelevant_passages = irrelevant_item['passage_text']

    random_passage_index = random.randint(0, len(irrelevant_passages)-1)
    random_passage = irrelevant_passages[random_passage_index]
    return random_passage

query_list = []
revalent_list = []
irrelavant_list = []
for index, row in df_val.iterrows():
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
output_dict.to_pickle("data/df_val_preprocess_bing.pkl")
# create_frame = df_all.apply(lambda row: process_row(row), axis=1)
# print(create_frame.head())



