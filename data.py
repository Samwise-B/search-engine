from datasets import load_dataset
import pandas as pd
import random

ds = load_dataset("microsoft/ms_marco", "v1.1")
# df = ds.to_pandas()
# print(df)
# print(ds['train'].column_names)

df_train = pd.DataFrame(ds['train'])
# print(df_train[:11])

train_answers = df_train[['query', 'passages']][:11]

num_of_rows = len(train_answers.index) - 1 
#print(type(train_answers['passages']))

# list_of_tuples = [] # [(query_text, relevant_text, irrelevant_text)]
# for passage in train_answers['passages']['passage_text']:
#     print(passage)

def get_random_text(index):
    random_index = random.randint(0, num_of_rows)
    while (random_index == index):
        random_index = random.randint(0, num_of_rows)

    irrelevant_row = train_answers.iloc[random_index]
    irrelevant_item = irrelevant_row['passages']
    irrelevant_passages = irrelevant_item['passage_text']

    random_passage_index = random.randint(0, len(irrelevant_passages)-1)
    random_passage = irrelevant_passages[random_passage_index]
    return random_passage
    
list_of_tuples = []

for index, row in train_answers.iterrows():
    query = row['query']
    passages = row['passages']

    is_selected = passages['is_selected']
    passage_texts = passages['passage_text']
    
    irrelevant_passages = []
    for i in range(10):
        irrelevant_passages.append(get_random_text(index))
    
    #print(random_index)
    # irrelevant_row = train_answers.iloc[random_index]
    # irrelevant_passage = irrelevant_row['passages']
    # irrelevant_passages = irrelevant_passage['passage_text']
    print(len(irrelevant_passages))
    list_of_tuples.append((query, passage_texts, irrelevant_passages))

    

    # print(f"Row {index}:")
    # for i, passage in enumerate(passage_texts):
    #     #print(f"  Passage: {passage}, Selected: {is_selected[i]}")
    #     list_of_tuples.append((query, passage))

print("number of tuples")
print(len(list_of_tuples))
print("query")
print(list_of_tuples[0][0])
print("relevant passages")
print(len(list_of_tuples[0][1]))
print("irrelevant passages")
print(len(list_of_tuples[0][2]))

# df_test = pd.DataFrame(ds['test'])
# print(df_test.head())

# df_validation = pd.DataFrame(ds['validation'])
# print(df_validation.head())

# TODO negative sampling
# query, valid document, invalid documents
# take in a query --> eventually randomise the negative documents
