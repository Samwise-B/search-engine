from datasets import load_dataset
import pandas as pd
import random
import pickle

ds = load_dataset("microsoft/ms_marco", "v1.1")
# df = ds.to_pandas()
# print(df)
# print(ds['train'].column_names)

df_train = pd.DataFrame(ds['train'])
# print(df_train[:11])

train_answers = df_train[['query', 'passages']]

num_of_rows = len(train_answers.index) - 1
# print(type(train_answers['passages']))

# list_of_tuples = [] # [(query_text, relevant_text, irrelevant_text)]
# for passage in train_answers['passages']['passage_text']:
#     print(passage)

# TODO Data cleaning


def clean_text(text):
    text = text.lower()
    text = text.replace('\n', ' ')
    text = text.replace('\r', ' ')
    text = text.replace('\t', ' ')
    text = text.replace('.', ' <PERIOD> ')
    text = text.replace(',', ' <COMMA> ')
    text = text.replace('"', ' <QUOTATION_MARK> ')
    text = text.replace(';', ' <SEMICOLON> ')
    text = text.replace('!', ' <EXCLAMATION_MARK> ')
    text = text.replace('?', ' <QUESTION_MARK> ')
    text = text.replace('(', ' <LEFT_PAREN> ')
    text = text.replace(')', ' <RIGHT_PAREN> ')
    text = text.replace(' - ', ' <HYPHEN> ')
    text = text.replace(' – ', ' <HYPHEN2> ')
    text = text.replace('--', ' <HYPHENS> ')
    text = text.replace(':', ' <COLON> ')
    text = text.replace('"', ' <QUOTATION_MARK> ')
    text = text.replace("'", ' <SINGLE QUOTATION_MARK> ')
    # text = ascii(text)
    words = text.split(" ")

    return list(filter(None, words))

# def add_words_to_dict(words, word_to_int):
#     for word in words:
#         word_to_int.get(word, counter)

def get_lookup_table(words):
    word_to_int = {}
    word_counts = {}
    count = 1
    for word in words:
        word_counts[word] = word_counts.get(word, 0) + 1
        if not word_to_int.get(word):
            word_to_int[word] = count
            count += 1
    return word_to_int, word_counts


def clean_data(df):
    print("cleaning data ...")
    text_list = []
    for index, row in df.iterrows():
        if (index+1) % 1000 == 0:
            print("row", index)
        text_list = text_list + clean_text(row['query'])
        passages = row['passages']
        text = ''
        for passage in passages['passage_text']:
            # text += passage.strip()
            text_list = text_list + clean_text(passage)
            # print(clean_text(passage))
            # print(passage)

    return text_list


# probably need to call this on other columns too
text_words = clean_data(train_answers)
word_to_int, word_counts = get_lookup_table(text_words)
# save word_to_int and word_counts
with open("dictionaries/bing_word_to_int.pkl") as file:
    pickle.dump(word_to_int, file)

with open("dictionaries/bing_word_counts.pkl") as file:
    pickle.dump(word_counts, file)

print(len(word_to_int))
sorted_word_counts = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)
print(sorted_word_counts[:10])
# passages = train_answers['passages']

# clean_text(passages[0]['passage_text'][0])


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

    # print(random_index)
    # irrelevant_row = train_answers.iloc[random_index]
    # irrelevant_passage = irrelevant_row['passages']
    # irrelevant_passages = irrelevant_passage['passage_text']
    print(len(irrelevant_passages))
    list_of_tuples.append((query, passage_texts, irrelevant_passages))

    # print(f"Row {index}:")
    # for i, passage in enumerate(passage_texts):
    #     #print(f"  Passage: {passage}, Selected: {is_selected[i]}")
    #     list_of_tuples.append((query, passage))

# print("number of tuples")
# print(len(list_of_tuples))
# print("query")
# print(list_of_tuples[0][0])
# print("relevant passages")
# print(len(list_of_tuples[0][1]))
# print("irrelevant passages")
# print(len(list_of_tuples[0][2]))

