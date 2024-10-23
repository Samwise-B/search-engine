from datasets import load_dataset
import pandas as pd
import random
import pickle
from data_preprocessing import preprocess_wiki, create_lookup_tables_wiki, clean_text

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




# def add_words_to_dict(words, word_to_int):
#     for word in words:
#         word_to_int.get(word, counter)


# def get_lookup_table(words):

#     word_to_int = {}
#     word_counts = {}
#     count = 1
#     for word in words[0]:
#         if word is not None:
#             word_counts[word] = word_counts.get(word, 0) + 1
#             if not word_to_int.get(word):
#                 word_to_int[word] = count
#                 count += 1
#     return word_to_int, word_counts


def get_lookup_table(text_words, word_to_int):
    word_counts = {}

    # Count the occurrences of each word
    for item in text_words[0]:
        for word in item:
            word_counts[word] = word_counts.get(word, 0) + 1

    # Filter out words with a count less than 5
    filtered_word_counts = {word: count for word,
                            count in word_counts.items() if count > 5}

    # Create the lookup table (word to index)
    #word_to_int = wiki_word_to_int
    wiki_vocab_size = len(word_to_int)
    # word_to_int = {word: idx + wiki_vocab_size for idx,
    #                word in enumerate(filtered_word_counts.keys()) 
    #                if word_to_int.get(word, "missing") == "missing"}
    bing_word_to_int = {word: idx for idx,
                   word in enumerate(filtered_word_counts.keys())}
    
    word_to_int.update(bing_word_to_int)

    return word_to_int, filtered_word_counts


def clean(rows):
    query = rows['query']
    passages = rows['passages']['passage_text']
    passage_to_str = ' '.join(passages)
    cleaned_query = clean_text(query)
    cleaned_passages = clean_text(passage_to_str)
    return cleaned_query + cleaned_passages


def clean_data(df):
    print("cleaning data...")
    text_list = []
    text_cleaned = df.apply(lambda row: clean(row), axis=1)
    print(len(text_cleaned))
    text_list.append(text_cleaned)
    return text_list


# probably need to call this on other columns too
# text_words = clean_data(train_answers)
# word_to_int, word_counts = get_lookup_table(text_words)
text_words = clean_data(train_answers)

# print(type(text_words))

with open('data/text8') as f: text8: str = f.read()

corpus: list[str] = preprocess_wiki(text8)
wiki_lookup, ids_to_words = create_lookup_tables_wiki(corpus)
print(len(wiki_lookup))

word_to_int, word_counts = get_lookup_table(text_words, wiki_lookup)

# print(word_to_int.sort())
# print(word_counts.sort().head())

# save word_to_int and word_counts
# done!
with open("dictionaries/bing_word_to_int.pkl", "wb") as file:
    pickle.dump(word_to_int, file)

with open("dictionaries/bing_word_counts.pkl", "wb") as file:
    pickle.dump(word_counts, file)


# print(len(word_to_int))
# sorted_word_counts = sorted(
#     word_counts.items(), key=lambda item: item[1], reverse=True)
# print(sorted_word_counts[:10])
# passages = train_answers['passages']

# clean_text(passages[0]['passage_text'][0])


# def get_random_text(index):
#     random_index = random.randint(0, num_of_rows)
#     while (random_index == index):
#         random_index = random.randint(0, num_of_rows)

#     irrelevant_row = train_answers.iloc[random_index]
#     irrelevant_item = irrelevant_row['passages']
#     irrelevant_passages = irrelevant_item['passage_text']

#     random_passage_index = random.randint(0, len(irrelevant_passages)-1)
#     random_passage = irrelevant_passages[random_passage_index]
#     return random_passage


# list_of_tuples = []

# for index, row in train_answers.iterrows():
#     query = row['query']
#     passages = row['passages']

#     is_selected = passages['is_selected']
#     passage_texts = passages['passage_text']

#     irrelevant_passages = []
#     for i in range(10):
#         irrelevant_passages.append(get_random_text(index))

#     # print(random_index)
#     # irrelevant_row = train_answers.iloc[random_index]
#     # irrelevant_passage = irrelevant_row['passages']
#     # irrelevant_passages = irrelevant_passage['passage_text']
#     # print(len(irrelevant_passages))
#     list_of_tuples.append((query, passage_texts, irrelevant_passages))

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
