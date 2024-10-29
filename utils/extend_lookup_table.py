from datasets import load_dataset
import pandas as pd
import random
import pickle
from data_preprocessing import preprocess_wiki, create_lookup_tables_wiki, clean_text

ds = load_dataset("microsoft/ms_marco", "v1.1")
df_train = pd.DataFrame(ds['train'])
train_answers = df_train[['query', 'passages']]
num_of_rows = len(train_answers.index) - 1


def get_extended_lookup_table(text_words, word_to_int):
    word_counts = {}

    # Count the occurrences of each word
    for item in text_words[0]:
        for word in item:
            word_counts[word] = word_counts.get(word, 0) + 1

    # Filter out words with a count less than 5
    filtered_word_counts = {word: count for word,
                            count in word_counts.items() if count > 5}

    # Load wiki word lookup table
    print("wiki_lookup", list(word_to_int.items())[:100])
    wiki_vocab_size = len(word_to_int)
    print(f"wiki size: {wiki_vocab_size}")

    # extend to marco dataset
    bing_word_to_int = {}
    counter = wiki_vocab_size
    for word in filtered_word_counts.keys():
        if word_to_int.get(word) is None and bing_word_to_int.get(word) is None:
            bing_word_to_int[word] = counter
            counter += 1
    

    print("bing lookup", list(bing_word_to_int.items())[-100:])
    print(f"bing vocab size: {len(bing_word_to_int)}")
    
    word_to_int.update(bing_word_to_int)
    print(f"combined vocab size: {len(word_to_int)}")

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
# print(len(wiki_lookup))

word_to_int, word_counts = get_extended_lookup_table(text_words, wiki_lookup)

# save extended word_to_int and word_counts
# done!
with open("dictionaries/bing_word_to_int.pkl", "wb") as file:
    pickle.dump(word_to_int, file)

with open("dictionaries/bing_word_counts.pkl", "wb") as file:
    pickle.dump(word_counts, file)