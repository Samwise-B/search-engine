import collections
import pickle

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
    text = text.replace('[', ' <LEFT_PAREN_SQUARE> ')
    text = text.replace(']', ' <RIGHT_PAREN_SQUARE> ')
    text = text.replace('-', ' <HYPHEN> ')
    text = text.replace('–', ' <HYPHEN2> ')
    text = text.replace('--', ' <HYPHENS> ')
    text = text.replace(':', ' <COLON> ')
    text = text.replace('"', ' <QUOTATION_MARK> ')
    text = text.replace("'", ' <SINGLE_QUOTATION_MARK> ')
    text = text.replace("*", ' <ASTERISK> ')
    text = text.replace("/", ' <FORWARD_SLASH> ')
    text = text.replace("\\", ' <BACKWARD_SLASH> ')
    text = text.replace("‘", ' <QUOTATION_TWO> ')
    text = text.replace("“", ' <QUOTATION_THREE> ')
    text = text.replace("”", ' <QUOTATION_FOUR> ')
    text = text.replace("=", ' <EQUALS> ')
    text = text.replace("$", ' <DOLLAR> ')
    text = text.replace("£", ' <POUND> ')
    # text = ascii(text)
    words = text.split(" ")

    return list(filter(None, words))

def tokenize(words, word_to_int):
    tokens = []
    for word in words:
        tokens.append(word_to_int.get(word, 0))

    return tokens

def tokenize_string(text, word_to_int):
   words = clean_text(text)
   tokens = tokenize(words, word_to_int)
   return tokens


def preprocess_wiki(text: str) -> list[str]:
  text = text.lower()
  text = text.replace('.',  ' <PERIOD> ')
  text = text.replace(',',  ' <COMMA> ')
  text = text.replace('"',  ' <QUOTATION_MARK> ')
  text = text.replace(';',  ' <SEMICOLON> ')
  text = text.replace('!',  ' <EXCLAMATION_MARK> ')
  text = text.replace('?',  ' <QUESTION_MARK> ')
  text = text.replace('(',  ' <LEFT_PAREN> ')
  text = text.replace(')',  ' <RIGHT_PAREN> ')
  text = text.replace('--', ' <HYPHENS> ')
  text = text.replace('?',  ' <QUESTION_MARK> ')
  text = text.replace(':',  ' <COLON> ')
  words = text.split()
  stats = collections.Counter(words)
  words = [word for word in words if stats[word] > 5]
  return words

def create_lookup_tables_wiki(words: list[str]) -> tuple[dict[str, int], dict[int, str]]:
  word_counts = collections.Counter(words)
  vocab = sorted(word_counts, key=lambda k: word_counts.get(k), reverse=True)
  int_to_vocab = {ii+1: word for ii, word in enumerate(vocab)}
  int_to_vocab[0] = '<PAD>'
  vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}
  return vocab_to_int, int_to_vocab

def load_word_to_int():
    with open("dictionaries/bing_word_to_int.pkl", "rb") as file:
        word_to_int = pickle.load(file)
    return word_to_int

def pad_batch(batch):
    max_lengths = batch.apply(lambda row: max(len(row['query']), len(row['relevant']), len(row['irrelevant'])), axis=1)
    batch_max_length = max(max_lengths)

    padded_batch = batch.apply(lambda row: pad_row_right(row, batch_max_length), axis=1)
    #print(padded_batch.head())
    return padded_batch['query'].tolist(), padded_batch['relevant'].tolist(), padded_batch['irrelevant'].tolist()

def pad_row_left(row, max_length, pad_value=0):
    query = ([pad_value] * (max_length - len(row['query']))) + row['query']
    relevant = ([pad_value] * (max_length - len(row['relevant']))) + row['relevant']
    irrelevant = ([pad_value] * (max_length - len(row['irrelevant']))) + row['irrelevant']
    """Pads a list with `pad_value` until it reaches `max_length`."""
    return pd.Series({'query': query, "relevant": relevant, "irrelevant": irrelevant})

def pad_row_right(row, max_length, pad_value=0):
    query = row['query'] + ([pad_value] * (max_length - len(row['query'])))
    relevant = row['relevant'] + ([pad_value] * (max_length - len(row['relevant'])))
    irrelevant = row['irrelevant'] + ([pad_value] * (max_length - len(row['irrelevant'])))
    """Pads a list with `pad_value` until it reaches `max_length`."""
    return pd.Series({'query': query, "relevant": relevant, "irrelevant": irrelevant})