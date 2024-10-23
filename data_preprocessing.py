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
    text = text.replace("'", ' <SINGLE QUOTATION_MARK> ')
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
        tokens.append(word_to_int.get(word, len(word_to_int)))

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