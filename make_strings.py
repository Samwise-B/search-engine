import pandas as pd

# Load the Parquet file
df = pd.read_parquet('../Data/train-00000-of-00001.parquet')

print(df['query'][:100])
query_string = df['query'].str.cat(sep=" ")
print(query_string)
 




df['passage_text'] = df['passages'].apply(lambda x: x['passage_text'] if 'passage_text' in x else None)
print(df['passage_text'])

# Flatten the arrays in the 'passage_text' and join them into a single long string
df['passage_text_flat'] = df['passage_text'].apply(lambda x: " ".join(x.tolist()) if isinstance(x, np.ndarray) else "")

# Now concatenate all the flattened passage_text strings into one long string
passages_string = df['passage_text_flat'].str.cat(sep=" ")

# Print the final concatenated string
print(passages_string)

