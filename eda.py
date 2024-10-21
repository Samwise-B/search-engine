import pandas as pd

# Load the Parquet file
# df = pd.read_parquet('nested_data_example.parquet')
df = pd.read_parquet('../Data/train-00000-of-00001.parquet')

print(df.info())

# Example of accessing nested dictionary fields inside the 'passages' column
# Assuming the 'passages' column contains dictionaries

# Extract the 'passage_text' from each dictionary in the 'passages' column
df['passage_text'] = df['passages'].apply(lambda x: x['passage_text'] if 'passage_text' in x else None)

# Extract the 'is_selected' field from each dictionary
df['is_selected'] = df['passages'].apply(lambda x: x['is_selected'] if 'is_selected' in x else None)

# Show the resulting dataframe
# print(df[['query', 'passage_text', 'is_selected']])

# count = 0

# for passage in df['passage_text']:
#     if passage.size == 0:
#         print("EMPTY.")
#         count+=1
#     else:
#         print("oK")
# print("Empty passages: ", count)

# null_count = df['query'].isna().sum()
# print(f"Number of null values: {null_count}")

