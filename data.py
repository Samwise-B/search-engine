from datasets import load_dataset
import pandas as pd

ds = load_dataset("microsoft/ms_marco", "v1.1")
# df = ds.to_pandas()
# print(df)
# print(ds['train'].column_names)

df_train = pd.DataFrame(ds['train'])
# print(df_train[:11])

train_answers = df_train[['query', 'passages']][:11]
print(train_answers['passages'][0])

# df_test = pd.DataFrame(ds['test'])
# print(df_test.head())

# df_validation = pd.DataFrame(ds['validation'])
# print(df_validation.head())

# TODO negative sampling
# query, valid document, invalid documents
# take in a query --> eventually randomise the negative documents
