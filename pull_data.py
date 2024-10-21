import os
import pandas as pd
# Visualize the distribution of passage lengths
import matplotlib.pyplot as plt
import tqdm
import collections
import more_itertools
import wandb
import torch
import pandas as pd
import psycopg2
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
import json



pq = pd.read_parquet('../Data/train-00000-of-00001.parquet')
# Keep only the 'query' column
df_query_passages = pq[['query','passages']]
# Save it to a CSV if needed
df_query_passages.to_csv('../Data/query_and_passages_only.csv', index=False)
