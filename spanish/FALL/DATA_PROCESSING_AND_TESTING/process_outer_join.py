import pickle
import json
import codecs
import numpy as np
import os
import pandas as pd
from nltk.tokenize import word_tokenize
import argparse
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

df = pd.read_csv('consolidated_emotions_2022.csv', lineterminator='\n')
df2 = pd.read_csv('sentiment_labels_2022.csv', lineterminator='\n')

df = df.sort_values(by=['date', 'tweet']).reset_index(drop=True)
df2 = df2.sort_values(by=['date', 'tweet']).reset_index(drop=True)

df3 = pd.concat([df.reset_index(drop=True), df2.reset_index(drop=True)], axis=1, join='outer')

df3.to_csv('outer_join_test.csv', index=False)

df4 = pd.read_csv('outer_join_test.csv', lineterminator='\n')