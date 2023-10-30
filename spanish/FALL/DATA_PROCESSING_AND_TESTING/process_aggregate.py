import pickle
import json
import codecs
import numpy as np
import os
import pandas as pd
from nltk.tokenize import word_tokenize
import datetime
import argparse
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys

pd.set_option('display.max_rows', 999)
pd.set_option('display.max_columns', 999)
pd.set_option('display.width', 999)

# processing before aggregation
'''df = pd.read_csv('./emotion_sentiment_outer_join_2022.csv', lineterminator='\n')

df['anger'] = 0
df['joy'] = 0
df['sadness'] = 0
df['fear'] = 0
df['pos'] = 0
df['neg'] = 0

for i, (date, emotion, sentiment) in tqdm(enumerate(zip(df['date'], df['emotion'], df['sentiment']))):
    if emotion == 'anger':
        df.loc[i, 'anger'] = 1
    elif emotion == 'joy':
        df.loc[i, 'joy'] = 1
    elif emotion == 'sadness':
        df.loc[i, 'sadness'] = 1
    elif emotion == 'fear':
        df.loc[i, 'fear'] = 1

    if sentiment == 1:
        df.loc[i, 'pos'] = 1
    elif sentiment == 0:
        df.loc[i, 'neg'] = 1

df = df.drop(['date2', 'tweet2', 'city2', 'emotion', 'sentiment'], axis=1) 

df.to_csv('for_agg.csv', index=False) '''

# daily aggregation
''' df = pd.read_csv('for_agg.csv', lineterminator='\n')
city_df = pd.read_csv('./Venezuela_cities.csv')
city_df = city_df.drop(['column1', 'name_translated', 'name_dbpedia', 'population'], axis=1).reset_index(drop=True)
df = df.merge(city_df, on='city', how='left')

agg_df = df.groupby(['date', 'country']).agg({'anger':'sum', 'joy':'sum', 'sadness':'sum', 'fear':'sum', 'pos':'sum', 'neg':'sum'}).reset_index()
agg_df.to_csv('daily_aggregated_data.csv', index=False)
print(agg_df) '''

# monthly aggregation
'''df = pd.read_csv('daily_aggregated_data.csv', lineterminator='\n')

df['day'] = 0
df['month'] = 0
split =  df['date'].str.split('-', expand=True)
df['day'], df['month'] = split[2], split[1]

monthly_aggregated_data = df.groupby(['month', 'country']).agg({'anger':'max', 'joy':'max', 'sadness':'max', 'fear':'max', 'pos':'max', 'neg':'max'})

monthly_aggregated_data.to_csv('monthly_aggregated_data_max.csv') '''

# getting sum, max, and avg
''' df1 = pd.read_csv('monthly_aggregated_data_mean.csv', lineterminator='\n')
df2 = pd.read_csv('monthly_aggregated_data_max.csv', lineterminator='\n')
df2 = df2.drop(['month', 'country'], axis=1)
df3 = pd.read_csv('monthly_aggregated_data_sum.csv', lineterminator='\n')
df3 = df3.drop(['month', 'country'], axis=1)
df4 = pd.concat([df1, df2, df3], axis=1)
df4['country_flow_data'] = 0
df4.to_csv('monthly_aggregated_data_all.csv', index=False) '''