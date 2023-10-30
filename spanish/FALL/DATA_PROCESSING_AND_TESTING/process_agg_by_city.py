'''

Some cities were removed from final data. If proportion of city population to number of tweets seemed unreasonable, city was removed. These cities are logged here.
--Barcelona (370,000 tweets vs. small municipality w/ less than 1,000 inhabitants) (source: https://citypopulation.de/en/colombia/meta/villavicencio/50001020__barcelona/)
--Fundaciôn (100,000 tweets vs. population of 8,000)
--Fuerte (unclear what population is being referred to)
--Granada (population of 80,000 vs 70,000 tweets)
--Garzón (means butler)
--Santa Cruz (population of 30,000 vs. 50,000 tweets)
--Rubio (means blonde)
--La Plata (35,000 tweets vs. population of 60,000 -- also, same name as a major city in Argentina)
--Marino (mistranslation)
--Pamplona (refers to a city in Spain)

'''

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
import matplotlib.pyplot as plt
import sys

sys.stdout = open('new_output.txt', 'a')

exceptions_list = ['Barcelona', 'Fundación', 'Fuerte', 'Granada', 'Garzón', 'Santa Cruz', 'Rubio', 'Marino', 'Pamplona']

country_list = ['Barbados', 'Brazil', 'Colombia', 'Cuba', 'Curaçao', 'Dominican Republic', 'Haiti', 'Peru', 'Puerto Rico', 'Trinidad and Tobago', 'Venezuela']

city_df = pd.read_csv('/Users/bernardmedeiros/Desktop/ToRun/FinalData/Venezuela_cities.csv')
city_df = city_df.drop(['column1', 'name_translated', 'name_dbpedia', 'population'], axis=1).reset_index(drop=True)

#print(city_df)

df = pd.read_csv('/Users/bernardmedeiros/Desktop/ToRun/FinalData/emotion_sentiment_outer_join_2022.csv', lineterminator='\n')
df = df.merge(city_df, on='city', how='left')


df['day'] = 0
df['month'] = 0
split =  df['date'].str.split('-', expand=True)
df['day'], df['month'] = split[2], split[1]

gm = df.groupby(['month'])

df_jan = gm.get_group('01')
df_feb = gm.get_group('02')
df_mar = gm.get_group('03')
df_apr = gm.get_group('04')
df_may = gm.get_group('05')
df_jun = gm.get_group('06')
df_jul = gm.get_group('07')
df_aug = gm.get_group('08')
df_sep = gm.get_group('09')
df_oct = gm.get_group('10')
df_nov = gm.get_group('11')
df_dec = gm.get_group('12')

pos = 0
neg = 0
anger = 0
fear = 0
joy = 0
sadness = 0
others = 0

df_list = [df_jan, df_feb, df_mar, df_apr, df_may, df_jun, df_jul, df_aug, df_sep, df_oct, df_nov, df_dec]
month_list = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October','November', 'December']

for df, month in zip(df_list, month_list):

    for country in country_list:

        for sentiment, emotion, country_row, city in tqdm(zip(df['sentiment'], df['emotion'], df['country'], df['city'])):
                if country_row == country and str(city) not in exceptions_list:
                    if sentiment == 1:
                        pos += 1
                    elif sentiment == 0:
                        neg += 1
                    
                    if emotion == 'anger':
                        anger += 1
                    elif emotion == 'fear':
                        fear += 1
                    elif emotion == 'joy':
                        joy += 1
                    elif emotion == 'sadness':
                        sadness += 1
                    elif emotion == 'others':
                        others += 1

        with open('new_output.txt', 'a') as f:
            print("Month: ", month)
            print("Country: ", country)
            print("Pos: ", pos)
            print("Neg: ", neg)
            print("Anger: ", anger)
            print("Fear: ", fear)
            print("Joy: ", joy)
            print("Sadness: ", sadness)
            print("Others: ", others)
            print("---")

        pos = 0
        neg = 0
        anger = 0
        fear = 0
        joy = 0
        sadness = 0
        others = 0 

#df_jan.to_csv('jan_output.csv', index=False)

'''date_list = ['2022-02-01', '2022-03-01', '2022-04-01', '2022-05-01', '2022-06-01', '2022-07-01', '2022-08-01', '2022-09-01', '2022-10-01', '2022-11-01', '2022-12-01']

for limit_date in tqdm(date_list):
    for index, (row_date, city, emotion, sentiment) in tqdm(enumerate(zip(df['date'], df['city'], df['emotion'], df['sentiment']))):
        if pd.to_datetime(limit_date, format='%Y-%m-%d') > pd.to_datetime(row_date, format='%Y-%m-%d'):
            if sentiment == 1:
                pos += 1
            elif sentiment == 0:
                neg += 1
            
            if emotion == 'anger':
                anger += 1
            elif emotion == 'fear':
                fear += 1
            elif emotion == 'joy':
                joy += 1
            elif emotion == 'sadness':
                sadness += 1

            df.drop(index,axis=0,inplace=True)
    print("Date: ", limit_date, file=f)
    print("Pos: ", pos, file=f)
    print("Neg: ", neg, file=f)
    print("Anger ", anger, file=f)
    print("Fear ", fear, file=f)
    print("Joy ", joy, file=f)
    print("Sadness ", sadness, file=f)
    print("---", file=f)

#A = df['city'].value_counts()

#A.to_csv("output.csv",mode='a')

#with open("output.txt", "a") as f:
    #print(df['city'], file=f, mode='a')'''

