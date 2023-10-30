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

directory = "/Users/bernardmedeiros/Desktop/NewFullFallLabeledTweets2023"

list = []

spanish_tweet = []
location_list = []
date_list = []
pred_list = []

for filename in tqdm(os.listdir(directory)):
    f = pd.read_csv(os.path.join(directory, filename), lineterminator='\n')      
    for predicted_emotion, city, date_to_append, line in zip(f['predicted_emotion'], f['location'], f['date'], f['preprocessed_text']):
        file_date_str = os.path.basename(filename).split('_')[0]
        file_date = pd.to_datetime(file_date_str, format='%Y-%m-%d')
        #if len(f) == 0:
            #continue
        #text = process(line)
        '''if line != None:'''
        spanish_tweet.append(line)
        location_list.append(city)
        pred_list.append(predicted_emotion)
        date_list.append(date_to_append)

final_df = pd.concat([pd.DataFrame(date_list).reset_index(drop=True), pd.DataFrame(location_list).reset_index(drop=True), pd.DataFrame(spanish_tweet).reset_index(drop=True), pd.DataFrame(pred_list).reset_index(drop=True)], axis=1)
final_df.to_csv("consolidated_emotions_2022.csv", index=False, header=False)