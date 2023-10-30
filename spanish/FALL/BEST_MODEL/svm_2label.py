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

# set up arg parser to allow user to pick target directory for labeling
parser = argparse.ArgumentParser(description="Binary SVM Pipeline")
parser.add_argument("--unlabeled_directory", type=str, default="UnlabeledTweetsColombia", help="Directory containing csv files for tweets to label. Default is '/Users/bernardmedeiros/Desktop/ToRun/UnlabeledTweetsColombia'")
parser.add_argument('--start_date', type=str, default='2023-01-01', help='The start date (YYYY-MM-DD) of the date range to process.')
parser.add_argument('--end_date', type=str, default='2023-12-31', help='The end date (YYYY-MM-DD) of the date range to process.')
parser.add_argument('--output_name', type=str, default='sentiment_labels', help='Name given to output csv with labeled tweets.')
args = parser.parse_args()

# i don't think this function is necessary with my new setup 
''' def process_sep_data_text_loc(line):
    #all_info = json.loads(line)
    tweet_text = process(line)
    #lang = all_info['lang']
    #if lang == "es":
    return tweet_text '''

def process(text):
    #text = str(text)
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    return text

def bert_transformer(transformer, sentences):
    model = SentenceTransformer(transformer)
    # Sentences are encoded by calling model.encode()
    embedding = model.encode(sentences)
    #print("Embedding: ", embedding)
    #embedding = []

    #for sentence, embed in zip(sentences, embeddings): 
    #embedding.append(embed)
    X = pd.DataFrame(embedding)
    
    return X

directory = args.unlabeled_directory
date_range = pd.date_range(args.start_date, args.end_date)
spanish_tweet = []
location_list = []
date_list = []
for filename in tqdm(os.listdir(directory)):
    f = pd.read_csv(os.path.join(directory, filename), lineterminator='\n')      
    for city, date_to_append, line in zip(f['location'], f['date'], f['preprocessed_text']):
        file_date_str = os.path.basename(filename).split('_')[0]
        file_date = pd.to_datetime(file_date_str, format='%Y-%m-%d')
        if args.start_date and file_date < date_range[0]:
            continue
        if args.end_date and file_date > date_range[-1]:
            continue
        #if len(f) == 0:
            #continue
        #text = process(line)
        '''if line != None:'''
        spanish_tweet.append(line)
        location_list.append(city)
        date_list.append(date_to_append)
#spanish_tweet = list(set(spanish_tweet))
print("total tweets:", len(spanish_tweet))

print("Loading model")
model = pickle.load(open("svm_2label.sav", 'rb'))
pred_list = []
print("Loaded")
for i in tqdm(range(0, len(spanish_tweet), 2048)):
    #print(tweet)
    token = bert_transformer("hiiamsid/sentence_similarity_spanish_es", spanish_tweet[i:i+2048])
    #print(token.shape)

    # transpose matrix
    prediction = model.predict(token)
    for item in prediction:
        pred_list.append(item)
    #print(prediction)

'''number_of_pos = int(sum(np.array(pred_list)))
number_of_tweets = len(prediction)
number_of_neg = number_of_tweets - number_of_pos
print("pos:", number_of_pos, "neg:", number_of_neg)'''
print(pred_list)

df_result = pd.concat([pd.DataFrame(date_list).reset_index(drop=True), pd.DataFrame(location_list).reset_index(drop=True), pd.DataFrame(spanish_tweet).reset_index(drop=True), pd.DataFrame(pred_list).reset_index(drop=True)], axis=1)
df_result.to_csv(args.output_name + '.csv', index=False, header=False)