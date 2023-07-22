######
# Lexicon-based Emotion Detection
#
# Use the dictionary to label each labeled data set.
# Make sure you preprocess your data if you need to standardize it.
# Determine how accurate the dictionary model is for each labeled data set.
# Look at what is mislabeled and see if you notice any patterns.
# Redo the task by adding in an emoji dictionary as well. [see how dictionary works]
# Will punctuation help your task? If so, add in a punctuation dictionary.
#
# Eliza, Apollo, Kate
#####

#imports
import sys
sys.path += ['../', '../..']
import pandas as pd
import re
import json
import numpy as np
import time
import random
# from nrclex import NRCLex
from NRCLex import NRCLex
from nltk.tokenize import RegexpTokenizer
import argparse
from collections import Counter
from helper_funcs import *

EMOTIONS = ['anger', 'fear', 'sadness', 'disgust', 'joy']
EMOTIONS2 = ['anger', 'fear', 'sadness', 'joy']

def tsv_to_json(tsv_file):
    #Converts tsv file to json file that can be used by the NRCLex 4.0 code 
    df = pd.read_csv(tsv_file, sep='\t', encoding = 'utf8')
    dict = {}
    for index, row in df.iterrows():
        emotion_list = []
        for emotion in EMOTIONS:
            if row[emotion] == 1:
                emotion_list.append(emotion)  
        dict[row['Ukrainian Word']] = emotion_list      

    with open('Ukrainian-NRC-EmoLex.json', 'w') as outfile:
        json.dump(dict, outfile, indent=4, ensure_ascii=False)

def tokenize(tweets):
    #Tokenizes tweets using nltk
    regexp = RegexpTokenizer('\w+')
    tweets = tweets.map(str)
    token_col = tweets.apply(regexp.tokenize)
    return token_col

def emotion_classification(data, combined=False):
    #Uses NRCLex to return list of non-zero emotions detected in text
    arr = []
    for tweet in data['processed_tweets']:
        text_object = NRCLex(tweet).affect_list
        arr.append(text_object)
    if combined:
        arr = combine_and_delete_dups(arr)
    return arr

def emotion_classification2(data, combined=False):
    #Uses NRCLex to return list of top or tied emotions detected in text
    arr = []
    for tweet in data['processed_tweets']:
        text_object = NRCLex(tweet).top_emotions
        score_list_max = max([item[1] for item in text_object])
        top_emotions = [item[0] for item in text_object if (item[1] == score_list_max and item[1] != 0)]
        arr.append(top_emotions)
    if combined:
        arr = combine_and_delete_dups(arr)
    return arr

def emotion_classification_threshold(data, combined=False):
    #Uses NRCLex and iterates through different emotion prevalence thresholds
    thresholds = [0.3, 0.5, 0.7, 0.9, 1]
    threshold_preds = pd.DataFrame()
    for t in thresholds:
        arr = []
        for tweet in data['processed_tweets']:
            text_object = NRCLex(tweet).top_emotions
            top_emotions = [item[0] for item in text_object if item[1] >= t]
            arr.append(top_emotions)
            if combined:
                arr = combine_and_delete_dups(arr)
        threshold_preds[str(t)] = arr
    return(threshold_preds)
        
def accuracy(labels, predicted_labels, combined=False):
    '''
    input:
        labels: the actual labels of the tweets
        predicted_labels: the predicted emotion labels
        combined: whether anger and disgust are combined
    '''
    if combined:
        emotion_lst = EMOTIONS2
    else:
        emotion_lst = EMOTIONS
    
    total_tweets = len(predicted_labels)
    #store accuracy data
    dict = {}

    #random guess
    random_guesses = [random.choice(emotion_lst) for i in range(len(labels))]

    for emotion in emotion_lst:
        accuracy_count = 0
        emotion_count = 0
        random_count = 0
        fp = 0
        fn = 0
        tp = 0
        for true_label, predicted_label, random_guess in zip(labels, predicted_labels, random_guesses):
            if random_guess == true_label:
                random_count +=1
            if emotion == true_label:
                emotion_count +=1
                tp +=1
            if emotion in predicted_label and emotion == true_label:
                accuracy_count += 1
                random_count += 1
            if emotion not in predicted_label and emotion != true_label:
                accuracy_count += 1
            if emotion in predicted_label and emotion != true_label:
                fp +=1
            if emotion not in predicted_label and emotion == true_label:
                fn +=1
        dict[emotion]= {'accuracy': round(accuracy_count / total_tweets, 3),
                        'prevalence' : round(emotion_count / total_tweets, 3),
                        'f1 score' : round(tp/(tp + 0.5*(fp+fn)), 3),
                        'precision' : round(tp/(tp+fp), 3),
                        'recall' : round(tp/(tp+fn), 3),
                        'random_accuracy' : round(random_count / total_tweets, 3)}
    return(dict)


def accuracy_thresholds(labels, predicted_labels, combined=False):
    # same as previous accuracy but with different thresholds
    if combined:
        emotion_lst = EMOTIONS2
    else:
        emotion_lst = EMOTIONS
    
    total_tweets = len(predicted_labels)
    first_col = emotion_lst + ['average_accuracy', 'coverage_pct']
    accuracy_df = pd.DataFrame(data={'emotion': first_col})
    for col in predicted_labels.columns:
        count_nonempty = len([i for i in predicted_labels[col] if len(i) > 0])
        label_accuracies = []
        for emotion in emotion_lst:
            accuracy_count = 0
            for true_label, predicted_label in zip(labels, predicted_labels[col]):
                if emotion == true_label and emotion in predicted_label:
                    accuracy_count +=1
                if emotion not in predicted_label and emotion != true_label:
                    accuracy_count +=1
            label_accuracies.append(round(accuracy_count/total_tweets, 3))
        label_accuracies.append(round(np.mean(label_accuracies), 3))
        label_accuracies.append(round(count_nonempty/total_tweets,3))
        accuracy_df[col] = label_accuracies
    return(accuracy_df)

def main():
    #get arguments
    parser = argparse.ArgumentParser(description="Emotion Detection Lexicon")
    parser.add_argument('--emojis', help="True to encode emojis, False to not encode emojis", default=False)
    parser.add_argument("--combine", help="True to combine anger and disgust", default=False)
    parser.add_argument("--v", help="1: all nonzero emotion labels, 2: top emotion label, 3: top emotion labels with thresholds", default='1')
    parser.add_argument("--file", help='Tsv file where data is stored', default='../data/ukrainian_emotion_new.tsv')
    args = parser.parse_args()

    #convert to bool
    if args.emojis == 'True':
        emojis = True
    else:
        emojis = False

    if args.combine == 'True':
        combine = True
    else:
        combine = False

    tweet_df = preprocess(args.file, encode_emojis=emojis)
    if combine:
        tweet_df = combine_emotions(tweet_df)
    if args.v == '1':
        emotions = emotion_classification(tweet_df, combined=combine)
    elif args.v == '2':
        emotions = emotion_classification2(tweet_df, combined=combine)
    elif args.v == '3':
        emotions = emotion_classification_threshold(tweet_df, combined=combine)

    if args.v == '1' or args.v == '2':
        accuracy_data = accuracy(tweet_df['emotion'], emotions, combined=combine)
    elif args.v == '3':
        accuracy_data = accuracy_thresholds(tweet_df['emotion'], emotions, combined=combine)

    print(accuracy_data)


if __name__ == "__main__":
    main()
    
