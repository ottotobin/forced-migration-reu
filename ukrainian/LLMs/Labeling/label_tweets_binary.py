"""
Label tweets using binary BERT model 

Authors: Eliza Salamon, Apollo Callero, Kate Liggio
"""
import os
import glob
import json
import re
import pandas as pd
import torch
from transformers import BertForSequenceClassification, BertTokenizer
from tqdm import tqdm

import sys
sys.path.append('../')

from helper_funcs import preprocess_text, combine_cloudshare_data


def label(data):
    """
    Label tweets with binary model. Outputs 'others' if no emotion is detected
    """
    emotion_list = ['anger', 'fear', 'joy', 'sadness']

    #set up device
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    #load saved model and set everything up
    models = {emotion: BertForSequenceClassification.from_pretrained('bert-base-multilingual-uncased', num_labels=2) for emotion in emotion_list}
    for emotion in models:
        models[emotion].load_state_dict(torch.load('bert_' + emotion + '.pt',))
        models[emotion].to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased', return_token_type_ids=False)


    #get twitter data
    unlabeled = pd.read_csv(data)
    labeled = pd.DataFrame(columns=['date', 'city', 'id_str', 'language', 'tweet', 'predicted_emotion'])
    tweet_count = 0
    for index, item in tqdm(unlabeled.iterrows()):
        tweet = item['raw_tweet']
        clean_txt = preprocess_text(tweet)
        input = tokenizer(clean_txt, padding=True, max_length = 512,truncation=True,return_tensors="pt")
        mask = input['attention_mask'].to(device)
        inputs = input['input_ids'].to(device)
        #default others
        predicted_emotion = 'others'
        #keep track of highest probability
        max_prob = 0.0
        with torch.no_grad():
            for emotion in models:
                outputs = models[emotion](input_ids=inputs, attention_mask=mask)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)
                label = torch.argmax(logits, dim=1).item()
                emotion_prob = probabilities[0][1].item()  # Positive emotion probability
                if (label == 1) and (emotion_prob > max_prob):
                    max_prob = emotion_prob
                    predicted_emotion = emotion
        new_item = [item['date'], item['city'], item['tweet_id'], item['language'], item['raw_tweet'], predicted_emotion]
        labeled.loc[len(labeled)] = new_item
    labeled.to_csv('binary_predictions.csv')


def main():
    df = combine_cloudshare_data('tweets')
    label(df)


if __name__ == '__main__':
    main()
