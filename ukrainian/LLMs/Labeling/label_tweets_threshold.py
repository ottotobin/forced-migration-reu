"""
Label tweets using multiclass BERT model with a threshold

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
import argparse

import sys
sys.path += ['../', '../..']
from helper_funcs import preprocess_text, combine_cloudshare_data


def label(data, threshold):
    """
    Label tweets with threshold
    """
    #emotion/class mapping dicts
    emotion_to_class = {'anger': 0, 'fear': 1, 'joy': 2, 'sadness': 3}
    class_to_emotion = {0: 'anger', 1: 'fear', 2: 'joy', 3: 'sadness'}

    #load saved model and set everything up
    model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-uncased', num_labels=4)
    model.load_state_dict(torch.load('bert_good_model.pt'))
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased', return_token_type_ids=False)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model.to(device)

    unlabeled = pd.read_csv(data)
    labeled = pd.DataFrame(columns=['date', 'city', 'id_str', 'language', 'tweet', 'predicted_emotion'])
    model.eval()
    for index, item in tqdm(unlabeled.iterrows()):
        if item['language'] == 'uk':
            tweet = item['raw_tweet']
            clean_txt = preprocess_text(tweet)
            input = tokenizer(clean_txt, padding=True, max_length = 512,truncation=True,return_tensors="pt")
            mask = input['attention_mask'].to(device)
            inputs = input['input_ids'].to(device)
            with torch.no_grad():
                outputs = model(input_ids = inputs, attention_mask=mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            max_prob = torch.max(probabilities, dim=1)
            # set threshold for joy
            if max_prob[1].item() == 2:
                if max_prob[0].item() >= threshold:
                    predicted_emotion = class_to_emotion[max_prob[1].item()]
                else:
                    predicted_emotion = 'others'
            else:
                predicted_emotion = class_to_emotion[max_prob[1].item()]
            new_item = [item['date'], item['city'], item['tweet_id'], item['language'], item['raw_tweet'], predicted_emotion]
            labeled.loc[len(labeled)] = new_item
    labeled.to_csv('threshold_' + str(threshold) + '_predictions.csv')


def main():
    parser = argparse.ArgumentParser(description="Setting Threshold for Labeling")
    parser.add_argument('--threshold', required=True, default=0.7, help='Set threshold for Labeling others')

    args = parser.parse_args()

    threshold = float(args.threshold)
    df = combine_cloudshare_data('tweets')
    label(df, threshold)

if __name__ == '__main__':
    main()



