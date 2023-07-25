'''
To do:
  Test a threshold to relabel some predictions to 'others'
  if not super conifent in label

'''


from bert_multi import accuracy
from transformers import BertModel, BertTokenizer, BertForSequenceClassification
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
from torch.optim import AdamW
from tqdm import tqdm
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertModel
import argparse

import sys

sys.path += ['../', '../..']
from helper_funcs import *
EMOTIONS2 = ['anger', 'fear', 'sadness', 'joy']
#gather input

def get_test_data(file_name , emotion_col):
        '''
        Parameters:
          file_name: path to labeled data withou 'others'
          emotion_col: the labeled emotion column name in csv file 
          Returns:
             test_loader which loads the input and outputs in evaluate()
             mapping: dictionairy which maps a emotion to its corresponding number
        '''

        #load in test set
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased', return_token_type_ids=False)
        data, X , mapping = encode_and_vectorize_multi_class(file_name, combine=True)
        X_train, X_test, Y_train, Y_test = train_test_split(data['processed_tweets'], data[emotion_col], test_size=0.2, stratify=data[emotion_col], random_state=42)
        mapping['others'] = 4
        #load in dataset with 'others'
        others_df = pd.read_csv('../data/ukrainian_emotion_new.tsv', sep='\t', encoding = 'utf8')
        drop_rows = others_df.loc[~others_df['emotion'].isin(['others'])]
        others_df.drop(drop_rows.index, inplace=True)
        print(type(X_test) , '\n' , Y_test)
        for i , row in others_df.iterrows():
                tweet = pd.Series({6000 + i:preprocess_text(row['tweet'])})
                #X_test.append(tweet , ignore_index = True)
                X_test = pd.concat([X_test , tweet]) 
                Y_test = pd.concat ([ Y_test , pd.Series({6000 + i:'others'})])
                #Y_test.append(pd.Series({6000 + i:'others'}) , ignore_index = True)
        test_inputs = tokenizer(list(X_test), padding=True, max_length = 512, truncation=True,return_tensors="pt")
        Y_test = [mapping[i] for i in Y_test]
        Y_test= torch.LongTensor(list(Y_test))
        test_data = TensorDataset(test_inputs['input_ids'], test_inputs['attention_mask'], Y_test)
        test_loader = DataLoader(test_data, batch_size=20, shuffle=True)
        return test_loader , mapping

def get_models():
        '''
        Returns list of pre trained binary models
        '''
        device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
        fear_model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-uncased', num_labels=2)
        joy_model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-uncased', num_labels=2)
        anger_model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-uncased', num_labels=2)
        sadness_model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-uncased', num_labels=2)
        fear_model.load_state_dict(torch.load('bert_fear.pt' , map_location=device))
        joy_model.load_state_dict(torch.load('bert_joy.pt' , map_location=device))
        sadness_model.load_state_dict(torch.load('bert_sadness.pt' , map_location=device))
        anger_model.load_state_dict(torch.load('bert_anger.pt' , map_location=device))
        return [anger_model , fear_model , sadness_model , joy_model]

def evaluate(models, test_data, device):
    '''
    Parameters:
        model: bert model that was fine tuned previously
        test_data: df of proccessed tweets and labels that was not shown to bert

    Summuary:
        Evaluates the model on the test data and returns 2 lists of the predicted and true labels
    '''
    true_labels = []
    pred_labels = []
    for model in models:
         model.eval()
    with torch.no_grad():
        for test_input, test_mask, test_label in test_data:
            test_label = test_label.to(device)
            mask = test_mask.to(device)
            input_id = test_input.to(device)
            anger_output = models[0](input_id, attention_mask=mask )#, labels=test_label)
            anger_max_preds = list(torch.argmax(anger_output.logits, dim=1))
            fear_output = models[1](input_id, attention_mask=mask)#, labels=test_label)
            fear_max_preds = list(torch.argmax(fear_output.logits, dim=1))
            sadness_output = models[2](input_id, attention_mask=mask)#, labels=test_label)
            sadness_max_preds = list(torch.argmax(sadness_output.logits, dim=1))
            joy_output = models[3](input_id, attention_mask=mask)#, labels=test_label)
            joy_max_preds = list(torch.argmax(joy_output.logits, dim=1))
            # check if all outputs say 0 ,  then label other
            for i in range(len(joy_max_preds)):
                 if 0 == joy_max_preds[i] == anger_max_preds[i] == fear_max_preds[i] == sadness_max_preds[i]:
                     pred_labels.append('others')
                     continue
                #print('a' , list(joy_output.logits))
                 joy_confidence = list(joy_output.logits)[i][1]
                 fear_confidence = list(fear_output.logits)[i][1]
                 sadness_confidence = list(sadness_output.logits)[i][1]
                 anger_confidence = list(anger_output.logits)[i][1]
                 pred , max_confidence = 'joy' , joy_confidence
                 if fear_confidence > max_confidence:
                     pred , max_confidence = 'fear' , fear_confidence
                 if sadness_confidence > max_confidence:
                     pred , max_confidence = 'sadness' , sadness_confidence
                 if anger_confidence > max_confidence:
                     pred , max_confidence = 'anger' , anger_confidence
                 pred_labels.append(pred)
            true_labels.extend(test_label.cpu().tolist())

    #torch.save(model.state_dict(), "bert_good_model.pt")
    #print(true_labels , pred_labels)
    return true_labels, pred_labels

def main():
        device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
        data_path = '../data/ukrainian_emotion_new_new.tsv'
        binary_models = get_models()
        test_loader , mapping = get_test_data(data_path , 'emotion') #_' + EMOTIONS2[i])
        true, preds = evaluate(binary_models, test_loader, device)
        preds = [mapping[i] for i in preds]
        print(len(true) , len(preds))
        accuracy('Bert_all_binary', true, preds, 5, 20, .000005)
main()
