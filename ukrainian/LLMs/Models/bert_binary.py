'''
Builds a binary BERT model to classify emotions in tweets

Authors: Eliza Salamon, Apollo Callero, Kate Liggio
'''
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
sys.path.append('../', '../..')
from helper_funcs import *

EMOTIONS = ['anger', 'fear', 'sadness', 'disgust', 'joy']
EMOTIONS2 = ['anger', 'fear', 'sadness', 'joy']


def train(data, learning_rate, epochs, batch_size, emotion):
    model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-uncased', num_labels=2)
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased', return_token_type_ids=False)

    X_train, X_test, Y_train, Y_test = train_test_split(data['processed_tweets'], data['emotion_{}'.format(emotion)],
                                                        test_size=0.2, stratify=data['emotion_{}'.format(emotion)],
                                                        random_state=42)

    max_length = max([len(text) for text in X_train])

    train_inputs = tokenizer(list(X_train), padding=True, max_length=max_length, truncation=True, return_tensors="pt")
    test_inputs = tokenizer(list(X_test), padding=True, max_length=max_length, truncation=True, return_tensors="pt")

    Y_train = torch.LongTensor(list(Y_train))
    Y_test = torch.LongTensor(list(Y_test))

    train_data = TensorDataset(train_inputs['input_ids'], train_inputs['attention_mask'], Y_train)
    test_data = TensorDataset(test_inputs['input_ids'], test_inputs['attention_mask'], Y_test)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    # check for gpu availability
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    # set loss and optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)

    model.train()
    for epoch_num in range(epochs):
        print('Epoch ', epoch_num)
        for batch in tqdm(train_dataloader):
            batch_input, batch_attention_mask, batch_labels = batch
            batch_input = batch_input.to(device)
            batch_attention_mask = batch_attention_mask.to(device)
            batch_labels = batch_labels.to(device)
            output = model(batch_input, attention_mask=batch_attention_mask, labels=batch_labels)
            batch_loss = output.loss
            # reset gradients
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    model, true, preds = evaluate(model, test_loader, device, emotion)
    accuracy('Bert_binary_{}'.format(emotion), true, preds, epochs, batch_size, learning_rate)
    return model


def evaluate(model, test_data, device, emotion):
    '''
    Parameters:
        model: bert model that was fine tuned previously
        test_data: df of proccessed tweets and labels that was not shown to bert

    Summuary:
        Evaluates the model on the test data and Prints the accuracy to the console
    '''
    model = model.to(device)

    true_labels = []
    pred_labels = []

    model.eval()
    with torch.no_grad():
        for test_input, test_mask, test_label in test_data:
            test_label = test_label.to(device)
            mask = test_mask.to(device)
            input_id = test_input.to(device)
            output = model(input_id, attention_mask=mask, labels=test_label)
            max_preds = torch.argmax(output.logits, dim=1)
            pred_labels.extend(max_preds.cpu().tolist())
            true_labels.extend(test_label.cpu().tolist())

    torch.save(model.state_dict(), 'bert_good_model_{}.pt'.format(emotion))
    return model, true_labels, pred_labels


def accuracy(model_name, true, preds, epochs, batch_size, lr):
    """
    Outputs a file with model parameter information and various accuracy metrics
    """
    file_name = 'results/' + model_name + '.txt'

    print(true)
    print('\n')
    print(preds)

    # compute accuracies
    if type(true) is torch.Tensor:
        true = true.cpu().tolist()
    if type(preds) is torch.Tensor:
        preds = preds.cpu().tolist()

    with open(file_name, 'a') as f:
        f.write('epochs = {}, batch size = {}, learning rate = {}'.format(epochs, batch_size, lr))
        f.write('\n')
        f.write(classification_report(true, preds))
        f.write('\n')


def main():
    # load in major parameters and dataframe
    parser = argparse.ArgumentParser(description="Emotion Detection ML Models")
    parser.add_argument('--learningRate', help="learning rate for bert to use")
    parser.add_argument('--epochs', help='Number of epochs for training', default=5)
    parser.add_argument('--batches', help="batch size", default= 20)
    parser.add_argument('--file', help='tsv file where data is stored', default='../data/ukrainian_emotion_new_new.tsv')
    args = parser.parse_args()

    lr = float(args.learningRate)
    batch_size = int(args.batches)
    epochs = int(args.epochs)
    encoded, X = encode_and_vectorize_binary(args.file, combine=True)
    print(encoded)
    for emotion in EMOTIONS2:
        model = train(encoded, lr, epochs, batch_size, emotion)



if __name__ == "__main__":
    main()
