from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
import argparse
# import adam optim from torch.optim to get rid of warning
from torch.optim import AdamW
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import gc
from torch.utils.data import TensorDataset, DataLoader
from sklearn.feature_extraction.text import CountVectorizer
from torchtext.data.utils import get_tokenizer
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
import torch.optim as optim
from torchtext.vocab import GloVe, Vectors
from sklearn.metrics import classification_report
from imblearn.under_sampling import RandomUnderSampler

import sys
sys.path.append('../MachineLearning')
from ml_spanish import import_set, removeEmojis

# save results to file
def save_results_to_file(model_name, learning_rate, epochs, y_true, y_pred, emotion):
    filename = f'{emotion}_{model_name}_{epochs}ep_{learning_rate}lr.txt'
    with open(filename, 'w') as file:
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        cm = confusion_matrix(y_true, y_pred)
        file.write(f"accuracy: {acc}\n")
        file.write(f"f1 score: {f1}\n")
        file.write(f"confusion matrix:\n {cm}\n")
        file.write(str(classification_report(y_true, y_pred)))

# function to train model
def train_model(model, train_loader, optimizer, device, epochs, LLM):
    model.train()
    for epoch in range(epochs):
        for batch in train_loader:
            # iteratively send batches to gpu (supposedly helps memory issues?)
            if LLM:
                batch_inputs, batch_masks, batch_labels = batch
                batch_inputs, batch_masks, batch_labels = batch_inputs.to(device), batch_masks.to(device), batch_labels.to(device)
                outputs = model(input_ids=batch_inputs, attention_mask=batch_masks, labels=batch_labels)
                loss = outputs.loss
            else:
                batch_inputs, batch_labels = batch
                batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
                outputs = model(batch_inputs)
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # try to free up unused memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print(loss.item())
    return model

# function to evaluate the model
def eval_model(model, test_loader, device, LLM):
    model.eval()
    all_preds = []
    all_labels = []
    for batch in test_loader:
        if LLM:
            batch_inputs, batch_masks, batch_labels = batch
            batch_inputs, batch_masks, batch_labels = batch_inputs.to(device), batch_masks.to(device), batch_labels.to(device)
            with torch.no_grad():
                # no need to compute gradients during evaluation
                outputs = model(input_ids=batch_inputs, attention_mask=batch_masks, labels=batch_labels)
        else:
            batch_inputs, batch_labels = batch
            batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
            with torch.no_grad():
                outputs = model(batch_inputs) 
        # get predicted class
        preds = torch.argmax(outputs.logits, axis=1) if LLM else torch.argmax(outputs, axis=1)
        all_preds.extend(preds.cpu())
        all_labels.extend(batch_labels.cpu())
    return all_labels, all_preds


def gpt2(data, learning_rate, epochs, batch_size, emotion):
    # initalize gpt2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # gpt2 tokenizer does not have padding token - need to add it manually
    tokenizer.pad_token = tokenizer.eos_token
    # turn emotions into numerical labels
    data['emotion'] = data['emotion'].astype('category')
    labels = data['emotion'].cat.codes
    # train test split 
    X_train, X_test, y_train, y_test = train_test_split(data['tweet'], labels, test_size=0.2, random_state=11)
    # set the max_length as the length of the longest sequence in dataset
    max_length = max([len(tokenizer.encode(text)) for text in X_train])
    # tokenize and pad sequences
    train_encodings = tokenizer(list(X_train), truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')
    test_encodings = tokenizer(list(X_test), truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')
    # convert data to pytorch tensors
    train_inputs = train_encodings['input_ids']
    train_attention_mask = train_encodings['attention_mask']
    train_labels = torch.tensor(y_train.tolist())
    test_inputs = test_encodings['input_ids']
    test_attention_mask = test_encodings['attention_mask']
    test_labels = torch.tensor(y_test.tolist())
    # store inputs, attention mask and test labels into tensor dataset object
    train_data = TensorDataset(train_inputs, train_attention_mask, train_labels)
    test_data = TensorDataset(test_inputs, test_attention_mask, test_labels)
    # initialize the pretrained gpt2 classification model
    model = GPT2ForSequenceClassification.from_pretrained('gpt2', num_labels=2)
    model.config.pad_token_id = model.config.eos_token_id
    # setting up the optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    # set up device, use gpu if possible so my laptop doesn't blow up
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    # put tensors into data loader to separate data into batches
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)
    # train, test, and get metrics
    model = train_model(model, train_loader, optimizer, device, epochs, True)
    all_labels, all_preds = eval_model(model, test_loader, device, True)
    # compute_accuracy_metrics(all_labels, all_preds)
    save_results_to_file('gpt2', learning_rate, epochs, all_labels, all_preds, emotion)
    
def glove_spanish(data, learning_rate, epochs, batch_size, emotion): 
    # use the Spanish tokenizer
    tokenizer = get_tokenizer('spacy', language='es_core_news_sm')
    # tokenize the tweets
    tokenized = [tokenizer(tweet) for tweet in data['tweet']]
    # create vectors using GloVe embeddings
    glove = Vectors(name="multilingual_embeddings.es", cache='./')
    # get the dimension of glove vectors
    glove_dim = glove.vectors.size(1) 
    vectors = []
    for tweet in tokenized:
        if any(word in glove.stoi for word in tweet):
            tweet_vectors = sum(glove[word] for word in tweet if word in glove.stoi)
        else:
            # use zero vector if no words in tweet are in glove dict
            tweet_vectors = torch.zeros(glove_dim) 
        vectors.append(tweet_vectors)
    vectors = np.stack(vectors)
    # turn emotions into numerical labels
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(data['emotion'])
    # train test split 
    X_train, X_test, y_train, y_test = train_test_split(vectors, labels, test_size=0.2, random_state=11)
    # convert to tensors
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)
    # create data loaders for batches 
    train_data = TensorDataset(X_train, y_train)
    test_data = TensorDataset(X_test, y_test)
    # number of input features
    D_in = X_train.size(1)
    D_out = 2
    # simple linear layer
    model = nn.Linear(D_in, D_out)
    # stochastic gradient descent
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    # set up device, use gpu if possible so my laptop doesn't blow up
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    # loss function
    loss_fn = nn.CrossEntropyLoss()
    # set up the dataloaders
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)
    # train, eval, and get metrics
    model = train_model(model, train_loader, optimizer, device, epochs, False)
    all_labels, all_preds = eval_model(model, test_loader, device, False)
    save_results_to_file('glove', learning_rate, epochs, all_labels, all_preds, emotion)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, nargs='*', default=[0.001, 0.01, 0.1],
                        help='Learning rate - default is [0.001, 0.01, 0.1].')
    parser.add_argument('--epochs', type=int, nargs='*', default=[5, 10, 15],
                        help='Number of epochs - default is [5, 10, 15].')
    parser.add_argument('--bs', type=int, nargs='*', default=[16, 32, 64],
                        help='Batch size - default is [16, 32, 64].')
    parser.add_argument('--model', type=str, choices=['gpt2', 'glove', 'both'], default='both',
                        help='Model to use - default is both.')
    args = parser.parse_args()
    # import data set
    data = import_set('spanish_emotion.tsv')
    # list of emotions
    emotions = ['anger', 'fear', 'sadness', 'joy']
    # preprocess emojis into text
    for i in range(len(data)):
        row = data.iloc[i]
        new_tweet = removeEmojis(row['tweet'])
        row['tweet'] = new_tweet
    
    # run model for different learning rates, epochs and batch sizes
    for emotion in emotions:
        for lr in args.lr:
            for epoch in args.epochs:
                for bs in args.bs:
                    data_copy = data.copy()
                    data_copy['emotion'] = data_copy['emotion'].apply(lambda x: 1 if x == emotion else 0)
                    if args.model in ['gpt2', 'both']:
                        gpt2(data_copy, lr, epoch, bs, emotion)
                    if args.model in ['glove', 'both']:
                        glove_spanish(data_copy, lr, epoch, bs, emotion)

if __name__ == "__main__":
    main()
