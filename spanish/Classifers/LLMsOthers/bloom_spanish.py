import transformers
from transformers import BloomForSequenceClassification
from transformers import BloomTokenizerFast
import torch
import pandas as pd
import emoji
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
import sys
import gc
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import numpy as np

# direct to output
sys.stdout = open("bloom_output.txt", "w")

def getInput(fileName):

    data = pd.read_csv(fileName, sep='\t', on_bad_lines='skip', nrows=1000)
    data = data[~data['emotion'].isin(['surprise', 'others', 'disgust'])]
    data['tweet'] = [emoji.demojize(str(tweet), language='es').replace(":", " ").replace("_", " ") for tweet in data['tweet']]

    return data

''' def trainTestSplit(data):
     
    trainSize = int(0.8 * len(data))
    testSize = len(data) - trainSize
    trainDataset, testDataset = torch.utils.data.random_split(data, [trainSize, testSize])

    return trainDataset, testDataset '''

def trainModel(model, trainLoader, optimizer, device, epochs, LLM):

    model.train()
    for epoch in range(epochs):
        for batch in trainLoader:
            # iteratively send batches to GPU
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

''' def tokenizeData(data):

    tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-1b7")
    tokenizer.pad_token = tokenizer.eos_token
    encoding = tokenizer(list(data), return_tensors='pt', padding=True, truncation=True)
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    return input_ids, attention_mask '''

def runModel(data):

    # tokenize
    tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-1b7")
    tokenizer.pad_token = tokenizer.eos_token
    data['emotion'] = data['emotion'].astype('category')
    labels = data['emotion'].cat.codes
    tweets = data['tweet']

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(tweets, labels, test_size=0.2, random_state=11)
    X_train = [tweet[0] for tweet in X_train]
    X_test = [tweet[0] for tweet in X_test]
    max_length = max([len(tokenizer.encode(text)) for text in X_train])

    # get encodings
    trainEncodings = tokenizer(list(X_train), truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')
    testEncodings = tokenizer(list(X_test), truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')

    # convert data to pytorch tensors
    trainInputs = trainEncodings['input_ids']
    trainAttentionMask = trainEncodings['attention_mask']
    trainLabels = torch.tensor(y_train.tolist())
    testInputs = testEncodings['input_ids']
    testAttentionMask = testEncodings['attention_mask']
    testLabels = torch.tensor(y_test.tolist())

    # store inputs, attention mask and test labels into tensor dataset object
    trainData = TensorDataset(trainInputs, trainAttentionMask, trainLabels)
    testData = TensorDataset(testInputs, testAttentionMask, testLabels)

    # initializing model
    model = BloomForSequenceClassification.from_pretrained("bigscience/bloom-1b7", num_labels=len(data['emotion'].cat.categories))
    model.config.pad_token_id = model.config.eos_token_id

    # initializing optimizer
    optimizer = AdamW(model.parameters(), lr=1e-5)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [{'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                                {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)

    # set up dataloaders 
    trainLoader = DataLoader(trainData, shuffle=True, batch_size=8)
    testLoader = DataLoader(testData, shuffle=True, batch_size=8)

    # set up device
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    model = model.to(device)

    # train model
    model = trainModel(model, trainLoader, optimizer, device, 5, True)

    # evaluate model
    allLabels, allPreds = evalModel(model, testLoader, device, True)

    # compute accuracy
    computeAccuracy(allLabels, allPreds)

def evalModel(model, testLoader, device, LLM):

    model.eval()

    allPreds = []
    allLabels = []

    for batch in testLoader:
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
        allPreds.extend(preds.cpu())
        allLabels.extend(batch_labels.cpu())
    return allLabels, allPreds

def computeAccuracy(yTrue, yPred):
    acc = accuracy_score(yTrue, yPred)
    f1 = f1_score(yTrue, yPred, average='weighted')
    cm = confusion_matrix(yTrue, yPred)
    print("accuracy:", acc)
    print("f1 score:", f1)
    print("confusion matrix:\n", cm)
    print(classification_report(yTrue, yPred))

def main():

    data = getInput('spanish_emotion.tsv')
    runModel(data)

if __name__ == "__main__":
    main()