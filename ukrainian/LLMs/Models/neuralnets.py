import torch
import torch.nn as nn
import nltk
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score
from torch.utils.data import TensorDataset, DataLoader
from transformers import BloomTokenizerFast, BloomForSequenceClassification
from tqdm import tqdm
import argparse

#make sure we get all the files we need!
import sys
sys.path.append('../', '../..')
from helper_funcs import *

EMOTIONS = ['anger', 'fear', 'sadness', 'disgust', 'joy']
EMOTIONS2 = ['anger', 'fear', 'sadness', 'joy']


def mlp(data, epochs, learning_rate):
    """
    Basic MLP function to get the feel for neural nets
    """
    #vectorize
    vectorizer = CountVectorizer()
    v = vectorizer.fit_transform(data['processed_tweets'])
    ngrams = v.toarray()
    names = vectorizer.get_feature_names_out()
    print(len(names), "words in total")

    #encode labels
    for emotion in EMOTIONS2:
        print(emotion)
        # split data into train and test
        label_col = 'emotion_' + emotion
        X_train, X_test, Y_train, Y_test = train_test_split(ngrams, data[label_col], test_size=0.2, random_state = 42)
    
        #scale data
        scaler = StandardScaler()

        X_train = torch.FloatTensor(scaler.fit_transform(X_train))

        loss_fn = torch.nn.CrossEntropyLoss()

        D_in = len(names)
        D_out = 2

        # mlp model
        model = torch.nn.Linear(D_in, D_out)
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

        labels = torch.tensor(list(Y_train))
        
        #train
        for i in range(epochs):
            pred = model(X_train)
            loss = loss_fn(pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # evaluation
        X_test = torch.FloatTensor(scaler.fit_transform(X_test))
        pred = model(X_test).tolist()
        max_preds = [np.argmax(i) for i in pred]
        print('accuracy: ', round(accuracy_score(Y_test, max_preds), 3))
        print('f1: ', round(f1_score(Y_test, max_preds), 3))

def load_glove_model(File):
    """
    Loads the GloVe embeddings into a dictionary
    https://stackoverflow.com/questions/37793118/load-pretrained-glove-vectors-in-python
    """
    print("Loading Glove Model")
    glove_model = {}
    with open(File,'r') as f:
        for line in tqdm(f):
            split_line = line.split()
            word = split_line[0]
            glove_model[word] = list(map(float, split_line[1:]))
    print(f"{len(glove_model)} words loaded!")
    return glove_model

def tokenize(txt_lst):
   """
   Tokenizes list of tweets
   """
   regexp = nltk.tokenize.RegexpTokenizer('\w+')
   txt_lst = txt_lst.map(str)
   tokens = txt_lst.apply(regexp.tokenize)
   return tokens

def embed(dict, tweets):
    """
    creates a matrix representing each given tweet
    """
    #create matrix
    embedded_matrix = np.zeros((len(tweets), 300))
    for index, tweet in enumerate(tweets):
        tweet_vector_sum = np.zeros(300)
        for word in tweet:
            if word in dict:
                #sum embeddings for all words in individual tweet
                tweet_vector_sum = tweet_vector_sum + dict[word]
        embedded_matrix[index] = tweet_vector_sum
    return embedded_matrix

def train(model, dataloader, optimizer, epochs, device, bloom , bert):
    """
    train the model from DataLoader object with batching
    """
    #train time!
    model.train()
    for e in tqdm(range(epochs)):
        print(e)
        for batch in dataloader:
            if bloom:
                batch_inputs, batch_mask, batch_labels = batch
                batch_inputs, batch_mask, batch_labels = batch_inputs.to(device), batch_mask.to(device), batch_labels.to(device)
                pred = model(input_ids=batch_inputs, attention_mask=batch_mask, labels=batch_labels)
                loss = pred.loss
            loss_fn = nn.CrossEntropyLoss()
            if bert:
                batch_labels = batch[1].to(device)
                input_id,mask = batch[0]['input_ids'].squeeze(1).to(device) ,batch[0]['attention_mask'].to(device)
                pred = model(input_id , mask)
                loss = loss_fn(pred, batch_labels.long())
            else:
                batch_inputs, batch_labels = batch
                batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
                pred = model(batch_inputs)
                loss = loss_fn(pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    print('Done training!')
    return model

def evaluate(model, dataloader, device,bert):
    """     
    Evaluates the model using the test data
    Returns the predicted values
    """
    model.eval()
    labels = []
    for batch in dataloader:
        batch_inputs, batch_labels = batch
        if bert:
              mask = batch[0]['attention_mask'].to(device)
              input_id = batch[0]['input_ids'].squeeze(1).to(device)
              pred = model(input_id, mask)
        else:
           batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
           pred = model(batch_inputs)
        max_preds = torch.argmax(pred, axis=1)
        labels.extend(max_preds.cpu())
    return labels

def accuracy(model_name, true, preds, epochs, batch_size, lr):
    """
    Outputs a file with model parameter information and various accuracy metrics
    """
    file_name = 'results/' + model_name + '.txt'

    #compute accuracies
    if type(true) is torch.Tensor:
        true = true.cpu().tolist()

    with open(file_name, 'a') as f:
        f.write('epochs = {}, batch size = {}, learning rate = {}'.format(epochs, batch_size, lr))
        f.write('\n')
        f.write(classification_report(true, preds))
        f.write('\n')

def glove(corpus, data, lr, batch_size, epochs):
    """
    Uses GloVe embeddings to create model
    GlLoVe vectors obtained from https://lang.org.ua/en/models/#anchor4 (too big of a file to put on GitHub)
    """
    glove_dict = load_glove_model(corpus)

    #tokenize then...glove!
    tokenized_text = tokenize(data['processed_tweets'])
    embedded_text = embed(glove_dict, tokenized_text)

    X_train, X_test, Y_train, Y_test = train_test_split(embedded_text, data['encoded_emotion'], test_size=0.2, random_state = 42, stratify=data['encoded_emotion'])
    
    #transform to tensors
    X_train = torch.FloatTensor(X_train)
    Y_train = torch.LongTensor(list(Y_train))
    X_test = torch.FloatTensor(X_test)
    Y_test = torch.FloatTensor(list(Y_test))

    X_train = nn.utils.rnn.pad_sequence(X_train, batch_first=True)
    X_test = nn.utils.rnn.pad_sequence(X_test, batch_first=True)

    train_data = TensorDataset(X_train, Y_train)
    test_data = TensorDataset(X_test, Y_test)

    #set in and out dimensions
    D_in = X_train.size(1)
    D_out = len(data['encoded_emotion'].unique())

    #set loss
    loss_fn = nn.CrossEntropyLoss()

    #set up model, check for gpu
    model = nn.Linear(D_in, D_out)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    #set up dataloaders for batching
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

    #set optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    #get trained model
    trained_model = train(model, train_loader, optimizer, epochs, device, False)

    # evaluate model
    predictions = evaluate(trained_model, test_loader, device)

    #finally, get accuracy results
    accuracy('GloVe', Y_test, predictions, epochs, batch_size, lr)


def GPT2(data):
    # Define model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2ForSequenceClassification.from_pretrained('gpt2', num_labels=len(EMOTIONS2))
    model.train()

    for emotion in EMOTIONS2:
        print(emotion)
        # split data into train and test
        label_col = 'emotion_' + emotion
        X_train, X_test, Y_train, Y_test = train_test_split(data['processed_tweets'], data[label_col], test_size=0.2,
                                                            random_state=42)
        # tokenize and encode sequences in the training set
        tokens_train = tokenizer.batch_encode_plus(
            X_train.tolist(),
            max_length=512,
            pad_to_max_length=True,
            truncation=True,
            return_token_type_ids=False
        )
        # tokenize and encode sequences in the test set
        tokens_test = tokenizer.batch_encode_plus(
            X_test.tolist(),
            max_length=512,
            pad_to_max_length=True,
            truncation=True,
            return_token_type_ids=False
        )
        # convert lists to tensors
        train_seq = torch.tensor(tokens_train['input_ids'])
        train_mask = torch.tensor(tokens_train['attention_mask'])
        train_y = torch.tensor(Y_train.tolist())

        test_seq = torch.tensor(tokens_test['input_ids'])
        test_mask = torch.tensor(tokens_test['attention_mask'])
        test_y = torch.tensor(Y_test.tolist())

        # compute the loss, gradients, and update the parameters by calling optimizer.step()
        outputs = model(train_seq, attention_mask=train_mask, labels=train_y)
        loss = outputs.loss
        loss.backward()
        optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
        optimizer.step()
        # evaluation
        pred = model(test_seq, attention_mask=test_mask)
        max_preds = pred.logits.argmax(-1)
        print('accuracy: ', round(accuracy_score(Y_test, max_preds.detach().numpy()), 3))
        print('f1: ', round(f1_score(Y_test, max_preds.detach().numpy()), 3))

def mBERT(data): 
    return

def MUSE(data):
    # Download Ukrainian vectors from MUSE
    fasttext.util.download_model('uk', if_exists='ignore')  # Ukrainian
    ft = fasttext.load_model('cc.uk.300.bin')

    for emotion in EMOTIONS2:
        print(emotion)
        # split data into train and test
        label_col = 'emotion_' + emotion
        X_train, X_test, Y_train, Y_test = train_test_split(data['processed_tweets'], data[label_col], test_size=0.2,
                                                            random_state=42)

        # Transform the data to embeddings
        X_train_transformed = [ft.get_sentence_vector(x) for x in X_train]
        X_test_transformed = [ft.get_sentence_vector(x) for x in X_test]

        # Convert to tensors
        X_train_tensor = torch.tensor(X_train_transformed)
        X_test_tensor = torch.tensor(X_test_transformed)
        Y_train_tensor = torch.tensor(Y_train.tolist())
        Y_test_tensor = torch.tensor(Y_test.tolist())

        D_in = X_train_tensor.shape[1]
        D_out = 2

        # MLP model
        model = torch.nn.Linear(D_in, D_out)
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loss_fn = torch.nn.CrossEntropyLoss()

        # Training
        for i in range(20):
            pred = model(X_train_tensor)
            loss = loss_fn(pred, Y_train_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluation
        pred = model(X_test_tensor)
        max_preds = [np.argmax(i) for i in pred.tolist()]
        print('accuracy: ', round(accuracy_score(Y_test_tensor, max_preds), 3))
        print('f1: ', round(f1_score(Y_test_tensor, max_preds), 3))

def bloom(data, lr, batch_size, epochs):
    """
    Uses the Bloom algorithm to create tokens
    """
    #set up bloom
    tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-1b7")
    tokenizer.pad_token = tokenizer.eos_token

    X_train, X_test, Y_train, Y_test = train_test_split(data['processed_tweets'], data['encoded_emotion'], test_size=0.2, random_state=42)

    #get max length of tweet
    max_length = max([len(tokenizer.encode(text)) for text in X_train])

    train_inputs = tokenizer(list(X_train), padding='max_length', max_length = max_length, return_tensors='pt')
    test_inputs = tokenizer(list(X_test), padding='max_length', max_length = max_length, return_tensors='pt')

    train_tensor = torch.LongTensor(train_inputs['input_ids'])
    test_tensor = torch.LongTensor(test_inputs['input_ids'])

    train_masks = train_inputs['attention_mask']
    test_masks = test_inputs['attention_mask']

    Y_train = torch.tensor(list(Y_train))
    Y_train = torch.zeros(Y_train.shape[0], 4).scatter(1, Y_train.unsqueeze(1), 1.0)
    Y_test = torch.tensor(list(Y_test))
    Y_test = torch.zeros(Y_test.shape[0], 4).scatter(1, Y_test.unsqueeze(1), 1.0)

    train_data = TensorDataset(train_tensor, train_masks, Y_train)
    test_data = TensorDataset(test_tensor, test_masks, Y_test)

    #set up model and optimizer
    model = BloomForSequenceClassification.from_pretrained("bigscience/bloom-560m", num_labels=4, problem_type="multi_label_classification")
    model.config.pad_token_id = model.config.eos_token_id

    #set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    #set up loaders for batching
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

    trained_model = train(model, train_loader, optimizer, epochs, device, True)

    preds = evaluate(trained_model, test_loader, device)

    accuracy('Sample', Y_test, preds, epochs, batch_size, lr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default='../data/ukrainian_emotion_new_new.tsv')
    args = parser.parse_args()

    encoded, _ = encode_and_vectorize_multi_class(args.file, combine=True)
    # binary_encoded, _ = encode_and_vectorize_binary(file_name, combine=True)
    corpus_file = '../data/news.lowercased.lemmatized.glove.300d'
    learning_rate = 0.001
    batch_size = 10
    epochs = 30
    # mlp(binary_encoded, epochs, learning_rate)
    glove(corpus_file, encoded, learning_rate, batch_size, epochs)
    # bloom(encoded,learning_rate, batch_size, epochs)


if __name__ == "__main__":
    main()
