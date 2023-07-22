"""
Authors: Lina Laghzaoui, Grace Magny-Fokam, Toby Otto, Rich Pihlstrom

Description:
    This program allows the user to run sensitivity testing for bert, GloVe, muse, GPT2, and a standard
    bag-of-words mlp. We did not get BLOOM up and running as the memory consumption was too high to efficiently
    train. The user can edit the parameters in the data/tuning_params.json file to come up with a grid for
    the parameter testing to crawl through. In addition to param testing, the program will determine the
    best set of parameters (based solely on accuracy) and then train and save the model with the best parameters.

A note on bert:
    Given that our data is multi-labelled, we mostly just followed a binary-output method.
    Due to how long Bert takes, we opted to not train 6 binary models for each emotion but instead,
    have the model output a vector of 6 binary values. We then considered each element of the vector
    as a label for the respective emotion.

Necessary Packages:
    You will need to install pytrends in order to use the API in this code:
        pip3 install fasttext transformers torch scikit-learn

model: the name(s) of the model for which  you want to run testing/param-aggregation.
    The options for this are bert2, muse, GloVe, BagOfWords, and GPT2

filename: the path to the original datafile that you are using.

sensitivityTesting: whether or not you want to run sensitivity testing for the models.
    If you have already done comprehensive parameter testing, you can just aggregate and 
    train the best models by entering 0.

USAGE:
    python3 llm.py -m <muse, GloVe, bert2> -s <1,0>

"""

import re, os, torch, itertools, argparse, transformers, fasttext.util, multiprocessing
import pandas as pd
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score,recall_score
from torch.utils.data import DataLoader, TensorDataset
from fasttext import load_model
from transformers import (
    AutoModel,
    AutoTokenizer, 
    GPT2Tokenizer, 
    GPT2ForSequenceClassification,
    BertModel,
    BertTokenizer, 
    BertForSequenceClassification
)

# Function to preprocess the csv data
# Takes input of the original data file and the emoji flag
# Outputs a csv file and return the dataframe version of that file

# NB: from previous classification methods, we determined that emoji translation
# did not impact results much so, in the interested of minimizing tests, we had
# the emoji flag as false for all testing.
def preprocess(data_file, emoji_flag):

    # If the output file already exists, we can just read
    # it and return the dataframe
    if os.path.exists("data/processed_data.csv"):
        with open("data/processed_data.csv", "r") as f:
            return pd.read_csv(f)

    else:
        # read data
        df = pd.read_csv(data_file, sep="\t", encoding="utf8")

        # preprocess text - remove @s, #s, links
        emoji_pattern = re.compile(
            "["
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "]+",
            flags=re.UNICODE,
        )
        processed_tweets = []
        
        #Tokenize and process each tweet
        for token_list in tokenize(df["Tweet"]):
            #Translate the emojis and add/remove to sentence
            text = emoji_translator(token_list, emoji_flag)
            text = " ".join(
                word
                for word in text.split()
                if not (
                    word.startswith("http")
                    or word.startswith("@")
                    or word.startswith("#")
                )
            )
            text = emoji_pattern.sub(r"", text)
            text = text.lower()
            processed_tweets.append(text)

        # Create a new column for processed tweets
        df["processed_tweets"] = processed_tweets

        # Create a new column for combined disgust and anger.
        # Because our data is multi-labelled, we can just create
        # a new column without needing to be concerned about affceting
        # other classifications
        df["anger-disgust"] = df["anger"] + df["disgust"]
        df["anger-disgust"] = df["anger-disgust"].replace(2, 1)

        # Save data
        df.to_csv("data/processed_data.csv")

        return df

# tokenizes a list of tweets
def tokenize(tweets):
    regexp = RegexpTokenizer("\w+")
    tweets = tweets.map(str)
    token_col = tweets.apply(regexp.tokenize)
    return token_col

# translates emojis to Arabic text using emojis.csv
def emoji_translator(token_list, emoji_flag):
    return_tweet = ""
    emojis_df = pd.read_csv("data/emojis.csv")

    for token in token_list:
        # checks to see if emoji_flag is True, otherwise just omits the emoji altogether
        if token in emojis_df["emoji"].values and emoji_flag:
            return_tweet += emojis_df[emojis_df["emoji"] == token]["text"].values[0]
        else:
            return_tweet += token
        return_tweet += " "

    return return_tweet

# An oversampling function that returns new X_train and y_train lists
def resamp(X_train, y_train):
    from imblearn.over_sampling import RandomOverSampler
    oversampler = RandomOverSampler()

    X_train_reshaped = X_train.values.reshape(-1, 1)
    X_train, y_train = oversampler.fit_resample(X_train_reshaped, y_train)
    X_train = [l[0] for l in X_train]

    return X_train, y_train

lock = multiprocessing.Lock()
def get_gpu_device():
    num_gpus = torch.cuda.device_count()
    
    best_mem = 0
    best_i=0

    lock.acquire()
    for i in range(num_gpus):
        device = torch.device(f'cuda:{i}')
        torch.cuda.set_device(device)
        properties = torch.cuda.get_device_properties(device)
        available_memory = properties.total_memory - torch.cuda.memory_allocated(device)

        if available_memory > best_mem:
            best_mem = available_memory
            best_i = i
    lock.release()

    return torch.device(f'cuda:{best_i}')


################ RUN MLP CLASSIFIER ################

# Wrapper function for running an MLP classifier with specified parameters
# Takes training and test data and a configuration of parameters
# Returns a dictionary of the results for that specific model
def run_mlp(X_train, y_train, X_test, y_test, epochs, lr, batch_size=0, best_model=0, best_model_name="", threshold=0.5):
    
    input_size = X_train.shape[-1]
    output_size = 2

    # MLP model
    model = torch.nn.Linear(input_size, output_size)

    # accesssing GPU if available
    
    if torch.cuda.is_available():
        device = get_gpu_device()
        X_train = X_train.to(device)
        X_test = X_test.to(device)
        y_train = y_train.to(device)
        #y_test = y_test.to(device)
        model = model.to(device)

    # set internal model flag to train-mode
    model.train()

    # Define optimizers and loss function
    optimizer = torch.optim.SGD(model.parameters(), lr)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    # If the batch size is non-zero, we can define a dataloader object with
    # the batches broken up
    if batch_size != 0:
        d = TensorDataset(X_train, y_train)
        dataloader = DataLoader(d, batch_size=batch_size, shuffle=True)

    # training mlp model
    for i in range(epochs):

        # Use try catch block to iterate over batches in data loader if necessary
        try:
            for X_train_batch, y_train_batch in dataloader:
                mlp_train(X_train_batch, y_train_batch, model, loss_fn, optimizer)
        except NameError:
            mlp_train(X_train, y_train, model, loss_fn, optimizer)
    
    # If the best_model flag is active, that means we want to save
    # the model to be run on our unlabeled data
    if best_model:
        torch.save(model.state_dict(), best_model_name)

    # Get prediction labels
    pred_labels = mlp_test(X_test, model,threshold)

    # Add evaluation metric entry to return dictionary
    return evalDict(y_test, pred_labels)

# returns a dictionary of evaluation metrics
# takes a list of the predicted and actual values
# returns a dictionary of evaluation metrics
def evalDict(pred, act):
    evalDict = {
        "macro precision": precision_score(act, pred, average="macro", zero_division=0),
        "macro recall": recall_score(act, pred, average="macro", zero_division=0),
        "macro f1": f1_score(act, pred, average="macro", zero_division=0),
        "micro precision": precision_score(act, pred, average="micro", zero_division=0),
        "micro recall": recall_score(act, pred, average="micro", zero_division=0),
        "micro f1": f1_score(act, pred, average="micro", zero_division=0),
        "accuracy": accuracy_score(act, pred),
    }
    return evalDict

# the training function for the model
# takes the model parameters as inpu
def mlp_train(X, labels, model, loss_fn, optimizer):
    m = torch.nn.Softmax(dim=1)
    pred = m(model(X))

    loss = loss_fn(pred, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# the testing function for the model
# takes the model and the testing set and threshold as
# inputs and returns a list of predicted values
def mlp_test(X_test, model, threshold):
    X_test = X_test.float()

    # Define soft-max activiation
    m = torch.nn.Softmax(dim=1)
    pred = m(model(X_test)).tolist()
    #pred = model(X_test).tolist()
    #pred_labels = [np.argmax(i) for i in pred]

    # For each softmax prediction, we only label something
    # with a 1 if a) it is the higher values and b) if it is
    # also >= the threshold value. This allows more constraint
    # for labeling something as 1.
    pred_labels = []
    for probs in pred:
        if probs[1] > threshold and probs[1] > probs[0]:
            pred_labels.append(1)
        else:
            pred_labels.append(0)

    # return labels
    return pred_labels

####################################################

# Bag of words classifier
# Takes the dataframe, emotion list and other tuning parameters as input
# returns a dictionary of the evaluation metrics for this run for each emotion.
# This model is from the initial CountVectorizer activity we did and was not
# really considered in our actual testing. 
def BagOfWords(df, emotions, epochs, lr, resample, batch_size = 0, threshold=0.5):
    retDict = {}

    # Set training data to be the count vectors for each tweet.
    corpus = df["processed_tweets"]
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)

    for emotion in emotions:
        print("    "+emotion)

        # get train test split
        X_train, X_test, y_train, y_test = train_test_split(
            X.toarray(), df[emotion], test_size=0.2, random_state=65
        )  

        # resample if necessary
        if resample:
            X_train, y_train = resamp(pd.DataFrame({"Col": X_train.tolist()}), y_train)

        # convert everything to a tensor
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(np.array(y_train.tolist()))
        y_test = torch.tensor(np.array(y_test.tolist()))

        # Add evaluation metric entry to return dictionary
        retDict[emotion] = run_mlp(X_train, y_train, X_test, y_test, epochs, lr, batch_size, threshold)

    return retDict

# Bert binary output
# Takes the dataframe, emotion list and other tuning parameters as input
# returns a dictionary of the evaluation metrics for this run for each emotion.
def bert(df, emotions, epochs, lr, resample, batch_size = 0, threshold = 0.5):
    retDict = {}

    # Using multilingual uncased
    model_name = 'bert-base-multilingual-uncased'
    for emotion in emotions:
        
        # Initialize the model and the model tokenizer
        tokenizer = BertTokenizer.from_pretrained(model_name)
        transformers.logging.set_verbosity_error()
        model = BertForSequenceClassification.from_pretrained(model_name)
    
        # split data into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            df["processed_tweets"], df[emotion], test_size=0.2, random_state=65
        )

        # Resample if necessary
        if resample:
            X_train, y_train = resamp(X_train, y_train)
        else:
            X_train = X_train.tolist()

        # Get tokenizer encodings for the test and training set
        X_train_encoded_inputs = tokenizer(X_train, padding=True, truncation=True, return_tensors="pt")
        X_test_encoded_inputs = tokenizer(X_test.tolist(), padding=True, truncation=True, return_tensors="pt")
        
        # Define input_id and attention mask list
        X_train_input_ids = X_train_encoded_inputs["input_ids"]
        X_train_attention_mask = X_train_encoded_inputs["attention_mask"]
        X_test_input_ids = X_test_encoded_inputs["input_ids"]
        X_test_attention_mask = X_test_encoded_inputs["attention_mask"]

        # Convert labels to tensors
        y_train = torch.tensor(np.array(y_train.tolist()))
        y_test = torch.tensor(np.array(y_test.tolist()))

        # accessing GPU if it is available
        
        if torch.cuda.is_available():
            device = get_gpu_device()
            X_train_input_ids = X_train_input_ids.to(device)
            X_train_attention_mask = X_train_attention_mask.to(device)
            X_test_input_ids = X_test_input_ids.to(device)
            X_test_attention_mask = X_test_attention_mask.to(device)
            y_train = y_train.to(device)
            y_test = y_test.to(device)
        
        # define optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        # prep batch sizing if necessary
        if batch_size != 0:
            d = TensorDataset(X_train_input_ids, X_train_attention_mask, y_train)
            dataloader = DataLoader(d, batch_size=batch_size, shuffle=True)

        model.train()
        for i in tqdm(range(epochs), desc = "      Epoch"):
            
            # use try-catch to iterate over batches if necessary
            try:
                for X_train_input_ids_batch, X_train_attention_mask_batch, \
                    y_train_batch in tqdm(dataloader, desc = "          Batch"):

                    outputs = model(input_ids=X_train_input_ids_batch, 
                                    attention_mask=X_train_attention_mask_batch,
                                    labels=y_train_batch)
                    
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()
            except NameError:
                outputs = model(input_ids=X_train_input_ids, 
                                    attention_mask=X_train_attention_mask,
                                    labels=y_train)
                loss = outputs.loss
                loss.backward()
                optimizer.step()


        # Set the model in evaluation mode
        model.eval()

        # Pass the testing input tensors through the BERT model for inference
        with torch.no_grad():
            test_outputs = model(input_ids=X_test_input_ids, 
                                 attention_mask=X_test_attention_mask,
                                 labels=y_test)

        # Apply softmax and obtain the predicted probabilities for testing data
        softmax = torch.nn.Softmax(dim=1)
        test_predicted_probabilities = softmax(test_outputs.logits).tolist()
        test_predicted_labels=[]

        # Check for highest logit value and whether value is higher than threshold
        for logit in test_predicted_probabilities:
            if logit[1] > threshold and logit[1] > logit[0]:
                test_predicted_labels.append(1)
            else:
                test_predicted_labels.append(0)

        # get metric dictionary
        retDict[emotion]=evalDict(test_predicted_labels.tolist(), y_test.tolist())

    return retDict

# Bert binary *vector* output 
# Takes the dataframe, emotion list and other tuning parameters as input
# returns a dictionary of the evaluation metrics for this run for each emotion.
def bert2(df, emotions, epochs, lr, resample, batch_size = 0, best_model_emotion = "", threshold=0.5):
    
    # get matrix of the emotions so that we can consider the vector
    # bianry labels rather than just each label.
    emotions_mat = df[emotions].values
    
    retDict = {}

    # Initialize bert model
    model_name = 'bert-base-multilingual-uncased'
    transformers.logging.set_verbosity_error()
    bert_model = BertModel.from_pretrained(model_name, num_labels=6) # Note the 6 output nodes
    
    # Freeze model outputs per Helge's suggestion to minimize the computation time
    for param in bert_model.parameters():
        param.requires_grad = False

    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification(bert_model.config)

    # Enable gradient computation for the parameters of the last layer
    for param in model.classifier.parameters():
        param.requires_grad = True

    # split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        df["processed_tweets"], emotions_mat, test_size=0.2, random_state=65
    )

    # needs to be a list
    X_train = X_train.tolist()

    # define the encoding dfs with the bert tokenizer
    X_train_encoded_inputs = tokenizer(X_train, padding=True, truncation=True, return_tensors="pt")
    X_test_encoded_inputs = tokenizer(X_test.tolist(), padding=True, truncation=True, return_tensors="pt")
    
    # get input ids and attention masks
    X_train_input_ids = X_train_encoded_inputs["input_ids"]
    X_train_attention_mask = X_train_encoded_inputs["attention_mask"]
    X_test_input_ids = X_test_encoded_inputs["input_ids"]
    X_test_attention_mask = X_test_encoded_inputs["attention_mask"]

    # convert labels to tensors
    y_train = torch.tensor(y_train)
    y_test = torch.tensor(y_test)

    # accessing GPU if it is available
    if torch.cuda.is_available():
        device = get_gpu_device()
        X_train_input_ids = X_train_input_ids.to(device)
        X_train_attention_mask = X_train_attention_mask.to(device)
        X_test_input_ids = X_test_input_ids.to(device)
        X_test_attention_mask = X_test_attention_mask.to(device)
        y_train = y_train.to(device)
        #y_test = y_test.to(device)
        model = model.to(device)

    # define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Define the binary cross-entropy loss function
    criterion = nn.BCEWithLogitsLoss()

    # set up batching if necessary
    if batch_size != 0:
        d = TensorDataset(X_train_input_ids, X_train_attention_mask, y_train)
        dataloader = DataLoader(d, batch_size=batch_size, shuffle=True)

    for i in tqdm(range(epochs), desc = "      Epoch"):
    #for i in range(epochs):
        model.train()

        # use try-catch to iterate over batches if necessary
        try:
            for X_train_input_ids_batch, X_train_attention_mask_batch, \
                y_train_batch in dataloader:
                outputs = model(input_ids=X_train_input_ids_batch, 
                                attention_mask=X_train_attention_mask_batch)
                #loss = torch.sum(torch.square(outputs.logits - y_train_batch))
                loss = criterion(outputs.logits, y_train_batch.float())
                # for i in range(len(emotions)):
                #     torch.nn.functional.cross_entropy(outputs[:,i], y_train_batch[:,i]) 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
        except NameError:
            outputs = model(input_ids=X_train_input_ids, 
                                attention_mask=X_train_attention_mask,
                                labels=y_train)
            #loss = torch.sum(torch.square(outputs.logits - y_train))
            loss = outputs.loss
            loss = criterion(outputs, y_train.float())
            # for i in range(len(emotions)):
            #         torch.nn.functional.cross_entropy(outputs[:,i], y_train_batch[:,i]) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # if we are just training out best model, we can save it here
    if best_model_emotion != "":
        torch.save(model.state_dict(), f"best_models/bert2_{best_model_emotion}.pt")
        return
    else:
        # Set the model in evaluation mode
        model.eval()

        # Pass the testing input tensors through the BERT model for inference
        with torch.no_grad():
            test_outputs = model(input_ids=X_test_input_ids, 
                                    attention_mask=X_test_attention_mask)

        # Apply sigmoid and obtain the predicted probabilities for testing data
        sigmoid = torch.nn.Sigmoid()
        test_predicted_probabilities = sigmoid(test_outputs.logits)
        test_predicted_labels = []

        # for each output vector, if a value if greater than the threshold 
        # we set it to 1, otherwise we set it to 0
        for vect in test_predicted_probabilities:
            l = vect.tolist()
            for i in range(len(l)):
                if l[i] >= threshold:
                    l[i] = 1
                else:
                    l[i] = 0
            test_predicted_labels.append(l)

        # for each index of each vector, we evaluate the respective emotion of that index
        for i, emotion in enumerate(emotions):
            emotion_pred_labels = []
            y_test_labels = []
            for vect, row in zip(test_predicted_labels, y_test):
                emotion_pred_labels.append(vect[i])
                y_test_labels.append(row[i])
            retDict[emotion]=evalDict(emotion_pred_labels, y_test_labels)

        return retDict

# Muse model
# Takes the dataframe, emotion list and other tuning parameters as input
# returns a dictionary of the evaluation metrics for this run for each emotion.
MUSE_FT = []
def get_muse_ft():
    fasttext.util.download_model("ar", if_exists="ignore")
    if len(MUSE_FT) == 0:
        MUSE_FT.append(load_model("cc.ar.300.bin"))
    return MUSE_FT[0]

def muse(df, emotions, epochs, lr, resample, batch_size = 0, best_model_emotion = "",threshold=0.5):
    retDict = {}

    # Download Arabic vectors from MUSE
    ft = get_muse_ft()

    for emotion in emotions:
        runModel = True

        # if we are just getting the bets model for a specific
        # emotion, we don't have to loop over every emotion
        if best_model_emotion != "" and emotion != best_model_emotion:
            runModel = False
        
        if runModel:

            # split data into train and test
            X_train, X_test, y_train, y_test = train_test_split(
                df["processed_tweets"], df[emotion], test_size=0.2, random_state=65
            )

            # resample if necessary
            if resample:
                X_train, y_train = resamp(X_train, y_train)
            else:
                X_train = X_train.tolist()

            # Transform the data to embeddings
            X_train_transformed = np.array([ft.get_sentence_vector(x) for x in X_train])
            X_test_transformed = np.array([ft.get_sentence_vector(x) for x in X_test])

            # Convert to tensors
            X_train_tensor = torch.tensor(X_train_transformed, dtype=torch.float32)
            X_test_tensor = torch.tensor(X_test_transformed, dtype=torch.float32)
            y_train_tensor = torch.tensor(np.array(y_train.tolist()))
            y_test_tensor = torch.tensor(np.array(y_test.tolist()))

            # if we are saving a best model, we pass that to the run_mlp function
            if best_model_emotion!="":
                best_model_name = f"best_models/muse_{best_model_emotion}.pt"
                bm_flag = 1
            else:
                best_model_name = ""
                bm_flag = 0
            
            # run the mlp function
            retDict[emotion] = run_mlp(X_train_tensor, y_train_tensor, X_test_tensor, 
                                    y_test_tensor, epochs, lr, batch_size, bm_flag, best_model_name, threshold)

    return retDict

# GloVe
# Takes the dataframe, emotion list and other tuning parameters as input
# returns a dictionary of the evaluation metrics for this run for each emotion.

# We define a global variable here because, assuming consistent sample sizing
# for the corpus, the transformation will be the same across all parameter testing
# so we don't want to be redefining it every time we run.
GLOVE_TRANSFORMED = []
def GloVe_transform(df):
    global GLOVE_TRANSFORMED
    if len(df) != len(GLOVE_TRANSFORMED):
        fasttext.util.download_model("ar", if_exists="ignore")
        fasttext_model = load_model("cc.ar.300.bin")

        indices = {}
        embeddings = []

        i = 0
        # get embeddings
        for word in tqdm(fasttext_model.words):
            if word not in indices:
                indices[word] = i
                i += 1
            embeddings.append(fasttext_model[word])
        transformed = []

        # get transformations
        for tweet in tqdm(tokenize(df["processed_tweets"])):
            sentence = []
            for word in tweet:
                if word in indices:
                    sentence.append(embeddings[indices[word]])
            transformed.append(sum(sentence))
        GLOVE_TRANSFORMED = transformed
    return GLOVE_TRANSFORMED

def GloVe(df, emotions, epochs, lr, resample, batch_size = 0, best_model_emotion = "",threshold=0.5):
    transformed = GloVe_transform(df)

    retDict = {}
    #for emotion in tqdm(emotions, desc="      Emotions"):
    for emotion in emotions:
        runModel = True
        if best_model_emotion!="" and emotion != best_model_emotion:
            runModel = False
        
        if runModel:

            #print("    "+emotion)

            # split data into train and test
            X_train, X_test, y_train, y_test = train_test_split(
                transformed, df[emotion], test_size=0.2, random_state=65
            )

            # resample if necessary
            if resample:
                X_train, y_train = resamp(pd.DataFrame({"Col": X_train}), y_train)

            # convert things to tensors
            X_train = torch.tensor(np.array(X_train))
            X_test = torch.tensor(np.array(X_test))
            y_train = torch.tensor(np.array(y_train.tolist()))
            y_test = torch.tensor(np.array(y_test.tolist()))

            if best_model_emotion!="":
                best_model_name = f"best_models/GloVe_{best_model_emotion}.pt"
                bm_flag = 1
            else:
                best_model_name = ""
                bm_flag = 0
            
            # run the mlp function
            retDict[emotion] = run_mlp(X_train, y_train, X_test, 
                                    y_test, epochs, lr, batch_size, bm_flag, best_model_name, threshold)
    
    return retDict

# GPT2 implementation
# Takes the dataframe, emotion list and other tuning parameters as input
# returns a dictionary of the evaluation metrics for this run for each emotion.

# NB: this model took up a ton of memory on the server so we opted not to really use it
def GPT2(data, emotions, epochs, lr, resample, batch_size = 0, best_model_emotion = "", threshold=0.5):
    retDict = {}
    for emotion in emotions:

        # Define model and tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.padding = True
        tokenizer.pad_token = tokenizer.eos_token
        transformers.logging.set_verbosity_error()
        model = GPT2ForSequenceClassification.from_pretrained('gpt2')
        model.train()

        model_name = "gpt2"  # Replace with the specific GPT-2 model name if needed
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        # Set the padding token
        tokenizer.padding_side = "right"
        tokenizer.pad_token = tokenizer.eos_token

    
        # split data into train and test
        X_train, X_test, Y_train, Y_test = train_test_split(data['processed_tweets'], 
                                                            data[emotion], test_size=0.2, random_state=65)

        if resample:
            X_train, Y_train = resamp(X_train, Y_train)
        else:
            X_train = X_train.tolist()

        # Tokenize your text data
        X_train_tkn = tokenizer(X_train, padding=True, truncation=True, return_tensors="pt")
        X_test_tkn = tokenizer(X_test.tolist(), padding=True, truncation=True, return_tensors="pt")

        # Create input tensors
        X_train_input_ids = X_train_tkn["input_ids"]
        X_train_attention_mask = X_train_tkn["attention_mask"]
        X_test_input_ids = X_test_tkn["input_ids"]
        X_test_attention_mask = X_test_tkn["attention_mask"]

        Y_train = torch.tensor(np.array(Y_train.tolist()))
        Y_test = torch.tensor(np.array(Y_test.tolist()))

        # Pass the input tensors through the GPT-2 model
        with torch.no_grad():
            X_train_outputs = model(X_train_input_ids, attention_mask=X_train_attention_mask)
            X_train_embeddings = X_train_outputs.last_hidden_state[:, 0, :]
            
            X_test_outputs = model(X_test_input_ids, attention_mask=X_test_attention_mask)
            X_test_embeddings = X_test_outputs.last_hidden_state[:, 0, :]
        
        if emotion == best_model_emotion:
            best_model_name = f"best_models/GPT2_{best_model_emotion}.pt"
            bm_flag = 1
        else:
            best_model_name = ""
            bm_flag = 0
        retDict[emotion] = run_mlp(X_train_embeddings, Y_train, X_test_embeddings, 
                                   Y_test, epochs, lr, batch_size, bm_flag, best_model_name, threshold)
    return retDict

# Function for sensitivity testing
# takes the corpus, emotion list, and model as paramters
# and returns the filename of the results file
def sensitivity_testing(df, emotions, model, *args, **kwargs):
    model_name = model.__name__.split("(")[0]

    # define the global glove transform list if testing glove
    if model_name == "GloVe":
        print("Getting GloVe transforms")
        GloVe_transform(df)
    elif model_name == "muse":
        print("Getting muse ft")
        get_muse_ft()
    if "bert" in model_name:
        proc_num = 2
    else:
        proc_num = 20

    # outfile for results
    outfile = "output/paramTuning/{}.csv".format(model_name)

    # load in parameter file that will be looped over
    tuning_file = "data/tuning_params.json"
    with open(tuning_file, "r") as f:
        param_dict = pd.read_json(f)[model_name]

    # sort each parameter list
    epoch = sorted(param_dict["epoch"])
    lr = sorted([round(val, 6) for val in param_dict["lr"]])
    resample = sorted(param_dict["resample"])
    batch_size = sorted(param_dict["batch_size"])
    sample_size = sorted(param_dict["sample_size"])
    threshold = sorted([round(val, 4) for val in param_dict["threshold"]])

    #test_num = len(epoch) * len(lr) * len(resample) * len(batch_size) * len(sample_size) *len(threshold)
    
    # define iterable to loop over
    param_grid = list(itertools.product(sample_size, batch_size, lr, epoch, resample, threshold))

    # load previous testing data if it exists, otherwise create new output dataframe
    if os.path.exists(outfile):
        data_df = pd.read_csv(outfile)
    else:
        header = ["model", "sample_size", "resample", "epochs", "lr", "batch_size","threshold"]
        for emotion in emotions:
            header.append(emotion+"_accuracy")
            header.append(emotion+"_macro_f1")
        data_df = pd.DataFrame(columns = header)
    data_df['lr'] = pd.to_numeric(data_df['lr'], errors='coerce')

    # helper function to check if a parameter configuration has already been tested
    def check_for_already_tested(param_grid, data_df, model_name):
        retlist=[]
        for config in tqdm(param_grid):
            sample_size, batch_size, lr, epoch, resample, threshold = config
            row = [model_name, sample_size, resample, epoch, lr, batch_size, threshold]
            check_df = data_df[["model", "sample_size", "resample", "epochs", "lr", "batch_size","threshold"]]

            if not check_df.isin(pd.Series(row)).all(axis=1).any():
                retlist.append(config)

        return retlist
    print("\nRemove configurations that have already been tested")
    param_grid = check_for_already_tested(param_grid, data_df, model_name)

    # helper function to get the list of tuples to pass to the multi-processor
    def get_arglist(param_grid, df, emotions):
        arglist = []
        for config in tqdm(param_grid):
            sample_size, batch_size, lr, epoch, resample, threshold = config
            tup = (df,emotions,epoch,lr,resample,batch_size,"",threshold)
            arglist.append(tup)
        return arglist

    # most number of processes possible for GPUs for GLove was 20
    print("\nGetting argument list")
    arg = get_arglist(param_grid, df, emotions)
    batch_size = 3*proc_num
    batch_list = [arg[i:i+batch_size] for i in range(0, len(arg), batch_size)]

    # run multiprocessor
    print(f"\nRunning batch iterations. Processes/iteration = {proc_num}")
    for arglist in tqdm(batch_list, desc = "Batches"):
        with multiprocessing.Pool(processes=proc_num) as pool:
            results = pool.starmap(model, arglist)
        
        # get results for each configuration run and output
        rows = []
        print("    Getting Results")
        for result_dict, config in zip(results, param_grid):
            sample_size, batch_size, lr, epoch, resample, threshold = config
            row = [model_name, sample_size, resample, epoch, lr, batch_size, threshold]
            for emotion in emotions:
                row.append(result_dict[emotion]["accuracy"])
                row.append(result_dict[emotion]["macro f1"])
            rows.append(row)

        add_df = pd.DataFrame(rows, columns = data_df.columns)
        data_df = pd.concat([data_df, add_df], ignore_index=True)
        data_df.to_csv(outfile, index=False)
    
    return outfile

def get_best_params(file, emotions):
    # Load in best param file (if it exists)
    best_param_file = "output/Best_Params.csv"
    model_name = file.split("/")[2][:-4]
    best_param_df = pd.DataFrame()
    if os.path.exists(best_param_file):
        with open(best_param_file, "r") as f:
            best_param_df = pd.read_csv(f)
            best_param_df = best_param_df[best_param_df["model"]!=model_name]

    best_params = {
        "emotions":emotions,
        "epochs":[],
        "lr":[],
        "batch_size":[],
        "resample":[],
        "threshold":[]
    }

    # get a subset of the data that has the highest sample size
    # we don't want to get the accuracy of a test with a small sample size
    # as it will always be less reliable than a full sample size
    df = pd.read_csv(file)
    df = df[df["sample_size"] == df["sample_size"].max()]

    for emotion in emotions:

        # get the row(s) with the highest accuracy for the emotion
        best_row = df[df[f"{emotion}_accuracy"] == df[f"{emotion}_accuracy"].max()]
        # if there is a tie, use the higher f1 score
        if len(best_row) > 1:
            best_row = best_row[best_row[f"{emotion}_macro_f1"] == best_row[f"{emotion}_macro_f1"].max()]

        # format output dataframe
        row_df = best_row[best_row.columns[:7].tolist() + [f"{emotion}_accuracy",f"{emotion}_macro_f1"]].copy()
        row_df["emotion"] = emotion
        row_df = row_df.rename(columns={f"{emotion}_accuracy":"accuracy",f"{emotion}_macro_f1":"macro_f1"})
        if best_param_df.empty:
            best_param_df = row_df
        else:
            best_param_df = pd.concat([best_param_df, row_df])
        
        # append best configuration to return dictionary
        for key in ["epochs","lr","batch_size","resample","threshold"]:
            best_params[key].append(best_row[key])
        
        best_param_df.to_csv(best_param_file, index=False)

    best_params["sample_size"] = 1

    return best_params

def save_best_model(df, params, model, *args, **kwargs):
    # saving best params for a given model
    iterable = zip(params["emotions"],params["epochs"],params["lr"],params["batch_size"],params["resample"],params["threshold"])

    df = df.sample(frac=params["sample_size"])
    
    # for each configuration, run and save the best model.
    print(model.__name__)
    for emotion, epoch, lr, batch_size, resample, threshold in iterable:
        print(f"\n  emotion {emotion}; epochs {str(epoch.iloc[0])}; lr {str(lr.iloc[0])}; resample {str(resample.iloc[0])}; batch_size {str(batch_size.iloc[0])}; threshold {str(threshold.iloc[0])}")
        model(df, params["emotions"], epoch.iloc[0], lr.iloc[0], resample.iloc[0], int(batch_size.iloc[0]), best_model_emotion=emotion, threshold=threshold.iloc[0])

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", nargs="+", type=str, 
                        help="The name of the model you want to test", required=True)
    parser.add_argument("-s", "--sensitivityTest", nargs=1, type=int, default=0, choices=[0,1],
                        help="Whether you want to run sensitivity testing for the model")
    parser.add_argument("-f", "--filename", nargs=1, type=str, default="data/arabic_emotion_new_new.csv",
                        help="The path to your data file")
    args = parser.parse_args()

    emotions = ["anger", "fear", "sadness", "disgust", "joy", "anger-disgust"]
    
    filename = args.filename[0]
    df = preprocess(filename, emoji_flag=0)

    models = {
        "bert2":bert2,
        "muse":muse,
        "GloVe":GloVe,
        "GPT2":GPT2,
        "BagOfWords":BagOfWords
    }

    for m in args.model:
        model = models[m]
        if args.sensitivityTest[0]:
            file = sensitivity_testing(df, emotions, model)
        else:
            file = f"output/paramTuning/{model.__name__}.csv"
        best_params = get_best_params(file, emotions)
        print(f"Training Best {m} Models")
        save_best_model(df, best_params, model)

if __name__ == "__main__":
    main()
