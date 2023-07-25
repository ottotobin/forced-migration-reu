'''
Neural Networks Models: BETO, binary models (fear oversampled)
Authors: Adeline Roza, Bernardo Medeiros, Colin Hwang, Mattea Whitlow

To run (for testing): python3 bert_binary.py --epochs 5 --test True
'''

from transformers import BertForSequenceClassification
import pandas as pd
import torch
import emoji
import sys
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from torch.optim import NAdam
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
import argparse
from imblearn.over_sampling import RandomOverSampler

# reads data and filters out tweets we don't want
def readData(path):
    data = pd.read_csv(path, sep='\t', on_bad_lines='skip')
    # data_filter = data[~data['emotion'].isin(['surprise', 'others', 'disgust'])].sample(n=300, replace=False)
    data_filter = data[~data['emotion'].isin(['surprise', 'disgust', 'others'])]
    return data_filter

# removes emojis from tweets
def removeEmojis(data):
    for i in range(len(data)):
        row = data.iloc[i]
        new_tweet = emoji.demojize(str(row['tweet']), language='es')
        row['tweet'] = new_tweet
    return data

# oversamples training data
def overSample(train_texts, train_labels):
    sampler = RandomOverSampler(random_state=42, sampling_strategy=0.66666)
    train_texts, train_labels = sampler.fit_resample(train_texts.reshape(-1, 1), train_labels)
    train_texts = train_texts.flatten()

    return train_texts, train_labels

# trains and evaluates model
def runModel(file_name, device, lr, batch_size, epochs, oversample, test_sample, emo_list):
    for emotion in emo_list:
        # read in data
        data_filter = readData(file_name)

        # getting sample of data if indicated by test_sample argument
        if test_sample == True:
            data_filter = data_filter.sample(n=300, replace=False)

        emotion_label = [1 if label == emotion else 0 for label in data_filter['emotion']]

        # removing emojis
        data_filter = removeEmojis(data_filter)

        # splitting data for training and testing sets
        train_texts, test_texts, train_labels, test_labels = train_test_split(data_filter['tweet'].values.astype('U'), emotion_label, stratify=emotion_label, test_size=0.2)
        
        # oversampling fear
        if emotion == 'fear':
            if oversample == True:
                print("Oversampling!")
                train_texts, train_labels = overSample(train_texts, train_labels)

        # encoding labels 
        label_encoder = LabelEncoder()
        encoded_train_labels = label_encoder.fit_transform(train_labels)
        encoded_test_labels = label_encoder.transform(test_labels)

        # initializing tokenizer
        tokenizer = BertTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-uncased')

        #tokenizing and encoding training data
        train_val = [str(i) for i in train_texts]
        train_encoding = tokenizer(train_val, return_tensors='pt', padding=True, truncation=True, max_length=512)
        train_input_ids = train_encoding['input_ids']
        train_attention_mask = train_encoding['attention_mask']
        train_labels = torch.LongTensor(encoded_train_labels.tolist())

        #encoding testing data
        test_val = [str(i) for i in test_texts]
        test_encoding = tokenizer(test_val, return_tensors='pt', padding=True, truncation=True, max_length=512)
        test_input_ids = test_encoding['input_ids']
        test_attention_mask = test_encoding['attention_mask']
        test_labels = torch.LongTensor(encoded_test_labels.tolist())

        # use tensordata set so we can separate data into batches later
        train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_labels)
        test_dataset = TensorDataset(test_input_ids, test_attention_mask, test_labels)

        #training model
        num_labels = len(label_encoder.classes_)
        model = BertForSequenceClassification.from_pretrained('dccuchile/bert-base-spanish-wwm-uncased', num_labels=num_labels, return_dict=True)

        model = model.to(device)

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [{'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                                        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}] 
        optimizer = NAdam(optimizer_grouped_parameters, lr=lr)

        try:
                
            # use dataloader to separate data into batches
            train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
            total_loss = 0.0
            y_true = []
            y_pred = []

            num_epochs = epochs

            #training in loop
            for epoch in range(num_epochs):
                for batch in tqdm(train_dataloader, desc="Batch Progress"):
                    batch_input_ids, batch_attention_mask, batch_labels = batch
                    batch_input_ids = batch_input_ids.to(device)
                    batch_attention_mask = batch_attention_mask.to(device)
                    batch_labels = batch_labels.to(device)

                    model.train()
                    outputs = model(batch_input_ids, attention_mask=batch_attention_mask,labels = batch_labels)
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    total_loss += loss.item()

                    logits = outputs.logits
                    batch_predictions = torch.argmax(logits, dim=1)
                    y_true.extend(batch_labels.tolist())
                    y_pred.extend(batch_predictions.tolist())
            #evaluation
            model.eval()
            eval_dataloader = DataLoader(test_dataset, batch_size = batch_size)
            eval_loss = 0.0
            eval_y_true = []
            eval_y_pred = []

            with torch.no_grad():
                for eval_batch in tqdm(eval_dataloader, desc = "Eval Batch Progress"):
                    eval_batch_input_ids, eval_batch_attention_mask, eval_batch_labels = eval_batch
                    eval_batch_input_ids = eval_batch_input_ids.to(device)
                    eval_batch_attention_mask = eval_batch_attention_mask.to(device)
                    eval_batch_labels = eval_batch_labels.to(device)
                    eval_outputs = model(eval_batch_input_ids, attention_mask=eval_batch_attention_mask, labels=eval_batch_labels)
                    eval_loss += eval_outputs.loss.item()
                    eval_logits = eval_outputs.logits
                    eval_batch_predictions = torch.argmax(eval_logits, dim=1)
                    eval_y_true.extend(eval_batch_labels.tolist())
                    eval_y_pred.extend(eval_batch_predictions.tolist())

            average_loss = total_loss / (num_epochs * len(train_dataloader))
            accuracy = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='weighted')
            cm = confusion_matrix(y_true, y_pred)

            eval_average_loss = eval_loss / len(eval_dataloader)
            eval_accuracy = accuracy_score(eval_y_true, eval_y_pred)
            eval_f1 = f1_score(eval_y_true, eval_y_pred, average='weighted')
            eval_cm = confusion_matrix(eval_y_true, eval_y_pred)

            print("** For " + emotion + ":")
            print("Batch size: ", batch_size)
            print("Training Metrics:")
            print("Average Loss: ", average_loss)
            print("Accuracy: ", accuracy)
            print("F1: ", f1)
            print("Confusion Matrix: \n", cm)
            print()
            print("Evaluation Metrics:")
            print("Average Loss: ", eval_average_loss)
            print("Accuracy: ", eval_accuracy)
            print("F1: ", eval_f1)
            print("Confusion Matrix: \n", eval_cm)
            print()
            torch.save(model.state_dict(), "model_bert_" + emotion + "." + "pt")

        except Exception as e:
            print("Error: " + str(e))

        model.eval()


def main():
    # direct output
    sys.stdout = open("bert_binary_output.txt", "w")

    # set up parser
    parser = argparse.ArgumentParser(description="Emotion Detection - BETO LLM")
    parser.add_argument("--filename", type=str, default='spanish_emotion.tsv', help="Training dataset file. Default is 'spanish_emotion.tsv'.")
    parser.add_argument("--learningrate", type=float, default=1e-5, help="Set learning rate. Default is 1e-5.")
    parser.add_argument("--batchsize", type=int, default=16, help="Set batch size. Default is 16.")
    parser.add_argument("--epochs", type=int, default=10, help="Set number of epochs. Default is 10.")
    parser.add_argument("--oversample", type=str, default=True, help="Boolean, oversamples training set for fear model if true. Default is true.")
    parser.add_argument("--test", type=str, default=False, help="Boolean, trains models on a small sample of data if true. Default is false.")
    args = parser.parse_args()

    oversample = True
    if args.oversample:
        if args.oversample == 'True':
            oversample = True
        elif args.oversample == 'False':
            oversample = False
        else:
            print("Invalid input for 'oversample' parameter. Using default.")
            oversample = True

    test = False
    if args.test:
        if args.test == 'True':
            test = True
        elif args.test == 'False':
            test = False
        else:
            print("Invalid input for 'test' parameter. Using default.")
            test = False

    # run model
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    emo_list = ['anger', 'fear', 'joy', 'sadness']
    runModel(args.filename, device, args.learningrate, args.batchsize, args.epochs, oversample, test, emo_list)
    sys.stdout.close()
    sys.stdout = sys.__stdout__

if __name__ == "__main__":
    main()
