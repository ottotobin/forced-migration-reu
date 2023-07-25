'''
Neural Networks Models: mBert

'''
from transformers import BertForSequenceClassification
import pandas as pd
import torch
from torch.optim import AdamW
import emoji
import sys
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from tqdm import tqdm
from sklearn.model_selection import train_test_split

#direct to output
sys.stdout = open("mbert_output.txt", "w")

#read in data
data = pd.read_csv('spanish_emotion.tsv', sep='\t', on_bad_lines='skip')
data_filter = data[~data['emotion'].isin(['surprise', 'others', 'disgust'])]
tweets = data_filter['tweet']
tweets = [emoji.demojize(str(tweets), language='es').replace(":", " ").replace("_", " ") for tweet in tweets]

# sampling
sample_data = data_filter.sample(n=300, replace=False)

# splitting data for training and testing sets
train_texts, test_texts, train_labels, test_labels = train_test_split(sample_data['tweet'], sample_data['emotion'], test_size=0.2)

# encoding labels 
label_encoder = LabelEncoder()
encoded_train_labels = label_encoder.fit_transform(train_labels)
encoded_test_labels = label_encoder.transform(test_labels)

# tokenizing
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')

#tokenizing and encoding training data
train_encoding = tokenizer(train_texts.tolist(), return_tensors='pt', padding=True, truncation=True, max_length=512)
train_input_ids = train_encoding['input_ids']
train_attention_mask = train_encoding['attention_mask']
train_labels = torch.LongTensor(encoded_train_labels.tolist())

#encoding testing data
test_encoding = tokenizer(test_texts.tolist(), return_tensors='pt', padding=True, truncation=True, max_length=512)
test_input_ids = test_encoding['input_ids']
test_attention_mask = test_encoding['attention_mask']
test_labels = torch.LongTensor(encoded_test_labels.tolist())

# use tensordata set so we can separate data into batches later
train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_labels)
test_dataset = TensorDataset(test_input_ids, test_attention_mask, test_labels)

#training model
num_labels = len(label_encoder.classes_)
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-uncased', num_labels=num_labels, return_dict=True)
model.train()

# optimizer
from transformers import AdamW
optimizer = AdamW(model.parameters(), lr=1e-5)

#no decay
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [{'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                                {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5, no_deprecation_warning=True)

#batch sizes to try out
batch_sizes = [8, 16, 32, 64]

try:
    for batch_size in batch_sizes:
        # use dataloader to separate data into batches
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
        total_loss = 0.0
        y_true = []
        y_pred = []

        num_epochs = 5

        #training in loop
        for epoch in range(num_epochs):
            for batch in tqdm(train_dataloader, desc="Batch Progress"):
                batch_input_ids, batch_attention_mask, batch_labels = batch

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
                eval_outputs = model(eval_batch_input_ids, attention_mask=eval_batch_attention_mask, labels=eval_batch_labels)
                eval_loss += eval_outputs.loss.item()
                eval_logits = eval_outputs.logits
                eval_batch_predictions = torch.argmax(eval_logits, dim=1)
                eval_y_true.extend(eval_batch_labels.tolist())
                eval_y_pred.extend(eval_batch_predictions.tolist())

        #training metrics
        average_loss = total_loss / (num_epochs * len(train_dataloader))
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        cm = confusion_matrix(y_true, y_pred)

        #evaluation metrics
        eval_average_loss = eval_loss / len(eval_dataloader)
        eval_accuracy = accuracy_score(eval_y_true, eval_y_pred)
        eval_f1 = f1_score(eval_y_true, eval_y_pred, average='weighted')
        eval_cm = confusion_matrix(eval_y_true, eval_y_pred)

        #printing results
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

except Exception as e:
    print("Error: " + str(e))

model.eval()


sys.stdout.close()
sys.stdout = sys.__stdout__
