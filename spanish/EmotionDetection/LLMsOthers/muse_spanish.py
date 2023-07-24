import torch
import fasttext.util
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import emoji
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import precision_score
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin
import torch.nn.functional as F

# Set the target emotions for binary classification
EMOTIONS = ['anger', 'fear', 'sadness', 'joy']

loss_fn = torch.nn.BCEWithLogitsLoss()

def processData(data):
    data_filter = data[data['emotion'].isin(EMOTIONS)]
    tweets = data_filter['tweet']
    tweets = [emoji.demojize(str(tweets), language='es').replace(":", " ").replace("_", " ") for tweet in tweets]
    return data_filter

def train_and_evaluate(X_train_tensor, X_test_tensor, Y_train_tensor, Y_test_tensor, learning_rate, epochs, momentum, batch_size):
    D_in = X_train_tensor.shape[1]
    num_emotions = len(EMOTIONS)

    # MLP model for multiple binary classification
    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, num_emotions),
        torch.nn.Sigmoid()
    )

    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    # Convert target labels to one-hot encoding
    Y_train_onehot = torch.zeros((Y_train_tensor.shape[0], num_emotions))
    Y_train_onehot[torch.arange(Y_train_onehot.shape[0]), Y_train_tensor.long()] = 1

    # Training
    for i in range(epochs):
        pred = model(X_train_tensor)
        loss = loss_fn(pred, Y_train_onehot)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_value = loss.item()

    # Convert target labels for test set to one-hot encoding
    Y_test_onehot = torch.zeros((Y_test_tensor.shape[0], num_emotions))
    Y_test_onehot[torch.arange(Y_test_onehot.shape[0]), Y_test_tensor.long()] = 1

    # Evaluation
    pred = model(X_test_tensor)
    predictions = (pred > 0.5).int()

    accuracy_dict = {}
    f1_dict = {}
    loss_dict = {}
    true_pos_dict = {}
    false_pos_dict = {}
    true_neg_dict= {}
    false_neg_dict = {}

    for i, emotion in enumerate(EMOTIONS):
        accuracy = accuracy_score(Y_test_onehot[:, i], predictions[:, i])
        f1 = f1_score(Y_test_onehot[:, i], predictions[:, i])

        accuracy_dict[emotion] = accuracy
        f1_dict[emotion] = f1

        loss_dict[emotion] = loss_value

        tn, fp, fn, tp = confusion_matrix(Y_test_onehot[:, i], predictions[:, i]).ravel()

        true_pos_dict[emotion] = tp
        false_pos_dict[emotion] = fp
        true_neg_dict[emotion]= tn
        false_neg_dict[emotion] = fn

    return accuracy_dict, f1_dict, loss_dict, true_pos_dict, false_pos_dict, true_neg_dict, false_neg_dict

def main():
    # Load data and preprocess tweets
    data = pd.read_csv('spanish_emotion.tsv', sep='\t', on_bad_lines='skip')
    data = processData(data)

    # Download MUSE data set
    fasttext.util.download_model('es', if_exists='ignore')
    ft = fasttext.load_model('cc.es.300.bin')

    variations = [{'learning_rate': 0.015, 'epochs': 10, 'momentum': 0.9, 'batch_size': 20}, 
                  {'learning_rate': 0.001, 'epochs': 10, 'momentum': 0.9, 'batch_size': 20},
                  {'learning_rate': 0.01, 'epochs': 10, 'momentum': 0.9, 'batch_size': 20},
                  {'learning_rate': 0.0015, 'epochs': 10, 'momentum': 0.9, 'batch_size': 20},
                  ]
  
    for vari in variations:
        print("Testing Variation:", vari)

        # Split data into train and test
        label_col = 'emotion'  # Replace with the column name representing binary labels

        sample_data = data.sample(n=300, replace=False)

        # Testing and training sets
        X_train, X_test, Y_train, Y_test = train_test_split(sample_data['tweet'], sample_data[label_col], test_size=0.1, random_state=42)

        # Transform the data to embeddings
        X_train_transformed = [ft.get_sentence_vector(str(x)) for x in X_train]
        X_test_transformed = [ft.get_sentence_vector(x) for x in X_test]

        # Encode labels to integers
        label_encoder = LabelEncoder()
        Y_train_encoded = label_encoder.fit_transform(Y_train)
        Y_test_encoded = label_encoder.transform(Y_test)

        # Convert to tensors
        X_train_tensor = torch.tensor(np.array(X_train_transformed))
        X_test_tensor = torch.tensor(np.array(X_test_transformed))

        Y_train_tensor = torch.tensor(Y_train_encoded, dtype=torch.float32)
        Y_test_tensor = torch.tensor(Y_test_encoded, dtype=torch.float32)

        learning_rate = vari['learning_rate']
        epochs = vari['epochs']
        momentum = vari['momentum']
        batch_size = vari['batch_size']

        accuracy_dict, f1_dict, loss_dict, true_pos_dict, false_pos_dict, true_neg_dict, false_neg_dict = train_and_evaluate(X_train_tensor, X_test_tensor, Y_train_tensor, Y_test_tensor, learning_rate, epochs, momentum, batch_size)

        for emotion in EMOTIONS:
            print("Emotion:", emotion)
            print("Accuracy:", accuracy_dict[emotion])
            print("F1 Score:", f1_dict[emotion])
            print("Loss:", loss_dict[emotion])
            print("True Positive: ",true_pos_dict[emotion] )
            print("False Positive: ",false_pos_dict[emotion])
            print("True Negative: ",true_neg_dict[emotion])
            print("False Positive: ",false_neg_dict[emotion])

        # Save results to a CSV file
        df_results = pd.DataFrame({'emotion': EMOTIONS, 'accuracy': [accuracy_dict[emotion] for emotion in EMOTIONS], 'f1': [f1_dict[emotion] for emotion in EMOTIONS], 'loss': [loss_dict[emotion] for emotion in EMOTIONS]})
        df_results.to_csv('parameter_results.csv', sep='\n', index=False)

if __name__ == '__main__':
    main()

