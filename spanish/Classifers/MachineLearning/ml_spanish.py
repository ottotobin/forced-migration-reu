"""
Authors: Bernardo Medeiros, Colin Hwang, Mattea Whitlow, Adeline Roza

Last Update: 6/26/23

Parameters:
--gridsearch: perform gridsearch on parameters for decision trees and random forests
--emoji: preprocess emojis into text using emoji library

How to run:

- With default parameters:
python3 ml_spanish.py > output.txt

- With emoji proprocessing, grid search, combine disgust + anger, balancing the dataset:
python ml_spanish.py > output_grid_search.txt --gridsearch True --emoji True --balance True
"""

import numpy as np
from nltk import ngrams
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from nltk.tokenize import TweetTokenizer
from sklearn import tree
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.under_sampling import RandomUnderSampler
import argparse
import random
import emoji
import json
import os
import sys

# determine the baseline accuracy if our model predicted randomly
def randBaseline(data, isMultiClass=False):
    emotions = set(data['emotion'])
    gt_list = []
    est_list = []
    if isMultiClass == False:
        for emotion in emotions:
            for label in data['emotion']:
                gt_list.append(random.randint(0, 1))
                if label == emotion:
                    est_list.append(1)
                elif label != emotion:
                    est_list.append(0)
            # generate and print accuracy stats
            accuracy = accuracy_score(est_list, gt_list)
            print("Baseline accuracy, " + str(emotion) + ': ' + str(accuracy))
            print('--')
    elif isMultiClass == True:
        for label in data['emotion']:
            gt_list.append(random.randint(0, 6))
        est_list = sortLabels(data)
        # generate and print accuracy stats
        accuracy = accuracy_score(est_list, gt_list)
        print("Baseline accuracy: " + str(accuracy))
        print('--')

# import the data and remove surprise, others and disgust
def import_set(setName):
    data = pd.read_csv(setName, sep='\t', on_bad_lines='skip')
    data = data[(data['emotion'] != 'surprise') & (data['emotion'] != 'others') & (data['emotion'] != 'disgust')]
    return data

# turn the emotion labels into numerical categories
def sortLabels(data):
    label_list = []
    for label in data['emotion']:
        if label == 'anger':
            label = 0
        if label == 'sadness':
            label = 1
        if label == 'fear':
            label = 2
        if label == 'joy':
            label = 3
        label_list.append(label)
    return label_list

# perform cross validation on the results
def crossValidate(model, X, y, cv):
    scores = cross_val_score(model, X, y, cv=cv)
    print('Cross-validation array: ' + str(scores))
    print('Mean cross-validation score: ' + str(np.mean(scores)))
    return scores
    
# calculate accuracy, precision, recall and f1 scores
def getStats(pred, true, emotion):
    # generate and print confusion matrix stats
    tn, fp, fn, tp = confusion_matrix(pred, true).ravel()
    print("For " + str(emotion) + ":")
    print("True Positives: " + str(tp))
    print("True Negatives: " + str(tn))
    print("False Positives: " + str(fp))
    print("False Negatives: " + str(fn))
    if tp == 0:
        tp = 1
    if fp == 0:
        fp = 1
    if fn == 0:
        fn = 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = tp / (tp + ((1/2)*(fp+fn)))
    print("F1: " + str(f1))
    # generate and print accuracy stats
    accuracy = accuracy_score(pred, true)
    print("Accuracy: " + str(accuracy))
    return precision, recall, f1, accuracy

# combine anger and disgust into one emotion
def removeDisgust(data):
    i = 0
    for emotion_label in data["emotion"]:
        if emotion_label == 'disgust':
            emotion_label = 'anger'
        data["emotion"][i] = emotion_label
        i += 1
    return data

# preprocess emojis into text
def removeEmojis(tweet):
    tweet = emoji.demojize(str(tweet), language='es')
    tweet = tweet.replace(":", " ")
    tweet = tweet.replace("_", " ")
    return tweet

# find the best accuracy out of all the models tested
def compareAccuracy(all_model_stats, output_dir, file_addon, emotions):
    highest_accuracy = 0.0
    best_model = None
    best_emotion = None
    #writing to output file
    with open(os.path.join(output_dir, "best_accuracy" + file_addon + ".txt"), "w") as file:
        for emotion in emotions:
            for model in all_model_stats[emotion]:
                if all_model_stats[emotion][model] and all_model_stats[emotion][model]['accuracy'] > highest_accuracy:
                    highest_accuracy = all_model_stats[emotion][model]['accuracy']
                    best_model = model
                    best_emotion = emotion
        file.write("Highest Accuracy: " + str(highest_accuracy) + "\n")
        file.write("Best Model: " + best_model + "\n")
        file.write("Best Emotion: " + best_emotion + "\n")

# find the best f1 out of all the models tested
def compareF1(all_model_stats, output_dir, file_addon, emotions):
    highest_f1 = 0.0
    best_model = None
    best_emotion = None
    #writing to output file
    with open(os.path.join(output_dir, "best_f1" + file_addon + ".txt"), "w") as file:
        for emotion in emotions:
            for model in all_model_stats[emotion]:
                if all_model_stats[emotion][model] and all_model_stats[emotion][model]['f1'] > highest_f1:
                    highest_f1 = all_model_stats[emotion][model]['f1']
                    best_model = model
                    best_emotion = emotion
        file.write("Highest F1: " + str(highest_f1) + "\n")
        file.write("Best Model: " + best_model + "\n")
        file.write("Best Emotion: " + best_emotion + "\n")


# function that will train and test the models on various parameters
def evaluateModels(data, emotion, gridsearch, balance, isMultiClass = True):
    model_stats = {}
    vectorizer = CountVectorizer()
    sampler = RandomUnderSampler(random_state=42)
    # set up pipelines for each model
    pipelines = {
        "Logistic Regression": Pipeline([
            ('countvec', vectorizer),
            ('model', LogisticRegression(max_iter=2000)),
        ]),
        "One vs Rest Logistic Regression": Pipeline([
            ('countvec', vectorizer),
            ('model', OneVsRestClassifier(LogisticRegression(max_iter=2000))),
        ]),
        "Decision Tree": Pipeline([
            ('countvec', vectorizer),
            ('model', DecisionTreeClassifier()),
        ]),
        "Random Forest": Pipeline([
            ('countvec', vectorizer),
            ('model', RandomForestClassifier()),
        ]),
        "KNN": Pipeline([
            ('countvec', vectorizer),
            ('model', KNeighborsClassifier()),
        ]),
        "SVC": Pipeline([
            ('countvec', vectorizer),
            ('model', SVC()),
        ]),
        "XGBoost": Pipeline([
            ('countvec', vectorizer),
            ('model', XGBClassifier()),
        ]),
    }
    # set up parameter grids for grid search
    param_grids = {
        "Logistic Regression": {
            # 'model__penalty': ['l1', 'l2', 'elasticnet'],
            'model__C': [0.01, 0.1, 1, 10],
        },
        "One vs Rest Logistic Regression": {
            # 'model__estimator__penalty': ['l1', 'l2', 'elasticnet'],
            'model__estimator__C': [0.01, 0.1, 1, 10],
        },
        "Decision Tree": {
            'model__criterion': ['gini', 'entropy'],
            'model__splitter': ['best', 'random'],
            'model__max_depth': [None, 10, 20, 30],
        },
        "Random Forest": {
            'model__criterion': ['gini', 'entropy'],
            'model__n_estimators': [10, 50, 100, 200, 500],
            'model__max_depth': [None, 10, 20, 30],
        },
        "KNN": {
            'model__n_neighbors': list(range(5, 26)),
            'model__weights': ['uniform', 'distance'],
        },
        "SVC": {
            'model__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'model__C': [0.01, 0.1, 1, 10],
            'model__gamma': ['scale', 'auto'],
        },
        "XGBoost": {
            'model__n_estimators': [10, 50, 100, 200, 500],
            'model__max_depth': [None, 10, 20, 30],
            'model__learning_rate': [0.01, 0.1, 0.2]
        },
    }
    # get numerical labels
    if isMultiClass:
        emotion_label = sortLabels(data)
        print("Multi-class models:\n")
    else:
        emotion_label = [1 if label == emotion else 0 for label in data['emotion']]
        print("Binary Models:\n")
    # split into training and testing data sets
    X_train, X_test, y_train, y_test = train_test_split(data['tweet'].values.astype('U'), emotion_label, test_size=0.2, random_state=11)
    # use resampler
    if balance:
        X_train, y_train = sampler.fit_resample(X_train.reshape(-1, 1), y_train)
        X_train = X_train.flatten()
    # loop through each model and get stats
    for name, pipeline in pipelines.items():
        print("\n" + name + ":")
        # check that a param grid exists for model before doing grid search
        if gridsearch and name in param_grids:
            grid_search = GridSearchCV(pipeline, param_grids[name], cv=5)
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            predictions = best_model.predict(X_test)
            print("List of best parameters:")
            print(grid_search.best_params_)
        # if not grid search, fit and predict normally
        else:
            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)
        # get stats
        if isMultiClass:
            print(confusion_matrix(predictions, y_test))
            accuracy = accuracy_score(predictions, y_test)
            print("Accuracy: " + str(accuracy))
        else:
            precision, recall, f1, accuracy = getStats(predictions, y_test, emotion)
            model_stats[name] = {'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': accuracy}
        # cross-validation
        scores = crossValidate(pipeline, X_train, y_train, 5)
        print('--')
    return model_stats

def main():
    parser = argparse.ArgumentParser(description="Emotion Detection - Classic Machine Learning")
    parser.add_argument('--output_dir', type=str, required=True, help='Specify an output directory.')
    parser.add_argument('--input_file', type=str, required=True, help='Specify a (spanish) dataset with tweets and emotions (spanish_emotions.tsv) Lexicon')
    parser.add_argument("--gridsearch", help="Perform grid search for optimal parameters. Only for decision trees and random forests - True or False", default=False)
    parser.add_argument("--emoji", help="Perform emoji processing - set as true or false", default=False)
    parser.add_argument("--balance", help="Balance the dataset", default=False)
    args = parser.parse_args()
    # initialize output directory
    output_dir = args.output_dir
    # if the directory doesn't exist, make it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # variable for updating filenames
    file_addon = ''
    # initialize file addons for each argument specified
    if args.gridsearch:
        file_addon = file_addon + '_grid_search'  
    if args.emoji:
        file_addon = file_addon + '_no_emoji'
    if args.balance:
        file_addon = file_addon + '_balanced'
    # kinda unnecessary, initialize arg variables
    gridsearch = args.gridsearch 
    emoji = args.emoji
    balance = args.balance
    # importing labeled data set
    data = import_set(args.input_file)
    # list of emotions
    emotions = ['anger', 'fear', 'sadness', 'joy']
    # preprocess emjois into text
    if emoji:
        for i in range(len(data)):
            row = data.iloc[i]
            new_tweet = removeEmojis(row['tweet'])
            row['tweet'] = new_tweet
    # create dictionary that will hold all f1/accuracy scores
    all_model_stats = {}
    # redirect stdout to a file
    orig_stdout = sys.stdout
    f = open(os.path.join(output_dir, "output" + file_addon + ".txt"), 'w')
    sys.stdout = f
    # evalulate model for all emotions in binary case
    for emotion in emotions:
        model_stats = evaluateModels(data, emotion, gridsearch, balance, isMultiClass=False)
        all_model_stats[emotion] = model_stats
    # multi class case
    evaluateModels(data, emotion, gridsearch, balance)
    with open(os.path.join(output_dir, 'model_stats' + file_addon + '.json'), 'w') as file:
        json.dump(all_model_stats, file, indent=4, ensure_ascii=False)
    # get best accuracy and f1 score
    compareAccuracy(all_model_stats, output_dir, file_addon, emotions)
    compareF1(all_model_stats, output_dir, file_addon, emotions)
    # redirect stdout back to terminal
    sys.stdout = orig_stdout
    f.close()

if __name__ == "__main__":
    main()


