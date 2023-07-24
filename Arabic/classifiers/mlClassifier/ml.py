"""
Authors: Lina Laghzaoui, Grace Magny-Fokam, Toby Otto, Rich Pihlstrom

Description:
    Program runs multiple classic machine learning algorithms to classify text into emotional data. (LogisticRegression, NaiveBayes, DecisionTree, RandomForest, KNNeighbors, SVC).
    Breaks a textual input into a vectorized 'bag-of-words' model, that is then inputed into the classifier versions of these models.
    Additionally, includes sensitivity testing and cross validation functions for parameter tuning.

Necessary Packages:
    You will need to install pytrends in order to use the API in this code:
        pip3 install nltk scikit-learn imblearn

filename: the path to the original datafile that you are using.

USAGE:
    python3 mlClassifier.py -filename <input_filename>

"""

import re
import pandas as pd
import numpy as np
import json
import argparse
import os
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, make_scorer

# processes a .tsv file to change emojis to text, while
# stripping flag and transport/ map symbols
def preprocess(data_file, emoji_flag):

    if os.path.exists("data/processed_data.csv"):
        with open("data/processed_data.csv", "r") as f:
            return pd.read_csv(f)
    
    else:
        #read text data
        df = pd.read_csv(data_file, sep='\t', encoding = 'utf8')

        #preprocess text - remove @s, #s, links, and emojis
        emoji_pattern = re.compile("["
                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                "]+", flags=re.UNICODE)
        processed_tweets = []

        for token_list in tokenize(df["Tweet"]):
            text = emoji_translator(token_list, emoji_flag)
            text = ' '.join(word for word in text.split() if not (word.startswith('http') or \
                                                                word.startswith('@') or \
                                                                    word.startswith('#')))
            text = emoji_pattern.sub(r'', text)
            text = text.lower()
            processed_tweets.append(text)

        df['processed_tweets'] = processed_tweets
        df["anger-disgust"] = df["anger"] + df["disgust"]
        df["anger-disgust"] = df["anger-disgust"].replace(2,1)

        df.to_csv("data/processed_data.csv")

        return df 

# tokenizes a list of tweets
def tokenize(tweets):
    regexp = RegexpTokenizer('\w+')
    tweets = tweets.map(str)
    token_col = tweets.apply(regexp.tokenize)
    return token_col

# translates emojis to Arabic text using emojis.csv
def emoji_translator(token_list, emoji_flag):
    
    return_tweet = ""
    emojis_df = pd.read_csv("lexicon/emojis.csv")

    for token in token_list:
        # checks to see if emoji_flag is True, otherwise just omits the emoji altogether
        if token in emojis_df["emoji"].values and emoji_flag:
            return_tweet += emojis_df[emojis_df['emoji'] == token]["text"].values[0]
        else:
            return_tweet += token
        return_tweet += " "

    return return_tweet

# takes in corpus as a list of tweets
# and turns it into a binary bag of words
def vectorize(corpus):
    vectorizer = CountVectorizer()
    return vectorizer.fit_transform(corpus)

# trains the model given 20% of the total corpus
def trainModel(X, labs):
    return train_test_split(X,labs,test_size=0.2)

# runs the model to make predictions based off of trained data
def runModel(train_list, model):
    clf = model.fit(train_list[0], train_list[2])

    pred_list = clf.predict(train_list[1])
    real_list = train_list[3]

    return (pred_list, real_list)

# returns a dictionary of evaluation metrics
def evalDict(pred, act):
    evalDict = {
        "macro precision":precision_score(act, pred, average="macro", zero_division=0),
        "macro recall":recall_score(act, pred, average="macro"),
        "macro f1":f1_score(act, pred, average="macro"),
        "micro precision":precision_score(act, pred, average="micro", zero_division=0),
        "micro recall":recall_score(act, pred, average="micro"),
        "micro f1":f1_score(act, pred, average="micro"),
        "accuracy":accuracy_score(act,pred)
    }
    return evalDict

# creates a .csv for the output of the given model with accompanying eval metrics
# for each emotion in the data set
def run(df, emotions, model, columns, X, emoji_flag):
    eval_out = pd.DataFrame(columns=columns)

    for emotion in emotions:
        emotion_labels = df[emotion].tolist()

        train_list = trainModel(X, emotion_labels)
        
        pred, act = runModel(train_list, model)
        
        modelName = str(model).split("(")[0]

        row = [emotion, modelName]
        eval = evalDict(pred, act)
        for key in eval:
            row.append(eval[key])

        eval_out.loc[len(eval_out)] = row

    eval_out["emoji_flag"] = emoji_flag

    return eval_out

def sensitivityTesting(filename, modelList, paramDict, emotions, balance=True):
    df = preprocess(filename, emoji_flag=0)
    corpus = df["processed_tweets"].values.tolist()
    X = vectorize(corpus)

    scoring = {
        "Precision": make_scorer(precision_score),
        "Recall":make_scorer(recall_score),
        "Accuracy": make_scorer(accuracy_score),
        "F1": make_scorer(f1_score)
        }

    bestParams = {}

    for model in modelList:
        modelName = str(model).split("(")[0]
        print(modelName)
    
        bestParams[modelName]={}
        df_out = pd.DataFrame() 

        for emotion in emotions:
            print("   "+emotion)
            emotion_labels = df[emotion].tolist()

            X_train, X_test, y_train, y_test = train_test_split(X, emotion_labels, test_size=0.2)
            
            if balance:
                sampler = RandomUnderSampler(random_state=42)
                X_train, y_train = sampler.fit_resample(X_train, y_train)
            
            if model == RandomForestClassifier():
                clf = RandomizedSearchCV(model, paramDict[modelName], cv=5, scoring=scoring, refit="Accuracy")
            else:
                clf = GridSearchCV(model, paramDict[modelName], cv=5, scoring=scoring, refit="Accuracy")
            
            clf.fit(X_train, y_train)
            #predictions = clf.predict(X_test)

            #opt_params = clf.best_params_
            #bestParams[modelName][emotion]=opt_params

            df_emo = pd.DataFrame(clf.cv_results_)
            df_emo = df_emo.drop(["mean_fit_time","std_fit_time","mean_score_time","std_score_time"],axis=1)
            df_emo["emotion"] = emotion

            if df_out.empty:
                df_out = df_emo
            else:
                df_out = pd.concat([df_out, df_emo])
            
        bestParams[modelName] = summarize(df_out)

        outfile = "output/{}_paramTesting.csv".format(modelName)
        df_out.to_csv(outfile, index=False)
    
    with open("output/summary.json","w+") as o:
        json.dump(bestParams, o, indent = 4)

# prints the cross-validation scores as well as the
# mean score
def crossValidation(model, X, y, cv):
    scores = cross_val_score(model, X, y, cv=cv)
    print('Cross-validation array: ' + str(scores))
    print('Mean cross-validation score: ' + str(np.mean(scores)))

def summarize(df):
    retDict = {}
    for emotion in df["emotion"].unique():
        retDict[emotion] = {}
        for metric in ["Precision", "Recall", "Accuracy", "F1"]:
            key = "mean_test_"+metric
            retDict[emotion][metric]={
                "Best Mean "+metric:df[key].iloc[df[key].idxmax()],
                "Params":df["params"].iloc[df["mean_test_"+metric].idxmax()]
            }
    
    return retDict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename", nargs=1, type=str, default="data/arabic_emotion_new_new.csv",
                        help="The path to your data file")
    args = parser.parse_args()

    filename = args.filename[0]
    emotions = ['anger', 'fear', 'sadness', 'disgust', 'joy', 'anger-disgust']
    models = [LogisticRegression(), MultinomialNB(),\
              DecisionTreeClassifier(), RandomForestClassifier(),\
              KNeighborsClassifier(), SVC()]

    paramDict = {
        "LogisticRegression":{
            "C": [100, 10, 5, 4, 3, 2, 1.0, .5, .4, .3, .2, 0.1, 0.01],
            "max_iter":[1000, 5000, 10000]  
        },
        "MultinomialNB":{
            "alpha": [(i/20) for i in range(0, 20)]
        },
        "DecisionTreeClassifier":{
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random'],
            'max_depth': [None, 10, 20, 30],
        },
        "RandomForestClassifier":{
            'n_estimators': [10, 50, 100, 200],
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 5, 10],
        },
        "KNeighborsClassifier":{
            'n_neighbors': [i for i in range(26)],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'p': [1,2]
        },
        "SVC":{
            'C': [100, 10, 1.0, 0.1, 0.01],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
        }
    }

    sensitivityTesting(filename, models, paramDict, emotions)



if __name__ == '__main__':
    main()