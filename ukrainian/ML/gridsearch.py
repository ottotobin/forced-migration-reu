######
# ML Models Emotion Detection Grid Search for best parameters
# Logistic Regression, Decision Tree, KNN, NaiveBayes using sklearn
#
# Authors: Eliza Salamon, Apollo Callero, Kate Liggio
#####

# Import all your libraries.
import sys
sys.path += ['../', '../..']
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix , classification_report
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import category_encoders as ce
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
import json
import argparse
from googletrans import Translator
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from ML.ml_models import encode_and_vectorize_binary , encode_and_vectorize_multi_class
from helper_funcs import *

EMOTIONS = ['anger', 'fear', 'sadness', 'disgust', 'joy']


def encode_and_vectorize_binary(data_file, encode_emojis=False):
    '''
    input: 
        data file: string path to tweet data
        encode emojis: boolean on how to to handle emojis
    output:
        encoded: pd.DataFrame with tweet data and binary classifications for each emotion
        X: sparse matrix of features with a row for each tweets
    '''
    processed_df = preprocess(data_file,encode_emojis)
    #one hot encoding for emotions
    encoder = ce.OneHotEncoder(cols = ['emotion'], return_df=True, use_cat_names=True)
    encoded = encoder.fit_transform(processed_df)
    #get features
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(encoded['processed_tweets'])
    return encoded, X

def encode_and_vectorize_multi_class(data_file, encode_emojis):
    '''
    input: 
        data file: string path to tweet data
        encode emojis: boolean on how to to handle emojis
    output:
        proccessed: pd.DataFrame with tweet data and multi-classifications for each emotion
        X: sparse matrix of features with a row for each tweets
    '''
    processed_df = preprocess(data_file)
    labelencoder = LabelEncoder()
    processed_df['encoded_emotion'] = labelencoder.fit_transform(processed_df['emotion'])
    #get features
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(processed_df['processed_tweets'])
    return processed_df, X

def dec_tree_gridsearch_binary(encoded , X):
    '''
    input: 
        encoded: pd.DataFrame with tweet data and binary classifications for each emotion
        X: sparse matrix feature values
    output: 
        this file writes the best decision tree model to a json file
    max depth is working best with large values , but this will cause the model to over fit thus I keep max_depth low-ish
    '''
    params = dict(max_depth=[2,3, 4 , 8 , 12, 16] , criterion=['gini', 'entropy'] , 
                  splitter=["sbest", "random"],min_samples_split=[3,4,5,6],
                  min_samples_leaf=[1,2,3] , max_features=[ "sqrt", "log2"])
    best_trees = {}
    for emotion in EMOTIONS:
        label_col = 'emotion_' + emotion
        labels = encoded[label_col]
        X_train, X_test, Y_train, Y_test = train_test_split(X, labels, test_size=0.2)
        gs_tree = GridSearchCV(DecisionTreeClassifier(),param_grid=params, cv=5 , scoring='f1')
        best_tree = gs_tree.fit(X_train, Y_train).best_estimator_
        y_pred = best_tree.predict(X_test)
        best_trees[emotion + '_decision_tree'] = [ best_tree.get_params() , classification_report(Y_test, y_pred , output_dict=True) ]
    with open("data/best_decision_trees_binary.json", "w") as outfile:
        json.dump(best_trees, outfile , ensure_ascii=False,indent=2) 

def dec_tree_gridsearch_multiclass(encoded , X):
    '''
    input: 
        encoded:  pd.DataFrame with tweet data and encoded classifications for each emotion
        X: sparse matrix feature values
    output: 
        this file writes the best decision tree model to a json file
    '''
    params = dict(criterion=['gini', 'entropy'] , 
                  splitter=["sbest", "random"],min_samples_split=[ 1,2,3,4,5,6],
                  min_samples_leaf=[1] , max_features=[ "sqrt", "log2"])
    labels = encoded['encoded_emotion']
    X_train, X_test, Y_train, Y_test = train_test_split(X, labels, test_size=0.2)
    gs_tree = GridSearchCV(DecisionTreeClassifier(),param_grid=params, cv=5)
    best_tree = gs_tree.fit(X_train, Y_train).best_estimator_
    y_pred = best_tree.predict(X_test)
    with open("data/best_decision_tree_multi_class.json", "w") as outfile:
        json.dump([best_tree.get_params() , classification_report(Y_test , y_pred,output_dict=True)] ,  outfile , ensure_ascii=False,indent=4) 

def random_forest_gridsearch_binary(encoded , X):
    '''
    input: 
        encoded: pd.DataFrame with tweet data and binary classifications for each emotion
        X: sparse matrix feature values
    output: 
        this file writes the best random forest model and its metrics to a json file
    '''
    params = {'n_estimators': [ 10 , 20 ,30 , 40 , 50],
               'min_samples_split': [2,5,10],
               'bootstrap': [True, False],
               'criterion' :["gini", "entropy", "log_loss"]}
    best_trees = {}
    for emotion in EMOTIONS:
        print('work on ' , emotion)
        label_col = 'emotion_' + emotion
        labels = encoded[label_col]
        X_train, X_test, Y_train, Y_test = train_test_split(X, labels, test_size=0.2)
        gs_forest = GridSearchCV(RandomForestClassifier(),param_grid=params, cv=5 , scoring='f1')
        best_forest = gs_forest.fit(X_train, Y_train).best_estimator_
        y_pred = best_forest.predict(X_test)
        best_trees[emotion + '_random_forest'] = best_forest.get_params() ,classification_report(Y_test, y_pred,output_dict=True)
    with open("data/best_random_forest_binary.json", "w") as outfile:
        json.dump(best_trees, outfile , ensure_ascii=False,indent=4) 

def random_forest_gridsearch_multi(encoded , X):
    '''
    input: 
        encoded: pd.DataFrame with tweet data and encoded classifications for each emotion
        X: sparse matrix feature values
    output: 
        this file writes the best decision random forest model to a json file
    '''
    params = {'n_estimators': [ 10 , 20 ,30 , 40 , 50],
               'max_features': ['auto', 'sqrt'],
               'min_samples_split': [2,5,10],
               'min_samples_leaf': [1, 2, 4],
               'bootstrap': [True, False],
               'criterion' :["gini", "entropy", "log_loss"]}
    labels = encoded['encoded_emotion']
    X_train, X_test, Y_train, Y_test = train_test_split(X, labels, test_size=0.2)
    gs_forest = GridSearchCV(RandomForestClassifier(),param_grid=params, cv=5)
    best_forest = gs_forest.fit(X_train, Y_train).best_estimator_
    y_pred = best_forest.predict(X_test)
    with open("data/best_random_forest_multi_class.json", "w") as outfile:
        json.dump( [ best_forest.get_params() ,classification_report(Y_test, y_pred,output_dict=True)] ,  outfile , ensure_ascii=False,indent=4) 

def knn_gridsearch_binary(encoded , X):
    '''
    input: 
        encoded: pd.DataFrame with tweet data and binary classifications for each emotion
        X: sparse matrix feature values
    output: 
        this file writes the best knn model to a json file
    '''
    params = [{'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'leaf_size': [15, 20]}]
    best_knns = {}
    for emotion in EMOTIONS:
        label_col = 'emotion_' + emotion
        labels = encoded[label_col]
        X_train, X_test, Y_train, Y_test = train_test_split(X, labels, test_size=0.2)
        gs_knn = GridSearchCV(KNeighborsClassifier(),param_grid=params, cv=5 , scoring='f1')
        best_knn = gs_knn.fit(X_train, Y_train).best_estimator_
        y_pred = best_knn.predict(X_test)
        best_knns[emotion + '_knn'] = [ best_knn.get_params() , classification_report(Y_test, y_pred,output_dict=True) ]
    with open("data/best_knn_binary.json", "w") as outfile:
        json.dump(best_knns, outfile , ensure_ascii=False,indent=2) 

def knn_gridsearch_multiclass(encoded , X):
    '''
    input: 
        encoded: pd.DataFrame with tweet data and encoded classifications for each emotion
        X: sparse matrix feature values
    output: 
        this file writes the best knn model to a json file
    '''
    params = [{'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'leaf_size': [15, 20]}]
    labels = encoded['encoded_emotion']
    X_train, X_test, Y_train, Y_test = train_test_split(X, labels, test_size=0.2)
    gs_knn = GridSearchCV(KNeighborsClassifier(),param_grid=params, cv=5)
    best_knn= gs_knn.fit(X_train, Y_train).best_estimator_
    y_pred = best_knn.predict(X_test)
    with open("data/best_knn_multi_class.json", "w") as outfile:
        json.dump([best_knn.get_params() , classification_report(Y_test , y_pred,output_dict=True)] ,  outfile , ensure_ascii=False,indent=4) 

def svm_gridsearch_binary(encoded , X):
    '''
    input: 
        encoded: encoded: pd.DataFrame with tweet data and binary classifications for each emotion
        X: sparse matrix feature values
    output: 
        this file writes the best svm model to a json file
    '''
    params = {'C': [0.1, 1, 10, 100, 1000], 
                'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}  
    best_svms = {}
    for emotion in EMOTIONS:
        label_col = 'emotion_' + emotion
        labels = encoded[label_col]
        X_train, X_test, Y_train, Y_test = train_test_split(X, labels, test_size=0.2)
        gs_svm = GridSearchCV(SVC(),param_grid=params, cv=5 , scoring='f1')
        best_svm = gs_svm.fit(X_train, Y_train).best_estimator_
        y_pred = best_svm.predict(X_test)
        best_svms[emotion + '_svm'] = [ best_svm.get_params() , classification_report(Y_test, y_pred,output_dict=True) ]
    with open("data/best_svm_binary.json", "w") as outfile:
        json.dump(best_svms, outfile , ensure_ascii=False,indent=2) 
def svm_gridsearch_multiclass(encoded , X):
    '''
    input: 
        encoded: encoded: pd.DataFrame with tweet data and encoded classifications for each emotion
        X: sparse matrix feature values
    output: 
        this file writes the best svm model to a json file
    '''
    params = {'C': [3,5 ,7,10], 
                'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}   
    labels = encoded['encoded_emotion']
    X_train, X_test, Y_train, Y_test = train_test_split(X, labels, test_size=0.2)
    gs_svc = GridSearchCV(SVC(),param_grid=params, cv=5)
    best_svm= gs_svc.fit(X_train, Y_train).best_estimator_
    y_pred = best_svm.predict(X_test)
    with open("data/best_svm_multi_class.json", "w") as outfile:
        json.dump([best_svm.get_params() , classification_report(Y_test , y_pred,output_dict=True)] ,  outfile , ensure_ascii=False,indent=4) 

def lr_gridsearch_binary(encoded , X):
    '''
    input: 
        encoded: encoded: pd.DataFrame with tweet data and binary classifications for each emotion
        X: sparse matrix feature values
    output: 
        this file writes the best logistic regression model to a json file
    '''
    params = {'C': [0.01, 0.1, 1, 10],'penalty':['l1','l2']}
    best_lrs = {}
    for emotion in EMOTIONS:
        label_col = 'emotion_' + emotion
        labels = encoded[label_col]
        X_train, X_test, Y_train, Y_test = train_test_split(X, labels, test_size=0.2)
        gs_lr = GridSearchCV(LogisticRegression(),param_grid=params, cv=5 , scoring='f1')
        best_lr = gs_lr.fit(X_train, Y_train).best_estimator_
        y_pred = best_lr.predict(X_test)
        best_lrs[emotion + '_lr'] = [ best_lr.get_params() , classification_report(Y_test, y_pred,output_dict=True) ]
    with open("data/best_lr_binary.json", "w") as outfile:
        json.dump(best_lrs, outfile , ensure_ascii=False,indent=2) 
def lr_gridsearch_multiclass(encoded , X):
    '''
    input: 
        encoded: encoded: pd.DataFrame with tweet data and encoded classifications for each emotion
        X: sparse matrix feature values
    output: 
        this file writes the best logistic regression model to a json file
    '''
    params = {'C': [0.01, 0.1, 1, 10],'penalty':['l1','l2']}
    labels = encoded['encoded_emotion']
    X_train, X_test, Y_train, Y_test = train_test_split(X, labels, test_size=0.2)
    gs_lr = GridSearchCV(LogisticRegression(),param_grid=params, cv=5)
    best_lr= gs_lr.fit(X_train, Y_train).best_estimator_
    y_pred = best_lr.predict(X_test)
    with open("data/best_lr_multi_class.json", "w") as outfile:
        json.dump([best_lr.get_params() , classification_report(Y_test , y_pred,output_dict=True)] ,  outfile , ensure_ascii=False,indent=4) 



def main():
    parser = argparse.ArgumentParser(description="Emotion Detection ML MOdels")
    parser.add_argument('--emojis', help="True to encode emojis, False to not encode emojis", default=False)
    parser.add_argument("--model", help="ML model to build", default='all')
    parser.add_argument("--encode", help="Encoding categories as binary or not", default=True)
    parser.add_argument("--file")
    args = parser.parse_args()
    encode = True if (args.encode == 'True' or args.encode is True) else False
    emojis = True if args.emojis == 'True' else False
    data, features = encode_and_vectorize_binary(args.file, encode_emojis=emojis)
    #dec_tree_gridsearch_binary(data , features)
    random_forest_gridsearch_binary(data , features)
    #knn_gridsearch_binary(data , features)
    #svm_gridsearch_binary(data , features)
    #lr_gridsearch_binary(data, features)
    data, features = encode_and_vectorize_multi_class(args.file, encode_emojis=emojis)
    #dec_tree_gridsearch_multiclass(data , features)
    random_forest_gridsearch_multi(data , features)
    #knn_gridsearch_multiclass(data , features)
    #svm_gridsearch_multiclass(data , features)
    #lr_gridsearch_multiclass(data, features)
if __name__ == "__main__":
    main()

