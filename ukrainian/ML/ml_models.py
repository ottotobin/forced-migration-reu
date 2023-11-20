######
# ML Models Emotion Detection
# Logistic Regression, Decision Tree, KNN, NaiveBayes using sklearn
#
# Authors: Eliza Salamon, Apollo Callero, Kate Liggio
#####

# Import all your libraries.
import pandas as pd
import numpy as np
import json
import argparse
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import category_encoders as ce
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from googletrans import Translator
from sklearn.svm import SVC
import sys
sys.path += ['../', '../..']

from helper_funcs import *

EMOTIONS = ['anger', 'fear', 'sadness', 'disgust', 'joy']
EMOTIONS2 = ['anger', 'fear', 'sadness', 'joy']

# List of ML models
models = [('decision tree' , DecisionTreeClassifier()),( 'random forest' , RandomForestClassifier(max_depth=2, random_state=0))
          ,  ('logistic regression' , LogisticRegression())]

#Logistic Regression - binary
def logreg_binary(encoded, X, combined=False):
    if combined:
        emotion_list = EMOTIONS2
    else:
        emotion_list = EMOTIONS

    accuracies = 0
    f1s = 0
    for emotion in emotion_list:
        label_col = 'emotion_' + emotion
        labels = encoded[label_col]
        X_train, X_test, Y_train, Y_test = train_test_split(X, labels, test_size=0.2)
        clf = LogisticRegression().fit(X_train,Y_train)
        print('for ' + emotion + ' : ')

        acc = np.mean(cross_val_score(clf, X, labels, cv=5))
        print('5 CV average accuracy:{:.3f}'.format(acc))
        accuracies += acc
        
        f1 = np.mean(cross_val_score(clf, X, labels, cv=5, scoring='f1'))
        print("5 CV average F1:{:.3f}".format(f1))
        f1s += f1
    
    print('Overall accuracy{:.3f}'.format(accuracies/len(emotion_list)))
    print('Overall f1:{:.3f}'.format(f1s/len(emotion_list)))

#Logistice Regression - multi-class
def logreg_multiclass(encoded, X):
    labels = encoded['encoded_emotion']
    X_train, X_test, Y_train, Y_test = train_test_split(X, labels, test_size=0.2)
    clf = LogisticRegression(multi_class='ovr', solver='liblinear').fit(X_train,Y_train)
    preds = clf.predict(X_test)
    print(confusion_matrix(Y_test, preds))
    print('accuracy: ', accuracy_score(Y_test, preds))
    print('f1', f1_score(Y_test, preds, average='weighted'))

# Notes:
# lots of false negatives
# some are better than baselines
# can do k-folds to improve

#Decision Tree - binary
def dectree_binary(encoded, X, combined = False):
    if combined:
        emotion_list = EMOTIONS2
    else:
        emotion_list = EMOTIONS

    accuracies = 0
    f1s = 0

    for emotion in emotion_list:
        label_col = 'emotion_' + emotion
        labels = encoded[label_col]
        X_train, X_test, Y_train, Y_test = train_test_split(X, labels, test_size=0.2)
        clf = DecisionTreeClassifier().fit(X_train,Y_train)
        print('for ' + emotion + ' : ')
        
        acc = np.mean(cross_val_score(clf, X, labels, cv=5))
        print('5 CV average accuracy:{:.3f}'.format(acc))
        accuracies += acc
        
        f1 = np.mean(cross_val_score(clf, X, labels, cv=5, scoring='f1'))
        print("5 CV average F1:{:.3f}".format(f1))
        f1s += f1
    
    print('Overall accuracy{:.3f}'.format(accuracies/len(emotion_list)))
    print('Overall f1:{:.3f}'.format(f1s/len(emotion_list)))


#Decision Tree - multi-class
def dectree_multiclass(encoded,X):
    labels = encoded['encoded_emotion']
    X_train, X_test, Y_train, Y_test = train_test_split(X, labels, test_size=0.2)
    clf = DecisionTreeClassifier().fit(X_train,Y_train)
    preds = clf.predict(X_test)
    print(confusion_matrix(Y_test, preds))
    print('accuracy: ', accuracy_score(Y_test, preds))
    print('f1: ', f1_score(Y_test, preds, average='weighted'))


# Notes:
# Can play with max_depth and types of split
# Accuracies are around the same
# More false positives than logistic regression

# K Nearest Neighbor - binary
def knn_binary(encoded, X, combined=False):
    if combined:
        emotion_list = EMOTIONS2
    else:
        emotion_list = EMOTIONS

    accuracies = 0
    f1s = 0

    for emotion in emotion_list:
        label_col = 'emotion_' + emotion
        labels = encoded[label_col]
        X_train, X_test, Y_train, Y_test = train_test_split(X, labels, test_size=0.2 , random_state=5)
        clf = KNeighborsClassifier().fit(X_train,Y_train)
        print('for ' + emotion + ' : ')
        
        acc = np.mean(cross_val_score(clf, X, labels, cv=5))
        print('5 CV average accuracy:{:.3f}'.format(acc))
        accuracies += acc
        
        f1 = np.mean(cross_val_score(clf, X, labels, cv=5, scoring='f1'))
        print("5 CV average F1:{:.3f}".format(f1))
        f1s += f1
    
    print('Overall accuracy{:.3f}'.format(accuracies/len(emotion_list)))
    print('Overall f1:{:.3f}'.format(f1s/len(emotion_list)))


# Notes:
# very sparse

# K Nearest Neighbor - multi-class
def knn_multiclass(encoded, X):
    labels = encoded['encoded_emotion']
    X_train, X_test, Y_train, Y_test = train_test_split(X, labels, test_size=0.2)
    clf = KNeighborsClassifier().fit(X_train,Y_train)
    preds = clf.predict(X_test)
    print(confusion_matrix(Y_test, preds))
    print('accuracy: ', accuracy_score(Y_test, preds))
    print('f1: ', f1_score(Y_test, preds, average='weighted'))

# Naive Bayes - binary
def bayes_binary(encoded, X, combined=False):
    if combined:
        emotion_list = EMOTIONS2
    else:
        emotion_list = EMOTIONS

    accuracies = 0
    f1s = 0

    for emotion in emotion_list:
        label_col = 'emotion_' + emotion
        labels = encoded[label_col]
        X_train, X_test, Y_train, Y_test = train_test_split(X.toarray(), labels, test_size=0.2)
        clf = GaussianNB().fit(X_train,Y_train)
        print('for ' + emotion + ' :')
        preds = clf.predict(X_test)
        print(confusion_matrix(Y_test, preds))
        
        acc = np.mean(cross_val_score(clf, X.toarray(), labels, cv=5))
        print('5 CV average accuracy:{:.3f}'.format(acc))
        accuracies += acc
        
        f1 = np.mean(cross_val_score(clf, X.toarray(), labels, cv=5, scoring='f1'))
        print("5 CV average F1:{:.3f}".format(f1))
        f1s += f1
    
    print('Overall accuracy{:.3f}'.format(accuracies/len(emotion_list)))
    print('Overall f1:{:.3f}'.format(f1s/len(emotion_list)))


# Naive Bayes - multi-class
def bayes_multiclass(encoded, X):
    labels = encoded['encoded_emotion']
    X_train, X_test, Y_train, Y_test = train_test_split(X.toarray(), labels, test_size=0.2)
    clf = GaussianNB().fit(X_train,Y_train)
    preds = clf.predict(X_test)
    print(confusion_matrix(Y_test, preds))
    print('accuracy: ', accuracy_score(Y_test, preds))
    print('f1: ', f1_score(Y_test, preds, average='weighted'))

# SVM - binary
def svm_binary(encoded, X, combined=False):
    kernel = 'linear'
    C = 1
    if combined:
        emotion_list = EMOTIONS2
    else:
        emotion_list = EMOTIONS

    accuracies = 0
    f1s = 0

    for emotion in emotion_list:
        label_col = 'emotion_' + emotion
        labels = encoded[label_col]
        X_train, X_test, Y_train, Y_test = train_test_split(X, labels, test_size=0.2)
        clf = SVC(kernel=kernel, C=C).fit(X_train,Y_train)
        print('for ' + emotion + ' : ')
        
        acc = np.mean(cross_val_score(clf, X, labels, cv=5))
        print('5 CV average accuracy:{:.3f}'.format(acc))
        accuracies += acc
        
        f1 = np.mean(cross_val_score(clf, X, labels, cv=5, scoring='f1'))
        print("5 CV average F1:{:.3f}".format(f1))
        f1s += f1
    
    print('Overall accuracy{:.3f}'.format(accuracies/len(emotion_list)))
    print('Overall f1:{:.3f}'.format(f1s/len(emotion_list)))


# SVM - multi-class
def svm_multiclass(encoded, X):
    kernel = 'linear'
    C = 1
    labels = encoded['encoded_emotion']
    X_train, X_test, Y_train, Y_test = train_test_split(X, labels, test_size=0.2)
    clf = SVC(kernel=kernel, C=C, decision_function_shape='ovo').fit(X_train,Y_train)
    preds = clf.predict(X_test)
    print(confusion_matrix(Y_test, preds))
    print('accuracy: ',accuracy_score(Y_test, preds))
    print('f1: ', f1_score(Y_test, preds, average='weighted'))

# Random Forest - binary
def random_forest_binary(encoded, X, combined=False):
    if combined:
        emotion_list = EMOTIONS2
    else:
        emotion_list = EMOTIONS

    accuracies = 0
    f1s = 0

    for emotion in emotion_list:
        label_col = 'emotion_' + emotion
        labels = encoded[label_col]
        X_train, X_test, Y_train, Y_test = train_test_split(X, labels, test_size=0.2)
        clf = RandomForestClassifier().fit(X_train,Y_train)
        print('for ' + emotion + ' : ')
        
        acc = np.mean(cross_val_score(clf, X, labels, cv=5))
        print('5 CV average accuracy:{:.3f}'.format(acc))
        accuracies += acc
        
        f1 = np.mean(cross_val_score(clf, X, labels, cv=5, scoring='f1'))
        print("5 CV average F1:{:.3f}".format(f1))
        f1s += f1
    
    print('Overall accuracy{:.3f}'.format(accuracies/len(emotion_list)))
    print('Overall f1:{:.3f}'.format(f1s/len(emotion_list)))

        

# Random Forest - multi-class
def random_forest_multiclass(encoded, X):
    labels = encoded['encoded_emotion']
    X_train, X_test, Y_train, Y_test = train_test_split(X, labels, test_size=0.2)
    clf = RandomForestClassifier().fit(X_train,Y_train)
    preds = clf.predict(X_test)
    print(confusion_matrix(Y_test, preds))
    print('accuracy: ', accuracy_score(Y_test, preds))
    print('f1: ', f1_score(Y_test, preds, average='weighted'))


# Outputs info about mislabeled tweets to separate json file
def mislabel_analysis(tweet_data , features , model):
    '''
    input:
        tweet data: df of tweets
        features: sparse array of features
        model: the machine learning model to use for classification
    output:
        this function throws all mislabeled tweets into a json file
        #first item in the list is the real emotion on false positives , and the predicted emotion on false negatives
        e.g
        {
            anget_FP: [ [ ['other'] , 'Game of thrones is awesome I'm not angry' ] ] 
            anger_FN: [  ['disgust'] , 'I'm mad Im gonna hulk smash'  ] ]
        }
    '''

    #get a dataframe with a binary classifcation for each model
    indices = np.arange(1519)
    label_col = 'emotion_' + EMOTIONS[0]
    labels = tweet_data[label_col]
    X_train, X_test, Y_train, Y_test , train_idx,test_idx = train_test_split(features, labels,indices, test_size=0.2 , random_state=5)
    test_tweets = tweet_data[~tweet_data.index.isin(train_idx)]
    for emotion in EMOTIONS:
        label_col = 'emotion_' + emotion
        labels = tweet_data[label_col]
        X_train, X_test, Y_train, Y_test , train_idx,test_idx = train_test_split(features, labels,indices, test_size=0.2 , random_state=5)
        model.fit(X_train, Y_train)
        predicted_emotions = model.predict(X_test)
        test_tweets[label_col + '_prediction'] = 0
        i = 0
        for index in test_idx:
            test_tweets.loc[index ,label_col + '_prediction' ] = predicted_emotions[i]
            i+= 1
    # output misclassified tweets to json file
    translator = Translator()
    misclassifed = {}
    for emotion in EMOTIONS:
        label_col = 'emotion_' + emotion
        misclassifed[emotion+'_FP'] = []
        misclassifed[emotion+'_FN'] = []
        for i , row in test_tweets.iterrows():
            # false positive
            if row[label_col + '_prediction'] == 1 and row[label_col] == 0:
                #get the humuan labels for this tweet
                labels = []
                for col in ['emotion_' + i  for i in EMOTIONS ]:
                    if row[col] == 1:
                        labels.append(col)
                if row['emotion_others'] == 1:
                    labels.append('other')
                #google translate tweet
                detected_lang = translator.detect(row['tweet']).lang
                english_tweet = translator.translate(row['tweet'], dest='en' , src=detected_lang).text
                misclassifed[emotion+'_FP'] += [[labels , english_tweet]]
            # false negative
            if row[label_col + '_prediction'] == 0 and row[label_col] == 1:
                #get the incorrect predictions for this tweet
                labels = []
                for col in ['emotion_' + i + '_prediction' for i in EMOTIONS]:
                    if row[col] == 1:
                        labels.append(col)
                if row['emotion_others'] == 1:
                    labels.append('other')
                #google translate tweet 
                detected_lang = translator.detect(row['tweet']).lang
                english_tweet = translator.translate(row['tweet'], dest='en' , src=detected_lang).text
                misclassifed[emotion+'_FN'] += [[labels , english_tweet]]
    with open("misclassifed.json", "w") as outfile:
        json.dump(misclassifed, outfile , ensure_ascii=False,indent=4)  
        
                

def main():
    parser = argparse.ArgumentParser(description="Emotion Detection ML Models")
    parser.add_argument('--emojis', help="True to encode emojis, False to not encode emojis", default=False)
    parser.add_argument("--model", default='all', help="lr: logistic regression, dt: decision tree, knn: k-nearest neighbors, nb: naive bayes, rf: random forest, svm: svm")
    parser.add_argument("--encode", help="Encoding categories as binary or not", default=True)
    parser.add_argument("--combine", help="True to combine anger + disgust, False not combine", default=False)
    args = parser.parse_args()

    encode = True if (args.encode == 'True' or args.encode is True) else False
    emojis = True if args.emojis == 'True' else False
    combine = True if args.combine == 'True' else False


    data_file = '../data/ukrainian_emotion_big.tsv'
    if encode:
        data, features = encode_and_vectorize_binary(data_file, encode_emojis=emojis, combine=combine)
    else: 
        data, features, labels = encode_and_vectorize_multi_class(data_file, encode_emojis=emojis, combine=combine)

    log_reg = dec_tree = k = nb = rand_for = s = False

    #determine which model
    if args.model == 'all':
        log_reg = dec_tree = k = nb = rand_for = s = True
    elif args.model == 'lr':
        log_reg = True
    elif args.model == 'dt':
        dec_tree = True
    elif args.model == 'knn':
        k = True
    elif args.model == 'nb':
        nb = True
    elif args.model == 'rf':
        rand_for = True
    elif args.model == 'svm':
        s = True
    else:
        print("ERROR: no model selected")

    if log_reg:
        print('Logistic Regression')
        if encode:
            logreg_binary(data, features, combined=combine)
        else:
            logreg_multiclass(data, features)
        print('\n')

    if dec_tree:
        print('Decision Tree')
        if encode:
            dectree_binary(data, features, combined=combine)
        else:
            dectree_multiclass(data, features)
        print('\n')

    if k:
        print('KNN')
        if encode:
            knn_binary(data, features, combined=combine)
        else:
            knn_multiclass(data, features)
        print('\n')

    if nb:
        print('Naive Bayes')
        if encode:
            bayes_binary(data , features, combined=combine)
        else:
            bayes_multiclass(data , features)
        print('\n')

    if rand_for:
        print('Random Forest')
        if encode:
            random_forest_binary(data, features, combined=combine)
        else:
            random_forest_multiclass(data, features)
        print('\n')

    if s:
        print('SVM')
        if encode:
            svm_binary(data, features, combined=combine)
        else:
            svm_multiclass(data, features)
        print('\n')

if __name__ == "__main__":
    main()

