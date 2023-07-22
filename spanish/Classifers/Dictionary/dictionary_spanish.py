"""
Authors: Bernardo Medeiros, Colin Hwang, Mattea Whitlow, Adeline Roza

Program Description:

This program is designed to analyze tweets for emotions based on the NRCLex Spanish lexicon. The tweets are read from a tsv
file that contains columns for tweets and their emotion labels. The program provides three methods for analyzing emotions in 
tweets:

EmoScoreVar - Considers top emotions for a tweet. If multiple emotions tie, they are all used to label the tweet.
EmoScoreThreshold - Considers all emotions for a tweet that meet a certain threshold. If no emotion meets the threshold, 'n/a' 
is recorded.
getNRCEmotions - Considers all emotions associated with a tweet.

Each method calculates accuracy, precision, recall, F1 scores for each emotion, and the overall prevalence of each emotion in 
the dataset. The methods are then compared, and the method providing the highest accuracy and F1 score is reported.

How to Run:

python3 emotion_detection.py --input_file spanish_emotion.tsv --output_dir [output directory]

"""

from NRCLex_spanish import NRCLex
import numpy as np
import pandas as pd
import json
import csv
from nltk.tokenize import TweetTokenizer
import emoji
from sklearn.metrics import accuracy_score
import random
import argparse
import os

# defining set of emotions
emotions = ['anger', 'fear', 'sadness','joy']

# get NRCLex data for top emotions (multiple emotions will be recorded if there is a tie between top emotions)
def EmoScoreVar(data, min_prevelance = 0.15):
    emo_stats = {}
    for emo in emotions: 
          emo_stats[emo] = {'tp': 0, 'fp': 0, 'fn': 0, 'acc_count': 0, 'count': 0, 'accuracy': 0, 'prevalence': 0, 'precision': 0, 'recall': 0, 'f1_score': 0}
    top_emo_score_list = {}
    for i in range(len(data)):
        row = data.iloc[i]
        label = row['emotion']
        tweet = row['tweet']
        tweet = removeEmojis(tweet)
        top_emos = NRCLex(str(tweet)).top_emotions
        top_emos = [item[0] for item in top_emos if item[1] != 0]
        top_emo_score_list[tweet] = [top_emos, label]
        for emo in emotions:
            if emo in top_emos and emo in str(label).strip():
                emo_stats[emo]['tp'] +=1
                emo_stats[emo]['acc_count'] +=1
            if emo in top_emos and emo not in str(label).strip():
                emo_stats[emo]['fp'] +=1
            if emo not in top_emos and emo in str(label).strip():
                emo_stats[emo]['fn'] +=1
            if emo in str(label).strip():
                emo_stats[emo]['count'] +=1
            else:
                if emo not in top_emos and emo not in str(label).strip():
                    emo_stats[emo]['acc_count'] +=1
    data_len = len(data)
    for emo in emotions: 
        accuracy, prevalence, precision, recall, f1 = getStats(emo, emo_stats, data_len)
        emo_stats[emo]['accuracy'] = accuracy
        emo_stats[emo]['prevalence'] = prevalence
        emo_stats[emo]['precision'] = precision
        emo_stats[emo]['recall'] = recall
        emo_stats[emo]['f1_score'] = f1


    return top_emo_score_list, emo_stats
    
# get NRXLex data for all top emotions meeting min_prevalence threshold
def EmoScoreThreshold(data, min_prevelance = 0.26):
    emo_stats = {}
    for emo in emotions: 
        emo_stats[emo] = {'tp': 0, 'fp': 0, 'fn': 0, 'acc_count': 0, 'count': 0, 'accuracy': 0, 'prevalence': 0, 'precision': 0, 'recall': 0, 'f1_score': 0}
    top_emo_score_threshold = {}
    data_len = 0
    for i in range(len(data)):
        row = data.iloc[i]
        label = row['emotion']
        tweet = row['tweet']
        tweet = removeEmojis(tweet)
        top_emos = NRCLex(str(tweet)).top_emotions
        # emo_threshold = [item[0] for item in top_emos if item[1] >= 0.51]
        top_emos = [(emotion, float(score)) if float(score) >= min_prevelance else ('n/a', 0.0) for emotion, score in top_emos]
        top_emo_score_threshold[tweet] = [top_emos, label]
        for emo in emotions:
            data_len += 1
            for item in top_emos:
                if 'n/a' not in item:
                    if emo in item and emo in str(label).strip():
                        emo_stats[emo]['tp'] +=1
                        emo_stats[emo]['acc_count'] +=1
                    if emo in item and emo not in str(label).strip():
                        emo_stats[emo]['fp'] +=1
                    if emo not in item and emo in str(label).strip():
                        emo_stats[emo]['fn'] +=1
                    if emo in str(label).strip():
                        emo_stats[emo]['count'] +=1
                    else:
                        if emo not in item and emo not in str(label).strip():
                            emo_stats[emo]['acc_count'] +=1
    for emo in emotions: 
        accuracy, prevalence, precision, recall, f1 = getStats(emo, emo_stats, data_len)
        emo_stats[emo]['accuracy'] = accuracy
        emo_stats[emo]['prevalence'] = prevalence
        emo_stats[emo]['precision'] = precision
        emo_stats[emo]['recall'] = recall
        emo_stats[emo]['f1_score'] = f1
    return top_emo_score_threshold, emo_stats
    
# removes emojis from single tweet
def removeEmojis(tweet):
    tweet = emoji.demojize(str(tweet), language='es')
    tweet = tweet.replace(":", " ")
    tweet = tweet.replace("_", " ")
    return tweet

# get NRCLex data (uses affect_list, which gives all emotions associated with a tweet, even those with lower scores than others)
def getNRCEmotions(data):
    emo_stats = {}
    for emo in emotions:
        emo_stats[emo] = {'tp': 0, 'fp': 0, 'fn': 0, 'acc_count': 0, 'count': 0, 'accuracy': 0, 'prevalence': 0, 'precision': 0, 'recall': 0, 'f1_score': 0}
    emo_score_list = {}
    for i in range(len(data)):
        # get the tweet and its label
        row = data.iloc[i]
        label = row['emotion']
        tweet = row['tweet']
        # translate emoji to text
        tweet = removeEmojis(tweet)
        # get emotions associated with tweets
        emotion = NRCLex(str(tweet)).affect_list
        # emo_score = emotion.raw_emotion_scores
        emo_score_list[tweet] = [emotion, label]
        for emo in emotions:
            # count tp, fn, fp for f1, accuracy, and recall scores
            if emo in str(label).strip():
                if emo in emotion:
                    emo_stats[emo]['tp'] += 1
                if emo not in emotion:
                    emo_stats[emo]['fn'] += 1
            if emo in emotion and emo not in str(label).strip():
                emo_stats[emo]['fp'] += 1
            # count emotion prevalence in label
            if emo in str(label).strip():
                emo_stats[emo]['count'] += 1
            # emotion count for acc calculations
            if emo in emotion and emo in str(label).strip():
                emo_stats[emo]['acc_count'] += 1
            else:
                if emo not in emotion and emo not in str(label).strip():
                    emo_stats[emo]['acc_count'] += 1
    data_len = len(data)
    for emo in emotions:
        accuracy, prevalence, precision, recall, f1 = getStats(emo, emo_stats, data_len)
        emo_stats[emo]['accuracy'] = accuracy
        emo_stats[emo]['prevalence'] = prevalence
        emo_stats[emo]['precision'] = precision
        emo_stats[emo]['recall'] = recall
        emo_stats[emo]['f1_score'] = f1
    
    return emo_score_list, emo_stats

# generates statistics for NRCLex data
def getStats(emo, emo_stats, data_len):
    accuracy = emo_stats[emo]['acc_count'] / data_len
    prevalence = emo_stats[emo]['count'] /  data_len
    try:
        precision = emo_stats[emo]['tp'] / (emo_stats[emo]['tp'] + emo_stats[emo]['fp'])
    except ZeroDivisionError:
        precision = float('NaN')
    try:
        recall = emo_stats[emo]['tp'] / (emo_stats[emo]['tp'] + emo_stats[emo]['fn'])
    except ZeroDivisionError:
        recall = float('NaN')
    try:
        f1 = 2 * ((precision * recall) /(precision + recall))
    except ZeroDivisionError:
        f1 = float('NaN') 
    return accuracy, prevalence, precision, recall, f1

# convert text file from corpus to json
def convertToJson(output_dir):
    data = {}
    with open('Spanish-NRC-EmoLex.txt', 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader)  
        for row in reader:
            word = row[-1] 
            emotions_json = []
            for i in range(1, len(header) - 2): 
                value = int(row[i])
                if value == 1:  
                    emotions_json.append(header[i])
            data[word] = emotions_json

    with open(os.path.join(output_dir, 'nrc_spanish.json'), 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

# combine disgust and anger into one emotion
def removeDisgust(data):
    i = 0
    for emotion_label in data["emotion"]:
        if emotion_label == 'disgust':
            emotion_label = 'anger'
        data["emotion"][i] = emotion_label
        i += 1
    return data   

# sorts labels into numerical categories
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
        if label != 'others':
            label_list.append(label)
    return label_list

# calculate baseline accuracy and output it to terminal
def randBaseline(data):
    gt_list = []
    est_list = []
    for label in data['emotion']:
        if label != 'others':
            gt_list.append(random.randint(0, 5))
    est_list = sortLabels(data)
    # generate and print accuracy stats
    accuracy = accuracy_score(est_list, gt_list)
    print("Baseline accuracy: " + str(accuracy))
    print('--')

# compare all the methods and output the best accuracy score + method
def compareAccuracy(data, output_dir):
    highest_accuracy = 0.0
    best_emotion = None
    best_model = None
    with open(os.path.join(output_dir, "best_accuracy_dictionary.txt"), "w") as file:
        top_emo_score_var, emo_stats_var = EmoScoreVar(data)
        var_accuracy_scores = [emo_stats_var[emo]['accuracy'] for emo in emotions]
        if max(var_accuracy_scores) > highest_accuracy:
            highest_accuracy = max(var_accuracy_scores)
            best_emotion = emotions[var_accuracy_scores.index(highest_accuracy)]
            best_method = "EmoScoreVar"
        top_emo_score_threshold, emo_stats_threshold = EmoScoreThreshold(data)
        threshold_accuracy_scores = [emo_stats_threshold[emo]['accuracy'] for emo in emotions]
        if max(threshold_accuracy_scores) > highest_accuracy:
            highest_accuracy = max(threshold_accuracy_scores)
            best_emotion = emotions[threshold_accuracy_scores.index(highest_accuracy)]
            best_method = "EmoScoreThreshold"
        emo_score_list, emo_stats_nrc = getNRCEmotions(data)
        nrc_accuracy_scores = [emo_stats_nrc[emo]['accuracy'] for emo in emotions]
        if max(nrc_accuracy_scores) > highest_accuracy:
            highest_accuracy = max(nrc_accuracy_scores)
            best_emotion = emotions[nrc_accuracy_scores.index(highest_accuracy)]
            best_method = "getNRCEmotions"
        file.write("Highest Accuracy: " + str(highest_accuracy) + "\n")
        file.write("Best Emotion: " + best_emotion + "\n")
        file.write("Best Method: " + best_method + "\n")

# compare all the methods and output the best f1 score + method
def compareF1(data, output_dir):
    highest_f1 = 0.0
    best_emotion = None
    best_model = None
    with open(os.path.join(output_dir, "best_f1_dictionary.txt"), "w") as file:
        top_emo_score_var, emo_stats_var = EmoScoreVar(data)
        var_f1_scores = [emo_stats_var[emo]['f1_score'] for emo in emotions]
        if max(var_f1_scores) > highest_f1:
            highest_f1 = max(var_f1_scores)
            best_emotion = emotions[var_f1_scores.index(highest_f1)]
            best_method = "EmoVarScore"
        top_emo_score_threshold, emo_stats_threshold = EmoScoreThreshold(data)
        threshold_f1_scores = [emo_stats_threshold[emo]['f1_score'] for emo in emotions]
        if max(threshold_f1_scores) > highest_f1:
            highest_f1 = max(threshold_f1_scores)
            best_emotion = emotions[threshold_f1_scores.index(highest_f1)]
            best_method = "EmoVarThreshold"
        emo_score_list, emo_stats_nrc = getNRCEmotions(data)
        nrc_f1_scores = [emo_stats_nrc[emo]['f1_score'] for emo in emotions]
        if max(nrc_f1_scores) > highest_f1:
            highest_f1 = max(nrc_f1_scores)
            best_emotion = emotions[nrc_f1_scores.index(highest_f1)]
            best_method = "getNRCEmotions"
        file.write("Highest F1: " + str(highest_f1) + "\n")
        file.write("Best Emotion: " + best_emotion + "\n")
        file.write("Best Method: " + best_method + "\n")

# function to test different thresholds to use
def thresholdtesting(data):
    x_values = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]  
    results = []
    # loop through all the threshold values
    for x in x_values:
        emo_stats_tests, emo_stat_thresh_test = EmoScoreThreshold(data, x)
        results.append(emo_stat_thresh_test)
    return results

def main():
    parser = argparse.ArgumentParser(description="Arguments for spanish dictionary model")
    # arguments for intput file and output directory
    parser.add_argument('--output_dir', type=str, required=True, help='Specify an output directory.')
    parser.add_argument('--input_file', type=str, required=True, help='Specify a (spanish) dataset with tweets and emotions (spanish_emotions.tsv) Lexicon')
    args = parser.parse_args()
    # store output directory
    output_dir = args.output_dir
    # if the directory doesn't exist, make it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # read in data
    data = pd.read_csv(args.input_file, sep='\t', on_bad_lines='skip')
    # remove surprise, others and disgust from the data
    data = data[~data['emotion'].isin(['surprise', 'others', 'disgust'])]
    # get emotion stats from using top emo criteria
    NRC_emotions_list, emo_stats = getNRCEmotions(data)
    # output results to emo_score_list.json
    with open(os.path.join(output_dir, "emo_score_list.json"), "w", encoding="utf-8") as file:
        json.dump(NRC_emotions_list, file, indent=4, ensure_ascii=False)
    with open(os.path.join(output_dir, "emo_score_stats_affect_list.json"), "w", encoding="utf-8") as file:
        json.dump(emo_stats, file, indent=4, ensure_ascii=False)
    # use top emo score criteria
    top_emo_score, emo_stats = EmoScoreVar(data)
    with open(os.path.join(output_dir, "top_emo_score_list.json"), "w", encoding="utf-8") as file:
        json.dump(top_emo_score, file, indent=4, ensure_ascii=False)
    with open(os.path.join(output_dir, "emo_score_stats_top.json"), "w", encoding = "utf-8") as file:
        json.dump(emo_stats, file, indent=4, ensure_ascii=False)
    # use threshold criteria
    top_emo_threshold_list, emo_stats = EmoScoreThreshold(data)
    with open(os.path.join(output_dir, "top_emo_threshold.json"), "w", encoding = "utf-8") as file:
        json.dump(top_emo_threshold_list, file, indent=4, ensure_ascii=False)
    with open(os.path.join(output_dir, "emo_scores_stats_threshold.json"), "w", encoding = "utf-8") as file:
        json.dump(emo_stats, file, indent=4, ensure_ascii=False)
    # try different thresholds using threshold testing
    results = thresholdtesting(data)
    with open(os.path.join(output_dir, 'resultsTest.json'), 'w') as file:
        json.dump(results, file, indent=4, ensure_ascii=False)
    # get best accuracy and f1 from all three criteria
    compareAccuracy(data, output_dir)
    compareF1(data, output_dir)

if __name__ == "__main__":
    main()

