"""
Authors: Lina Laghzaoui, Grace Magny-Fokam, Toby Otto, Rich Pihlstrom

Description:
    Program runs a dictionary-based emotion classifier over a dataset of tweets.
    The inputed data is vectorized in a 'bag-of-words' model. Notably emojis are translated into text using the emoji to Arabic dictionary(lexicon/emojis.csv).
    The data is outputted to outfile.csv
    
Necessary Packages: 
    You will need to install pytrends in order to use the API in this code:
        pip3 install nltk

threshold: the confidence threshold you want an emotion score to meet to be counted as present.

filename: the path to the original datafile that you are using.

USAGE:
    python3 lexicon.py -t <[0,1]> -f <filename> 

"""
import pandas as pd
import numpy as np
import argparse
import re
import random
import os
from NRCLex import NRCLex
from nltk.tokenize import RegexpTokenizer

#Global list of emotions, should match NRCLex.py file
EMOTIONS = ['anger', 'fear', 'sadness', 'disgust', 'joy', 'anger-disgust']

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

def tokenize(tweets):
    regexp = RegexpTokenizer('\w+')
    tweets = tweets.map(str)
    token_col = tweets.apply(regexp.tokenize)
    return token_col

'''#combine anger and disgust + remove duplicates
def combineDisgustAnger(data):
    for label in data['emotions']:
        if label==['disgust']:
            label='anger'
        data=list(set(data))
    return data'''

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

#Function to classify tweets using NRCLex
def emotion_classification(data, threshold, top_emo_flag):
    
    retList = []

    coverageDict = {}
    for emotion in EMOTIONS:
        coverageDict[emotion]={
            "total":0,
            "covered":0
        }

    for tweet in data:
        #Use library call to get emotion classification
        text_object = NRCLex(tweet)
        topEmo = text_object.top_emotions

        anger_disgust_score = 0
        for tup in topEmo:
            if tup[0] == "anger" or tup[0] == "disgust":
                anger_disgust_score += tup[1]
        topEmo.append(("anger-disgust",anger_disgust_score))

        d={
            "tweet":tweet,
            "threshold":threshold,
            "top emotion flag":top_emo_flag
        }
    
        #Check if emotion value is greater than threshold
        #and add classification to return-dictionary 
        minEmo = threshold
        if top_emo_flag:
            vals = []
            for tup in topEmo:
                vals.append(tup[1])
            maxVal = max(vals)
            minEmo = max(threshold, maxVal)

        for tup in topEmo:
            if tup[0] in EMOTIONS and tup[1] >= minEmo:
                d[tup[0]]=tup[1]
            
            #Increment coverage values for threshold coverage metric
            if tup[0] in EMOTIONS:
                coverageDict[tup[0]]["total"] +=1
                if tup[1] >= threshold:
                    coverageDict[tup[0]]["covered"] += 1
    
        retList.append(d)
    
    #Calculate coverage for each emotion
    for emotion in coverageDict:
        try:
            coverageDict[emotion]["coverage"] = coverageDict[emotion]["covered"]/coverageDict[emotion]["total"]
        except ZeroDivisionError:
            coverageDict[emotion]["coverage"] = 0
    
    #Append the coverage dictionary to the return list. We just pop it later.
    retList.append(coverageDict)

    return retList

def eval(tweet_df, model_labels, emoji_flag):
    retDict={}

    #Pop coverage dictionary off the end of the model_labels.
    coverageDict = model_labels[-1]
    model_labels = model_labels[:-1]

    for emotion in EMOTIONS:

        retDict[emotion] = {
            "emoji flag":emoji_flag,
            "threshold": model_labels[0]["threshold"],
            "coverage": coverageDict[emotion]["coverage"],
            "top emotion flag": model_labels[0]["top emotion flag"],
            "Baseline Prevalence":0,
            "0-baseline TN":0,
            "0-baseline FN":0,
            "Random baseline TP":0,
            "Random baseline FP":0,
            "Random baseline TN":0,
            "Random baseline FN":0,
            "TP":0,
            "FP":0,
            "TN":0,
            "FN":0
        }

        #Manually count TP, FP, TN, FN and baseline count to assess accuracy.
        for i in range(0, len(model_labels)):
            if tweet_df[emotion].iloc[i] == 1:
                retDict[emotion]["Baseline Prevalence"] += 1/len(model_labels)
                retDict[emotion]["0-baseline FN"] += 1
                retDict[emotion]["Random baseline "+random.choice(["TP","FN"])]+=1
                
                if emotion in model_labels[i] and model_labels[i][emotion] > 0:
                    retDict[emotion]["TP"] += 1
                else:
                    retDict[emotion]["FN"] += 1

            else:
                retDict[emotion]["0-baseline TN"] += 1
                retDict[emotion]["Random baseline "+random.choice(["TN","FP"])]+=1
                if emotion not in model_labels[i] or model_labels[i][emotion] == 0:
                    retDict[emotion]["TN"] += 1
                else:
                    retDict[emotion]["FP"] += 1
        
        #Calculate evaluation metrics.
        retDict[emotion]["0-baseline Accuracy"] = 100*(retDict[emotion]["0-baseline TN"])/\
            (retDict[emotion]["0-baseline TN"]+retDict[emotion]["0-baseline FN"])
        retDict[emotion]["Random baseline Accuracy"] = 100*(retDict[emotion]["Random baseline TN"]+retDict[emotion]["Random baseline TP"])/\
            (retDict[emotion]["Random baseline TN"]+retDict[emotion]["Random baseline TP"]+\
                retDict[emotion]["Random baseline FN"]+retDict[emotion]["Random baseline FP"])
        retDict[emotion]["Accuracy"]=100*(retDict[emotion]["TP"]+retDict[emotion]["TN"])/\
            (retDict[emotion]["TP"]+retDict[emotion]["TN"]+retDict[emotion]["FP"]+retDict[emotion]["FN"])
        
        retDict[emotion]["Precision"]=100*retDict[emotion]["TP"]/\
            (retDict[emotion]["TP"]+retDict[emotion]["FP"])
        
        retDict[emotion]["Recall"]=100*retDict[emotion]["TP"]/\
            (retDict[emotion]["TP"]+retDict[emotion]["FN"])
        
        retDict[emotion]["F1"]=(2*retDict[emotion]["Precision"]*retDict[emotion]["Recall"])/\
            (retDict[emotion]["Precision"]+retDict[emotion]["Recall"])

        keys = list(retDict[emotion].keys())
        for key in keys:
            if any(word in key for word in ["TP","FP","TN","FN"]):
                retDict[emotion].pop(key)
        
    return pd.DataFrame.from_dict(retDict, orient='index')
            
def main():
    #Add argument option for min-threshold classification
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", help="threshold for emotion score", type=float, nargs="?", default=0.0)
    parser.add_argument("-f", "--filename", nargs=1, type=str, default="data/arabic_emotion_new_new.csv",
                        help="The path to your data file")
    args = parser.parse_args()
    
    threshold_vals = np.linspace(0, 1, num=11)
    
    file_name = args.filename[0]
    outfile="outfile.csv"
    
    df_out = pd.DataFrame()

    grid = [(emoji_flag, top_emo_flag, threshold) \
            for emoji_flag in [0,1] for top_emo_flag in [0,1] \
                for threshold in np.linspace(0, 1, num=10)]
    
    for emoji_flag, top_emo_flag, threshold in grid:
        
        print("""
        - emoji_flag: {},
        - top_emo_flag: {},
        - threshold: {}
        """.format(emoji_flag, top_emo_flag, threshold))

        tweet_df = preprocess(file_name, emoji_flag=emoji_flag)

        #replace/append output df with df of each emoji_flag value
        if df_out.empty:
            df_out = eval(tweet_df, emotion_classification(tweet_df["processed_tweets"], \
                                                           threshold, top_emo_flag=top_emo_flag), emoji_flag)
        else:
            df_out = pd.concat([df_out, eval(tweet_df, \
                                             emotion_classification(tweet_df["processed_tweets"], \
                                                                    threshold, top_emo_flag=top_emo_flag), emoji_flag)])
        
    df_out.to_csv(outfile)

if __name__ == '__main__':
    main()