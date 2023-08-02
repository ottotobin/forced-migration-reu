import pandas as pd
import re
import json
import os
import datetime
import csv
import category_encoders as ce
from sklearn.model_selection  import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer

#different groupings of emotion categories
EMOTIONS = ['anger', 'fear', 'sadness', 'disgust', 'joy']
EMOTIONS2 = ['anger', 'fear', 'sadness', 'joy']
EMOTIONS_OTHERS = ['anger', 'fear', 'sadness', 'joy', 'others']

def preprocess(data_file , encode_emojis=False):
    """Preprocesses input data to remove emojis (if specified), unwanted symbols, links, @s, and hashtags

        Takes in a data file to be cleaned, and a boolean to indicate whether emojis should be encoded or removed
    """

    df = pd.read_csv(data_file, sep='\t', encoding = 'utf8', )
    print('preprocess size: ', df.shape[0])
    print(df.emotion.value_counts())
    #First, remove "surprise" and "others" from data
    drop_rows = df.loc[~df['emotion'].isin(EMOTIONS_OTHERS)]
    df.drop(drop_rows.index, inplace=True)

    #deal with 'tweet' column name and 'text' in different files
    if 'tweet' in df.columns:
        text_col = 'tweet'
    else:
        text_col = 'text'

    processed_tweets = []
    
    if encode_emojis:
        #replace emojis with text
        emoji_dict = {}
        with open('../data/emoji_uk.json') as json_file:
            emoji_dict = json.load(json_file)
        
        emoji_processing = []
        # replace emojis with words
        for tweet in df[text_col]:
            split_tweet = tweet.split()
            split_word_list = []
            for i in split_tweet:
                encoded_word = ''
                for char in i:
                    if char in emoji_dict.keys():
                        encoded_word = encoded_word + emoji_dict[char]
                    else:
                        encoded_word = encoded_word + char
                split_word_list.append(encoded_word)
            emoji_processing.append(" ".join(split_word_list))
        df[text_col] = emoji_processing

    #deal with remaining emojis and symbols
    emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            "]+", flags=re.UNICODE)
    processed_tweets = []
    for tweet in df[text_col]:
        text = ' '.join(word for word in tweet.split() if not (word.startswith('http') or word.startswith('@') or word.startswith('#')))
        text = emoji_pattern.sub(r'', text)
        text = text.lower()
        processed_tweets.append(text)

    df['processed_tweets'] = processed_tweets
    print('postprocess size: ', df.shape[0])
    print(df['emotion'].value_counts())
    return df

def preprocess_text(text):
    """preprocessing function, but for individual text"""
    emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "]+", flags=re.UNICODE)
    text = ' '.join(word for word in text.split() if not (word.startswith('http') or word.startswith('@') or word.startswith('#')))
    text = emoji_pattern.sub(r'', text)
    text = text.lower()
    return text

def combine_emotions(data):
    """Input: already processed tweets, combines similar emotions disgust and anger"""
    data['emotion'] = [item if item != 'disgust' else 'anger' for item in data['emotion']]
    return data

def combine_and_delete_dups(arr):
    """combines disgust and anger and deletes duplicates after data has been combined"""
    replaced = [list(map(lambda x: x if x != 'disgust' else 'anger', i)) for i in arr]
    final = [list(set(item)) for item in replaced]
    return final

def encode_and_vectorize_binary(data_file, encode_emojis=False, combine=False):
    '''
    input: data_file: tsv file with tweet data
           encode_emojis: boolean to indicate whether to encode emojis
           combine: boolean to indicate whether to combine anger and disgust
    output:
        encoded: pd.DataFrame with tweet data and binary encoded classifications for each emotion
        X: sparse matrix of features with a row for each tweets
    '''
    processed_df = preprocess(data_file,encode_emojis)
    if combine:
        processed_df = combine_emotions(processed_df)
    #one hot encoding for emotions
    encoder = ce.OneHotEncoder(cols = ['emotion'], return_df=True, use_cat_names=True)
    encoded = encoder.fit_transform(processed_df)
    #get features
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(encoded['processed_tweets'])
    return encoded.reset_index(), X

def encode_and_vectorize_multi_class(data_file, encode_emojis=False, combine=False):
    '''
    same input/output as encode_and_vectorize_binary but for multi-class classification
    '''
    processed_df = preprocess(data_file, encode_emojis)
    if combine:
        processed_df = combine_emotions(processed_df)
    labelencoder = LabelEncoder()
    processed_df['encoded_emotion'] = labelencoder.fit_transform(processed_df['emotion'])
    #get dictionary of labels
    mapping = {l: i for i, l in enumerate(labelencoder.classes_)}
    #get features
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(processed_df['processed_tweets'])
    return processed_df, X, mapping

def load_combined_data():
    '''
    Returns: pd.Dataframe of test set and tweets labeled 'other'
    '''
    #get dataframe of test labels
    file_name = '../data/ukrainian_emotion_new_new.tsv'
    data, X ,labels = encode_and_vectorize_multi_class(file_name, combine=True)
    _, _, y_train, y_test = train_test_split(data['processed_tweets'], data['encoded_emotion'], test_size=0.2, stratify=data['encoded_emotion'], random_state=42)
    test_df = pd.DataFrame({'text': y_train, 'emotion': y_test})

    #get dataframe of tweets labeled 'others)
    others_df = pd.read_csv('../data/ukrainian_emotion_new.tsv', sep='\t', encoding = 'utf8')
    drop_rows = others_df.loc[~others_df['emotion'].isin(['others'])]
    others_df.drop(drop_rows.index, inplace=True)
    others_df.drop(['id', 'event' , 'offensive'], axis=1 , inplace=True)
    others_df.rename(columns={'tweet' : 'text'}, inplace=True)

    final = pd.concat([others_df, test_df])
    final.to_csv('../data/combined_data.csv')

def combine_cloudshare_data(directory):
    """
    Gets confusingly sotred Google Cloud share twitter scrapes and creates one cohesive csv file with all tweets
    """
    outfile = "unlabeled_tweets_2022.csv"
    # # if already created csv
    if os.path.exists(outfile):
       return outfile

    with open(outfile, "w+") as f:
        writer = csv.writer(f)
        header = ["date", "city", "raw_tweet"]
        writer.writerow(header)
        
        rows=[]
        # nestings of for loops to iterate over all of the 
        # json files stored in the larger cloud shre directory
        for filename in os.listdir(directory):
                #path to month date folder
                first_path = os.path.join(directory, filename)
                for datefile in os.listdir(first_path):
                        if '2022' in datefile:
                            #path to one day folder
                            second_path = os.path.join(first_path, datefile)
                            date = datetime.datetime.strptime(datefile.split('date=')[1], "%Y-%m-%d").strftime("%Y-%m-%d") 
                            for city in os.listdir(second_path):
                                    city_name = city.split('city=')[1]
                                    # data dir
                                    third_path = os.path.join(second_path, city)
                                    for json_file in os.listdir(third_path):
                                            # json file
                                            json_path = os.path.join(third_path, json_file)
                                            jsonObj = pd.read_json(json_path, lines=True)
                                            print(jsonObj)
                                            text = jsonObj['preprocessed_text'].values[0]
                                            row = [date, city_name, text.replace("\n","").replace("\r","")]
                                            rows.append(row)
                                            writer.writerow(row)
                    
    return outfile




