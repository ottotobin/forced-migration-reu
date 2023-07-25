import sys
import csv
sys.path.append('../')
from sklearn.model_selection import train_test_split
from helper_funcs import *
import pandas as pd
from transformers import BertForSequenceClassification, BertTokenizer
import torch
emotion_to_class = {'anger': 0, 'fear': 1, 'joy': 2, 'sadness': 3 , 'others':4}
class_to_emotion = {0: 'anger', 1: 'fear', 2: 'joy', 3: 'sadness' , 4:'others'}	

def load_data():
    '''
    Returns: pd.Dataframe of test set and tweets labeled 'other'
    '''
    #get dataframe of test labels
    file_name = '../data/ukrainian_emotion_new_new.tsv'

    epochs = 10
    data, X ,labels = encode_and_vectorize_multi_class(file_name, combine=True)
    _, text_emotions, _, label_emotions = train_test_split(data['processed_tweets'], data['encoded_emotion'], test_size=0.2, stratify=data['encoded_emotion'], random_state=42)


    #get dataframe of tweets labeled 'others)
    others_df = pd.read_csv('../data/ukrainian_emotion_new.tsv', sep='\t', encoding = 'utf8')
    drop_rows = others_df.loc[~others_df['emotion'].isin(['others'])]
    others_df.drop(drop_rows.index, inplace=True)
    others_df['emotion'] = [emotion_to_class[ row['emotion']] for i,row in others_df.iterrows()]
    others_df.drop(['id', 'event' , 'offensive'], axis=1 , inplace=True)
    
    #print('otehrs length before: ' , len(others_df.index))
    #print(others_df)
    #combine test set and 'others' tweets
    text_emotions, label_emotions =  list(text_emotions), list(label_emotions)
    #print(len(text_emotions) , len(label_emotions))
    for i in range(len(text_emotions)):
        #print([text_emotions[i] , label_emotions[i]])
        others_df.loc[len(others_df.index) + 1500 + i] = [text_emotions[i] , label_emotions[i]]
    #clean up some tweets
    others_df['tweet'] = others_df.apply(lambda row : preprocess_text(row['tweet']), axis = 1)
    #print('otehrs length after: ' , len(others_df.index))
    return others_df
def find_threshold(data):
    '''
    parameters: 
        data: dataframe of tweets with emotions and others as labels
    this function predicts an output for each tweet and then prints the accuracy% for each emotion and threshold
    '''

    #load in bert stuff
    model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-uncased', num_labels=4)
    model.load_state_dict(torch.load('bert_good_model.pt', map_location=torch.device('cpu')))
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased', return_token_type_ids=False)

    #get counts for each emotion in the test set
    tweet_type_counter = data['emotion'].value_counts()
    num_anger = tweet_type_counter[0]
    num_sadness = tweet_type_counter[3]
    num_fear = tweet_type_counter[1]
    num_joy = tweet_type_counter[2]
    num_others = tweet_type_counter[4]
    model.eval()

     #loop thru n thresholds
    for threshold in [.6 , .65 , .7 ,.75, .8 , .9 , .93 , .97]:
        mislabel_by_threshold= 0 # number of times the tweet emotion was predict right but the threshold made it be others
        caught_others = 0
        others_not_labeled_joy = 0
        joy_labels = 0
        correct_others, correct_joy, correct_fear , correct_anger , correct_sadness = 0,0,0,0,0
        
        #label each tweet and check if it is correct
        for i , row in data.iterrows():
            #find output/prediction
            tweet = row['tweet']
            true_label = class_to_emotion[row['emotion']]
            input = tokenizer(tweet, padding=True, max_length = 512,truncation=True,return_tensors="pt")
            mask = input['attention_mask']
            inputs = input['input_ids']
            with torch.no_grad():
                outputs = model(input_ids = inputs, attention_mask=mask)
            logits = outputs.logits
            probabilitys = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilitys, dim=1).item()
            predicted_emotion = class_to_emotion[predicted_class]
            
            # apply the threshold to convert joy into others
            if max(probabilitys[0]).item() < threshold and predicted_emotion == 'joy':
                predicted_emotion = 'others'
            #check if each emotion is correctly predicted, add it to a tally if so
            if true_label == predicted_emotion == 'joy':
                correct_joy += 1
            if predicted_emotion == 'joy':
                joy_labels+= 1
            if true_label == predicted_emotion == 'anger':
                correct_anger += 1
            if true_label == predicted_emotion == 'fear':
                correct_fear += 1
            if true_label == predicted_emotion == 'sadness':
                correct_sadness += 1
            if true_label == predicted_emotion == 'others':
                correct_others +=1
            if true_label == 'others' and predicted_emotion != 'joy':
                others_not_labeled_joy += 1

        results = {
            'threshold': threshold,
            'others': correct_others / num_others,
            'others_count': num_others,
            'anger': correct_anger / num_anger,
            'anger_count': num_anger,
            'sadness': correct_sadness / num_sadness,
            'sadness_count': num_sadness,
            'fear': correct_fear / num_fear,
            'fear_count': num_fear,
            'joy': correct_joy / num_joy,
            'joy_count': num_joy,
            'accuracy': sum([correct_others, correct_anger, correct_sadness, correct_fear, correct_joy]) / 1050,
            'joy_labels': joy_labels  # this will convert list into a string
        }
        with open('best_threshold_finder_results.csv', 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=results.keys())
            writer.writeheader()
            writer.writerow(results)
def main():
    tweets_df = load_data()
    find_threshold(tweets_df)
main()
