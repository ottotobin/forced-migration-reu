import os
import glob
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as nnf
import emoji
import argparse
'''
Description:

A script that will loop through csv files containing unlabled Twitter data and assign labels using
binary BETO models for anger, fear, sadness, and joy. The model with the highest probability of the
label being 1 (the emotion) will be chosen as the best model. If this probability is above 0.85, the
tweet is assigned that emotion. Otherwise, it will be assigned "others".

How To Run:

python script.py --unlabeled_dir ./UnlabeledTweets/ --labeled_dir ./LabeledTweets/ --sample True --sample_size 2000
'''

# use gpu 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# use bert tokenizer (same as the one used to train)
tokenizer = BertTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-uncased')
# specifiy emotions that we wish to classify
emotions = ['anger', 'fear', 'joy', 'sadness']

# this function laods all the binary models that we will be using for each emotion
def load_models(emotions):
    models = {}
    # load a binary model for each emotion
    for emotion in emotions:
        model_path = ' '
        # in the case of fear, oversampling provided better results
        if emotion == 'fear':
            model_path = f'./model_bert_{emotion}_oversampled.pt'
        else:
            model_path = f'./model_bert_{emotion}.pt'
        model = BertForSequenceClassification.from_pretrained('dccuchile/bert-base-spanish-wwm-uncased', num_labels=2)
        model.load_state_dict(torch.load(model_path))
        model = model.to(device)
        model.eval()
        models[emotion] = model
    return models

# demojize tweets and tokenize them
def preprocess_tweet(tweet):
    tweet = emoji.demojize(str(tweet)).replace(":", " ").replace("_", " ")
    inputs = tokenizer(tweet, return_tensors='pt', padding=True, truncation=True, max_length=512)
    return inputs['input_ids'].to(device), inputs['attention_mask'].to(device)

# for each tweet, get the emotion probabilities.
def get_emotion_probs(models, input_ids, attention_mask):
    emotion_probs = {emotion: [] for emotion in emotions}
    for emotion, model in models.items():
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.to("cpu")
        probs = nnf.softmax(logits, dim=-1)
        emotion_probs[emotion] = probs[:, 1].item()
    return emotion_probs

# determine which emotion should be assigned based off softmax probabilities
def process_emotion_probs(emotion_probs):
    max_emotion = max(emotion_probs, key=emotion_probs.get)
    # set a threshold of 0.85. If max probability is less than 0.85, assign others
    if emotion_probs[max_emotion] < 0.85:
        return 'others'
    else:
        return max_emotion

def main():
    parser = argparse.ArgumentParser(description="Label Tweets")
    parser.add_argument('--unlabeled_dir', type=str, required=True, help='Path to unlabeled tweets directory')
    parser.add_argument('--labeled_dir', type=str, required=True, help='Path to labeled tweets directory')
    parser.add_argument('--sample', type=bool, default=False,
                        help='A boolean flag indicating if only a sample of the dataframe should be processed.')
    parser.add_argument('--sample_size', type=int, default=2000,
                        help='The number of samples to take if the sample flag is set to True.')
    parser.add_argument('--start_date', type=str, default=None,
                        help='The start date (YYYY-MM-DD) of the date range to process.')
    parser.add_argument('--end_date', type=str, default=None,
                        help='The end date (YYYY-MM-DD) of the date range to process.')
    args = parser.parse_args()
    models = load_models(emotions)
    # get the date range from the start_date and end_date arguments
    date_range = pd.date_range(args.start_date, args.end_date)
    # go through every unlabeled tweet csv
    for filename in glob.glob(os.path.join(args.unlabeled_dir, '*.csv')):
        # extract the date from the filename
        file_date_str = os.path.basename(filename).split('_')[0]
        file_date = pd.to_datetime(file_date_str, format='%Y-%m-%d')
        # check if the date is within the specified range
        # if no start or end date, label everything
        if args.start_date and file_date < date_range[0]:
            continue
        if args.end_date and file_date > date_range[-1]:
            continue
        df = pd.read_csv(filename)
        # some dfs are empty? Skip over them.
        if len(df) == 0:
            continue
        if args.sample:
            df = df.sample(args.sample_size, random_state=11)
        predicted_emotions = []
        probabilities = {emotion: [] for emotion in emotions}
        for i in range(len(df)):
            tweet = df.iloc[i]
            input_ids, attention_mask = preprocess_tweet(tweet['preprocessed_text'])
            emotion_probs = get_emotion_probs(models, input_ids, attention_mask)
            predicted_emotion = process_emotion_probs(emotion_probs)
            predicted_emotions.append(predicted_emotion)
            for emotion in emotions:
                probabilities[emotion].append(round(emotion_probs[emotion], 2))
        # store results into a new dataframe and convert it into a csv file
        df_emotions = pd.DataFrame({'predicted_emotion': predicted_emotions})
        df_probabilities = pd.DataFrame(probabilities)
        df_result = pd.concat([df.reset_index(drop=True), df_emotions.reset_index(drop=True),
                              df_probabilities.reset_index(drop=True)], axis=1)
        new_filename = os.path.join(args.labeled_dir, os.path.basename(filename).replace('unlabeled', 'labeled'))
        df_result.to_csv(new_filename, index=False)

if __name__ == "__main__":
    main()