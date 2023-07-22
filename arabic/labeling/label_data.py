"""
Authors: Lina Laghzaoui, Grace Magny-Fokam, Toby Otto, Rich Pihlstrom

Description:
    The program uses the saved models from llm.py to label an unlabeled dataset. The data that
    we are using is from cloudshare. As such, we have methods to aggregate the json files in
    specific dircetory structure provided. These methods will output an aggregated csv, 
    so if that exists already, you don't have to have all the cloudshare stuff on your machine.
    For the labeling, we follow a similar preparation procedure as in llm.py, but instead of training
    and testing, we load in the already-trained data and run it on the unlabeled data.

Necessary Packages:
    You will need to install pytrends in order to use the API in this code:
        pip3 install fasttext transformers torch scikit-learn

csDir: the path to the directory that has the cloudshare data. As stated above, this is not
    necessary if you are not re-aggregating your cloudshare data

model: the name of the model that you want to label. The outputs from llm.py follow a naming convention
    that includes the model name in the filename, so you should just have to put in the model name
    for everything to work properly.

USAGE:
    python3 label_data.py -m <muse, GloVe, bert2> -csDir "data/cloud-share"
"""

import re, os, csv, torch, datetime, fasttext.util, transformers, argparse
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from nltk.tokenize import RegexpTokenizer
from transformers import BertForSequenceClassification

# Function to preprocess the csv data
# Takes input of the original data file and the emoji flag
# Outputs a csv file and return the dataframe version of that file

# NB: from previous classification methods, we determined that emoji translation
# did not impact results much so, in the interested of minimizing tests, we had
# the emoji flag as false for all testing.
def preprocess(data_file, emoji_flag=0, lang = []):
    # If the output file already exists, we can just read
    # it and return the dataframe
    if os.path.exists("data/processed_cloudshare_data.csv"):
        with open("data/processed_cloudshare_data.csv", "r") as f:
            df = pd.read_csv(f)
    else:
        # read text data
        df = pd.read_csv(data_file, encoding="utf8")

        # preprocess text - remove @s, #s, links, and emojis
        emoji_pattern = re.compile(
            "["
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "]+",
            flags=re.UNICODE,
        )
        processed_tweets = []

        #Tokenize and process each tweet
        for token_list in tokenize(df["raw_tweet"]):
            #Translate the emojis and add/remove to sentence
            text = emoji_translator(token_list, emoji_flag)
            text = " ".join(
                word
                for word in text.split()
                if not (
                    word.startswith("http")
                    or word.startswith("@")
                    or word.startswith("#")
                )
            )
            text = emoji_pattern.sub(r"", text)
            text = text.lower()
            processed_tweets.append(text)
        
        # Create a new column for processed tweets
        df["processed_tweets"] = processed_tweets

        df.to_csv("data/processed_cloudshare_data.csv", index=False)

    # check if iso code(s) have been passed to indicate whether 
    # we shoudl filter out langauges in the unlabeled data
    if len(lang) > 0:
        df = df[df['lang'].apply(lambda x: any(item in lang for item in x))]
        
    return df
    
# tokenizes a list of tweets
def tokenize(tweets):
    regexp = RegexpTokenizer("\w+")
    tweets = tweets.map(str)
    token_col = tweets.apply(regexp.tokenize)
    return token_col

# translates emojis to Arabic text using emojis.csv
def emoji_translator(token_list, emoji_flag):
    return_tweet = ""
    emojis_df = pd.read_csv("data/emojis.csv")

    for token in token_list:
        # checks to see if emoji_flag is True, otherwise just omits the emoji altogether
        if token in emojis_df["emoji"].values and emoji_flag:
            return_tweet += emojis_df[emojis_df["emoji"] == token]["text"].values[0]
        else:
            return_tweet += token
        return_tweet += " "

    return return_tweet

def combine_cloudshare_data(directory):
    outfile = "data/cloud-share_data.csv"
    # if already created csv
    if os.path.exists(outfile):
        return outfile

    with open(outfile, "w+") as f:
        writer = csv.writer(f)
        header = ["date", "city", "tweet_id", "raw_tweet", "language"]
        writer.writerow(header)
        
        rows=[]
        # nestings of for loops to iterate over all of the 
        # json files stored in the larger cloud shre directory
        for filename in os.listdir(directory):
            path_date = os.path.join(directory, filename)
            if os.path.isdir(path_date):
                # data dir
                date = datetime.datetime.strptime(filename, "%Y%m%d").strftime("%Y-%m-%d") 

                path_date2 = os.path.join(path_date, "twitter_Sample_preprocessed_text.csv",f"date={date}")
                for city_dir in os.listdir(path_date2):
                    # city= dir
                    city = city_dir.split("=")[1]

                    path_city = os.path.join(path_date2, city_dir)
                    for json_file in os.listdir(path_city):
                        # json file
                        json_path = os.path.join(path_city, json_file)
                        jsonObj = pd.read_json(json_path, lines=True)
                        for id, text, lang in zip(jsonObj["id_str"], jsonObj["preprocessed_text"], jsonObj["lang"]):
                            row = [date, city, id, text.replace("\n","").replace("\r",""), lang]
                            if row not in rows:
                                rows.append(row)
                                writer.writerow(row)
    
    return outfile

def label_bert(emotions, CS_data, best_param_df):
    df_out = CS_data
    test_data = CS_data["processed_tweets"]

    # load saved model and set everything up
    for i, emotion in enumerate(emotions):
        # get threshold
        threshold = best_param_df[best_param_df["emotion"]==emotion]["threshold"].iloc[0]

        print(f"    {emotion}")
        # perpare bert model and load from best saved model for the emotion
        transformers.logging.set_verbosity_error()
        model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-uncased', num_labels=6)   
        model.load_state_dict(torch.load(f"best_models/bert2_{emotion}.pt", map_location=torch.device('cpu')))
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased', return_token_type_ids=False)
        model.eval()

        test_data_encoded_inputs = tokenizer(test_data.tolist(), padding=True, truncation=True, return_tensors="pt")
        test_data_input_ids = test_data_encoded_inputs["input_ids"]
        test_data_attention_mask = test_data_encoded_inputs["attention_mask"]

        # get test outputs
        with torch.no_grad():
            test_outputs = model(input_ids=test_data_input_ids, 
                                    attention_mask=test_data_attention_mask)

        # Apply sigmoid and obtain the predicted probabilities for testing data
        sigmoid = torch.nn.Sigmoid()
        pred = sigmoid(test_outputs.logits)

        # check if logit values meet minimum threshold and collect accordingly
        test_predicted_labels=[]
        for vect in pred:
            if vect.tolist()[i] >= threshold:
                test_predicted_labels.append(1)
            else:
                test_predicted_labels.append(0)

        #test_predicted_labels = [round(vect.tolist()[i], 0) for vect in test_predicted_probabilities]
        df_out[emotion] = test_predicted_labels
    
    df_out.to_csv("output/labels/bert2.csv", index=False)

    return df_out

def label_muse(emotions, CS_data, best_param_df):
    df_out = CS_data
    test_data = CS_data["processed_tweets"]

    for emotion in emotions:
        threshold = best_param_df[best_param_df["emotion"]==emotion]["threshold"].iloc[0]
        print(f"    {emotion}")

        # prepare model and load saved model for each emotion
        fasttext.util.download_model("ar", if_exists="ignore")
        ft = fasttext.load_model("cc.ar.300.bin")

        test_transformed = np.array([ft.get_sentence_vector(x) for x in test_data])
        test_tensor = torch.tensor(test_transformed, dtype=torch.float32)

        input_size = test_tensor.shape[-1]
        output_size = 2
        model = torch.nn.Linear(input_size, output_size)
        model.load_state_dict(torch.load(f"best_models/muse_{emotion}.pt", map_location=torch.device('cpu')))

        # collect predictions and check if they meet threshold
        m = torch.nn.Softmax(dim=1)
        pred = m(model(test_tensor)).tolist()
        #pred_labels = [np.argmax(i) for i in pred]
        pred_labels = []
        for p in pred:
            if p[1] > p[0] and p[1] >= threshold:
                pred_labels.append(1)
            else:
                pred_labels.append(0)
        df_out[emotion] = pred_labels

    df_out.to_csv("output/labels/muse.csv", index=False)

    return df_out

def label_glove(emotions, CS_data, best_param_df):
    df_out = CS_data
    test_data = CS_data["processed_tweets"]

    fasttext.util.download_model("ar", if_exists="ignore")
    fasttext_model = fasttext.load_model("cc.ar.300.bin")

    indices = {}
    embeddings = []

    # set up transform list of GloVe
    i = 0
    for word in fasttext_model.words:
        if word not in indices:
            indices[word] = i
            i += 1
        embeddings.append(fasttext_model[word])
    transformed = []
    for tweet in tokenize(test_data):
        sentence = []
        for word in tweet:
            if word in indices:
                sentence.append(embeddings[indices[word]])
        vect_sum = sum(sentence)
        if type(vect_sum) == int:
            vect_sum = np.zeros(transformed[0].shape, dtype=np.float32)
        transformed.append(vect_sum)

    for emotion in emotions:
        threshold = best_param_df[best_param_df["emotion"]==emotion]["threshold"].iloc[0]
        print(f"    {emotion}")

        test_tensor = torch.tensor(np.array(transformed))

        # load best model for each emotion 
        input_size = test_tensor.shape[-1]
        output_size = 2
        model = torch.nn.Linear(input_size, output_size)
        model.load_state_dict(torch.load(f"best_models/GloVe_{emotion}.pt", map_location=torch.device('cpu')))

        # collect predictions and check if they meet the threshold
        m = torch.nn.Softmax(dim=1)
        pred = m(model(test_tensor)).tolist()
        pred_labels = []
        for p in pred:
            if p[1] > p[0] and p[1] >= threshold:
                pred_labels.append(1)
            else:
                pred_labels.append(0)

        #pred_labels = [np.argmax(i) for i in pred]
        df_out[emotion] = pred_labels

    df_out.to_csv("output/labels/GloVe.csv", index=False)

    return df_out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", nargs="+", type=str, 
                        help="The name of the model you want to label", required=True)
    parser.add_argument("-c", "--csDir", nargs=1, type=str,
                        help="The path to the cloudshare directory")
    args = parser.parse_args()

    models = {
        "bert2":label_bert,
        "muse":label_muse,
        "GloVe":label_glove
    }

    emotions = ["anger", "fear", "sadness", "disgust", "joy", "anger-disgust"]
    df = combine_cloudshare_data(args.csDir[0])
    processed = preprocess(df)

    with open("output/Best_Params.csv", "r") as f:
        best_param_df = pd.read_csv(f)

    lab_dict = {}
    for m in args.model:
        print(f"Labelling {m}")
        model = models[m]
        emotion_best_param = best_param_df[best_param_df["model"]==m]
        lab_dict[m] = model(emotions, processed, emotion_best_param)
        print

if __name__ == "__main__":
    main()
