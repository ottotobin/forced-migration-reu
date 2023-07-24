import os
import json
import glob
import pandas as pd

# get the list of all dates (directories) within the 'task5_tweets' directory
dates_dir = glob.glob('./venezuela/*')

for date_dir in dates_dir:
    # initialize an empty list to store the rows
    rows = []
    # need to add this weird directory bc that's how it's structured
    base_dir = os.path.join(date_dir, 'twitter_Sample_preprocessed_text.csv')
    # iteratively go through each date, city, and json file
    for date_dir in glob.glob(f'{base_dir}/date=*'):
        for city_dir in glob.glob(f'{date_dir}/city=*'):
            # fix any messed up backslashes
            city_dir = city_dir.replace("\\", "/")
            # get the date and city from the path
            _, date_path, city_path = city_dir.rsplit('/', 2)
            date = date_path.split('=')[1]
            city = city_path.split('=')[1]
            for file_path in glob.glob(f'{city_dir}/*.json'):
                with open(file_path, 'r', encoding='utf-8') as file: 
                    for line in file:
                        # load json data and get the tweet and tweet id
                        data = json.loads(line)
                        if data["lang"] == "es":
                            tweet = data['preprocessed_text']
                            id_str = data['id_str']
                            # append results to list
                            rows.append({
                                'location': city,
                                'id_str': id_str,
                                'date': date,
                                'preprocessed_text': tweet,
                            })
    # create DataFrame once per directory
    df_results = pd.DataFrame(rows, columns=['location', 'id_str', 'date', 'preprocessed_text'])
    # record data in csv file for each date
    df_results.to_csv(f'./UnlabeledTweets/{date}_unlabeled_tweets.csv', index=False, encoding='utf-8')
