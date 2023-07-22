import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import argparse
'''
Description:

This script will count all the emotions for every labeled tweet csv file
within the labeled tweets directory. This information is then plotted as 
a time series so that we can visualize emotions as a signal and compare it
to other signals such as migration data.

How to run: 

python script.py --input_dir ./LabeledTweets --output_dir ./OutputDir
'''

# get the emotion counts from the csv files
def process_files(directory, normalize):
    emotions = ['anger', 'fear', 'joy', 'sadness', 'others']
    emotion_counts = {emotion: {} for emotion in emotions}
    # loop through every csv file within the labeled tweets dir
    for filename in glob.glob(os.path.join(directory, '*.csv')):
        date_str = os.path.basename(filename).split('_')[0]
        date = pd.to_datetime(date_str)
        df = pd.read_csv(filename)
        counts = df['predicted_emotion'].value_counts()
        total_tweets = len(df)
        for emotion in emotions:
            count = counts.get(emotion, 0)
            if normalize:
                count /= total_tweets
            emotion_counts[emotion][date] = count
    return emotion_counts

# plot the emotion counts in a time series
def plot_emotion_counts(emotion_counts, output_dir, normalize):
    # plot individual graphs
    for emotion, counts in emotion_counts.items():
        dates, counts = zip(*sorted(counts.items())) 
        plt.figure(figsize=(10,6)) 
        plt.plot(dates, counts, marker='o')
        plt.title(emotion.capitalize() + " Time Series")
        plt.xlabel('Date')
        plt.ylabel('Emotion Count')
        plt.gcf().autofmt_xdate()
        plt.grid(True)
        filename = f'{emotion}_time-series'
        if normalize:
            filename += '_normalized'
        plt.savefig(os.path.join(output_dir, filename + '.png'))
        plt.close()
    # plot combined graph
    plt.figure(figsize=(10,6))
    for emotion, counts in emotion_counts.items():
        dates, counts = zip(*sorted(counts.items()))
        plt.plot(dates, counts, marker='o', label=emotion.capitalize())
    plt.title("All Emotions Time Series")
    plt.xlabel('Date')
    plt.ylabel('Emotion Count')
    plt.gcf().autofmt_xdate()
    plt.grid(True)
    plt.legend(loc='upper right')
    filename = 'all_emotions_time-series'
    if normalize:
        filename += '_normalized'
    plt.savefig(os.path.join(output_dir, filename + '.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Process and plot emotion counts.")
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory containing labeled tweets')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for plots and csv file')
    parser.add_argument('--normalize', default=False, help='Normalize emotion counts to ratios')
    args = parser.parse_args()
    # directories for the labeled tweets and output file containing emo counts
    input_dir = args.input_dir
    output_dir = args.output_dir
    # check if output_dir exists, if not create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_filename = 'emotion_counts'
    if args.normalize:
        output_filename += '_normalized'
    output_file = os.path.join(output_dir, output_filename + '.csv')
    # call necessary functions
    emotion_counts = process_files(input_dir, args.normalize)
    emotion_counts_df = pd.DataFrame(emotion_counts)
    emotion_counts_df.to_csv(output_file)
    plot_emotion_counts(emotion_counts, output_dir, args.normalize)

if __name__ == "__main__":
    main()
