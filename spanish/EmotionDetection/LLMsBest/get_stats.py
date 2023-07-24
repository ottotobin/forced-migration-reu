import pandas as pd
import matplotlib.pyplot as plt

# load and return data
def loadData(path):
    df = pd.read_csv(path)
    return df

# get emotion counts
def countEmotions(df):

    # initialize counter variables
    others = 0
    anger = 0
    fear = 0
    sadness = 0
    joy = 0

    # loop through dataframe to count emotions
    for emotion in df['predicted_emotion']:
        if str(emotion.strip()) == 'others':
            others += 1
        elif str(emotion.strip()) == 'anger':
            anger += 1
        elif str(emotion.strip()) == 'fear':
            fear += 1
        elif str(emotion.strip()) == 'sadness':
            sadness += 1
        elif str(emotion.strip()) == 'joy':
            joy += 1
        else:
            print("emotion not recogized")

    # return counter variables
    return anger, fear, sadness, joy, others

# visualize data
def getViz(anger, fear, sadness, joy, others):

    # initialize label list
    labels = ['Anger', 'Fear', 'Sadness', 'Joy', 'Others']

    # put count variables in a list
    sizes = [anger, fear, sadness, joy, others]

    # initialize color list
    colors = ['red', 'orange', 'blue', 'yellow', 'purple']

    # initialize explode value list
    explode_array = [0.1, 0.4, 0.4, 0.1, 0.0]

    # vizualize data
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, colors=colors, startangle=90, explode = explode_array, autopct='%1.2f%%')
    ax.set_title('Predicted Label Emotion Distribution\nBinary, 2000 Tweets', pad=10)
    plt.show()
    plt.savefig('prediction_distribution.png')

def main():
    # load predicted label data
    df = loadData('predicted_labels.csv')

    # get counts for each emotion
    anger, fear, sadness, joy, others = countEmotions(df)

    # get visualization
    getViz(anger, fear, sadness, joy, others)

if __name__ == "__main__":
    main()