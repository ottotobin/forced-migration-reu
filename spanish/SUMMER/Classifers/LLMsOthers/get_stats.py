'''
Description:

This script obtains the emotion counts within a labeled emotion csv file so we can visualize which emotions are being
predicted the most/least often. This helps us determine if our model os over/under representing certain emotions. 
'''

import pandas as pd
import matplotlib.pyplot as plt

# df = pd.read_csv('predicted_labels_2000.csv')
df = pd.read_csv('predicted_labels_multilingual.csv')

count = 0
others = 0
anger = 0
fear = 0
sadness = 0
joy = 0

for emotion in df['predicted_emotion']:
    count += 1
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

labels = ['Anger', 'Fear', 'Sadness', 'Joy', 'Others']
sizes = [anger, fear, sadness, joy, others]
colors = ['red', 'orange', 'blue', 'yellow', 'purple']
explode_array = [0.1, 0.4, 0.4, 0.1, 0.0]

fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, colors=colors, startangle=90, explode = explode_array, autopct='%1.2f%%')
ax.set_title('Predicted Label Emotion Distribution\nBinary, 2000 Tweets', pad=10)
plt.show()


