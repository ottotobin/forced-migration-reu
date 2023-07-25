# code-examples

# Ukrainian Emotion Detection
### Authored by Kate, Eliza, and Apollo

## Task 1: Tokenization
### 1.1. File Descriptors
* `tokenize.py` - script that Reads in a file, cleans langugage, and outputs frequency of tokens
* `ukrainian_emotion.tsv` - tab-separated file containing 30 ukrainian tweets and their labeled emotions
* `ukrainian_emotion_new.tsv` - tab-separated file containing 1520 ukrainian tweets 
### 1.2 How to Run
* `python3 tokenize.py`
## Task 2: Emotion Detection - Dictionary
### 2.1 File Descriptors
* `lexicon.py` - script that cleans data file, predicts emotions for each tweet using dictionary, and outputs accuracies for each emotion
* `Ukrainian-NRC-EmoLex.json` - dictionary containing ukrainian words and their associated emotions, currently not compatible with NRCLex 3.0
* `NRCLex.py` - script that determine word affect based on the lexicon
### 2.2 Parameters
* --emojis
  * Boolean: True to encode emojis as translated words, False to remove them from text
  * Default to False
* --combine 
  * Boolean: True to combine anger and disgust, False to leave them separate
  * Default to False
* --v
  * Version: what emotions to include based on NRCLex output in {1,2,3}
  * Default: 1
  * 1: All non-zero labels
  * 2: Top emotion labels
  * 3: Multiple threshold values
### 2.3 How to Run
* `python3 lexicon.py`
* `python3 lexicon.py --emojis True --combine True --v 3`
## Task 3: Emotion Detection - Classic ML Models
### 3.1 File Descriptors
* `ml_models.py` - script that reads in file, preprocesses data, runs various ML models on a 80/20 train/test split of data, then cross validates 5 times for accuracies
* `misclassified.json` -  Contains mislassified tweets and their humuan label on FP and the predictions for that tweet on FN
### 3.2 Parameters
* --emojis
  * Boolean: True to encode emojis as translated words, False to remove them from text
  * Default to False
* --model 
  * Model: what model to use
  * Default: all
  * Options
    * all: runs all models
    * lr: logistic regression
    * svm: support vector machine
    * rf: random forest
    * dt: decision tree
    * nb: naive bayes
    * knn: k nearest neighbors
* --encode
  * Boolean: True for binary encoding, False for multi class label encoding
  * Default to True
* --combine 
  * Boolean: True to combine anger and disgust, False to leave them separate
  * Default to False
### 3.2 How to Run
* `python3 ml_models.py`
* `python3 ml_models.py --emojis True --model all --encode False`
## Task 4: Emotion Detection - LLMS
### 4.1 File Descriptors
* `bert_multi.py` - BERT model for multi class classification
* `bert_binary.py` - BERT model for binary classification
* `label_tweets.py` - labels tweets using BERT multi class model
* `label_tweets_binary.py` - labels tweets using BERT binary class model
* `helper_funcs.py` - encoding functions, preprocessing functions, any other helper functions needed by models
### 4.2 Parameters
* --learningRate
  * Learning rate for BERT to use
  * Required
* --batches
  * Batch size for BERT to use
  * Required
### 3.2 How to Run
* `python3 bert_multi.py --learningRate 0.000008 --batches 5`
* `python3 bert_binary.py --learningRate 0.00001 --batches 10`
* `python3 label_tweets.py`
* `python3 label_tweets_binary.py`