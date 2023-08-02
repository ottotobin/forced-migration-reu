# Ukrainian Emotion Detection
## Authored by Kate Liggio, Eliza Salamon, and Apollo Callero 
###     Summer 2023


## 1. Emotion Detection - Dictionary
### 1.1 File Descriptors
* `lexicon_model.py` - script that cleans data file, predicts emotions for each tweet using dictionary, and outputs accuracies for each emotion
* `Ukrainian-NRC-EmoLex.json` - dictionary containing ukrainian words and their associated emotions, currently not compatible with NRCLex 3.0
* `NRCLex.py` - script that determine word affect based on the lexicon
### 1.2 Parameters
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
### 1.3 How to Run
* `python3 lexicon.py`
* `python3 lexicon.py --emojis True --combine True --v 3`
## 2. Emotion Detection - ML
### 2.1 File Descriptors
* `ml_models.py` - script that reads in file, preprocesses data, runs various ML models on a 80/20 train/test split of data, then cross validates 5 times for accuracies
* `misclassified.json` -  Contains mislassified tweets and their humuan label on FP and the predictions for that tweet on FN
* `gridsearch.py` - Performs a grid search on each ML model to find best parameters
### 2.2 Parameters
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
### 2.3 How to Run
* `python3 ml_models.py`
* `python3 ml_models.py --emojis True --model all --encode False`
## 3. Emotion Detection - LLMS
### 3.1 File Descriptors
* `bert_multi.py` - BERT model for multi class classification
* `bert_binary.py` - BERT model for binary classification
* `label_tweets.py` - labels tweets using BERT multi class model
* `label_tweets_binary.py` - labels tweets using BERT binary class model
* `label_tweets_threshold.py` - labels tweets using BERT multi class model and a threshold
### 3.2 Parameters
* --learningRate
  * Learning rate for BERT to use
  * Required
* --batches
  * Batch size for BERT to use
  * Required
### 3.3 How to Run
* `python3 bert_multi.py --learningRate 0.000008 --batches 5`
* `python3 bert_binary.py --learningRate 0.00001 --batches 10`
* `python3 label_tweets.py`
* `python3 label_tweets_binary.py`
* `python3 label_tweets_threshold.py --threshold 0.8`
## 4. Indicators
* Folder for ACLED, Google Trends, and Emotion data
### 4.1 File Descriptors
* `acled_corr.ipynb`
* `clean_data.ipynb`
* `emotions_corr.ipynb`
* `emotions_explore.ipynb`
* `pca_and_regression.py` - visualizes and analyzes trends dataset with plots, normalizations, correlation analyses, and PCA
* `process_trends.py` - processes trends data files, categorizes, aggregates, cleans, and restructures the data
* `trends_corr.ipynb`
### 4.2 How to Run
* `python3 pca_and_regression.py`
* `python3 process_trends.py`
## Other Files
* `helper_funcs.py` - encoding functions, preprocessing functions, any other helper functions needed by models
* data folder - training data for models, json files for GLoVe and emojis
* `requirements.txt` - requirements for running code
