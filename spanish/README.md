# Spanish Emotion Detection
### Authors: Adeline, Bernardo, Colin, and Mattea 

## Directory and File Descriptors

### EmotionDetection Directory
* This directory contains files for emotion detection for the spanish team's dictionary, classic machine learning, and large language models. 
* As of 7/20/23, this directory includes the following sub directories:
    * A `Dictionary` directory which contains a dictionary model for emotion detection with tweets dataset
    * A `MachineLearning` directory that contains classic machine learning models for the same task. The following models were implemented: logistic regression, decision trees, random forests, KNN, SVC,and XGBoost.
    * An `LLMs` directory that contains our best LLM model (BERT, specifically BETO) for the same task. 
    * An `LLMsOld` directory that contains code for several LLMs that we did not use. The following LLMs were implemented: GLoVE, GPT2, BERT (mBERT and BETO), MUSE, and BLOOM.

   #### Dictionary Directory
   * Scripts & Input Files
        * `nrc_spanish.json` - Dictionary containing spanish words and their associated emotions. The dataset was obtained as a text file from this website: https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm. The .txt version can be found in the directory as `Spanish-NRC-EmoLex.txt`
        * `dictionary_spanish.py` - A script that implements NRCLex - a dictionary model for predicting emotions associated with text. We provided `nrc_spanish.json` as the lexicon to the NRCLex object, which assigned emotions to tweets from our dataset: `spanish_emotion.tsv`. It calculates various metrics such as precision, recall, accuracy, f1 score, and emotion prevalence and experiments with emotion preprocessing, feature engineering (ex. combining anger and disgust), and how NRCLex selects the emotion (threshold vs affect list vs top emotion vs random).
        * `NRCLex_spanish.py` - An edited version of the NRCLex source code that uses a spanish lexicon rather than english.
   * Results - Affect List
       * `emo_score_list.json` - Results from EmoLex using affect list, with emojis removed. This shows in json format the tweet, the predicted emotions (affect list) and the label.
        * `emo_score_list_emoji.json` - Results from EmoLex using affect list, without processing emojis. This shows in json format the tweet, the predicted emotions (affect list) and the label.
        * `emo_score_stats_affect_list.json` - Results from Emolex using affect list, with emojis removed. This shows in json format various evaluation metrics for each emotion. This includes true positives, false positives, false negatives, accuracy, prevlaence, precision, recall, and f1 score.
        * `emo_score_stats_affect_list_emoji.json` - Results from Emolex using affect list, without processing emojis. This shows in json format various evaluation metrics for each emotion. This includes true positives, false positives, false negatives, accuracy, prevlaence, precision, recall, and f1 score.
   * Results - Top Emotion
        * `top_emo_score_list.json` - Results from EmoLex using the top emotions rather than the entire affect list, with emojis removed. This contains the tweet, the predicted top emotions and the label.
        * `top_emo_score_list_emoji.json` - Results from EmoLex using the top emotions rather than the entire affect list, without processing emojis. This contains the tweet, the predicted top emotions and the label.
        * `emo_score_stats_top.json` - Results from Emolex using top emotions, with emojis removed. This shows in json format various evaluation metrics for each emotion. This includes true positives, false positives, false negatives, accuracy, prevlaence, precision, recall, and f1 score.
        * `emo_score_stats_top_emoji.json` - Results from Emolex using top emotions, without processing emojis. This shows in json format various evaluation metrics for each emotion. This includes true positives, false positives, false negatives, accuracy, prevlaence, precision, recall, and f1 score.
   * Results - Threshold 
        * `top_emo_threshold.json` - Results from EmoLex using a threshold parameter, with emojis removed. This contains the tweet, the predicted top emotions and the label.
        * `top_emo_threshold_emoji.json` - Results from EmoLex using a threshold parameter, without processing emojis. This contains the tweet, the predicted top emotions and the label.
        * `emo_score_stats_threshold.json` - Results from Emolex using top emotions with a threshold parameter, with emojis removed. This shows in json format various evaluation metrics for each emotion. This includes true positives, false positives, false negatives, accuracy, prevlaence, precision, recall, and f1 score.
        * `emo_score_stats_threshold_emoji.json` - Results from Emolex using top emotions with a threshold parameter, without emoji processing. This shows in json format various evaluation metrics for each emotion. This includes true positives, false positives, false negatives, accuracy, prevlaence, precision, recall, and f1 score.
   * DEPENDENCIES:
       * Install the following packages prior to running the program:
       ```
       pip install numpy pandas nltk emoji scikit-learn
       ```
   * TO RUN: 
   ```
   python3 dictionary_spanish.py
   ```
   
   #### MachineLearning Directory
   * This directory contains all files associated with implementations of classic machine learning models. It has the following scripts:
       * `ml_spanish.py` - Script that reads in data from `spanish_emotions.tsv` file using various machine learning models to and calculates the accuracy and f1 score. The following machine learning models were used: KNN, Decision Trees, Random Forests, Logistic Regression
       * `output.txt` - Contains the resulting accuracies, f1 scores, and confusion matrix from the various machine learning models.
   * The next files will be of four categories. Additional text after the base file name will indicate what preprocessing/sensitivity analysis was applied.
       * `best_accuracy.txt` - Contain the model with the best accuracy for the best emotion:
       * `best_f1.txt` - Contains the model with the best F1 score for the best emotion
       * `model_stats.txt` - Contains F1 and accuracy scores for all models on all emotions in json format. This was done to make the output more readable. Only contains stats for binary models.
       * `output.txt` - This contains the full output for all the stats for all the models and emotions.
   * This is a description of what the addons for each file represent:
       * `Nothing`- If nothing was included after the base file name, no preprocessing or parameters were applied.
       * `_no_emoji` - Emoji preprocessing was applied.
       * `_no_disgust` - Anger and disgust combined into one emotion.
       * `_balanced` - Dataset is balanced using undersampling.
       * `_grid_search` - Grid search was applied on every model.
   * DEPENDENCIES:
       * Install the following packages prior to running the program:
       ```
       pip install numpy nltk scikit-learn pandas xgboost imbalanced-learn argparse emoji
       ```
   * TO RUN:
       * No parameters:
       ```
       python3 ml_spanish.py
       ```
       * Find parameters using grid search (only for decision trees and random forests) and allow for emoji preprocessing:
       ```
       python ml_spanish.py --gridsearch True --emoji True --disgust True --balance True
       ```
   
   #### LLMsOthers Directory
   * This directory contains all files where we experimented with implementations of large language models to find the optimal model for labeling Twitter data. For this task, we used GLoVE, GPT2, mBERT, MUSE, BLOOM to process our Spanish text dataset and predict associated emotions. It has the following scripts:
       * `gpt2_glove.py` - This script applies GPT-2 and GloVe models for emotion classification on the Spanish tweet data. It trains the models, evaluates them, and outputs the results as text files.
           * How to run:
               * Run the script with specified learning rate, epochs, batch size, and model type. The following example runs the GPT-2 model with a learning rate of 0.01, for 10 epochs and a batch size of 32:
               ```
               python gpt2_glove.py --lr 0.01 --epochs 10 --bs 32 --model gpt2
               ```
               If no parameters are provided, the script will run with default values.
           * Description of the arguments:
               * `lr` - Learning rate for the model training. Default values are [0.001, 0.01, 0.1].
               * `epochs` - Number of epochs for model training. Default values are [5, 10, 15].
               * `bs` - Batch size for training. Default values are [16, 32, 64].
               * `model` - Model to be used ('gpt2', 'glove', or 'both'). The default is 'both'.
       * `glove_results` - A directory contained all results for different parameter combinations for GloVe.
       * `gpt2_results` - A directory contained all results for different parameter combinations for GPT2.
       * `multilingual_embeddings.es` - A file that contains GloVe embeddings for the spanish language. The embeddings were taken from https://github.com/dcferreira/multilingual-joint-embeddings.
       * `mbert_spanish.py` - This script applies BERT models for classification on Spanish tweet data. It trains and evaluates the model on several different batch sizes, and returns accuracy and f1 metrics for each respective batch test in the 'mbert_output.txt' file.
            * How to run:
              ```
              python mbert_spanish.py
              ```
      * `mbert_output.txt` - This text file contains the accuracy scores, f1 scores, and confusion matrices for the multiclass BERT model for each batch size. It returns the training dataset metrics and the testing dataset metrics.
      * `muse_2_spanish.py` - This script applies (Facebook Research) MUSE model for classification on spanish_emotion.tsv a file of Spanish tweets that are preprocessed before the model trains and evaluates itself using differing parameters. Parameters: learning_rate, epochs, momentum, and batch size. It prints and outputs to 'parameter_results.csv' f1 scores and accuracy scores for each emotion in test runs.
         * How to run:
              ```
              python muse_2_spanish.py
              ```
       * `parameter_results.csv` - This csv file contains the information on parameters ran and each emotion's accuracy scores, f1 scores. It also determines which parameters were most accurate based on f1 score. NOTE: There seems to be a weird amount of duplicate f1 and accuracy scores. 
   * DEPENDENCIES:
       * Install the following packages prior to running the programs. These are all the dependencies for all of the scripts:
       ```
       pip install transformers torch pandas numpy scikit-learn torchtext imbalanced-learn emoji spacy tqdm argparse
       python -m spacy download es_core_news_sm
       ```
   
   #### LLMsBest Directory (best model):
   * This directory contains the script for training our best binary classifiers for each emotion. This directory also contains other scripts for labeling data and visualizing the data.
       * `bert_binary.py` - This file uses the pretrained BETO (a variant of BERT) imported from Pytorch to train binary classifiers for emotion detection.
           * How to run:
               * Run the script with specified learning rate, batch size, epochs, and boolean arguments for running the script with oversampling or running the model on a subset of the training data. Example:
               ```
               python3 bert_binary.py --epochs 5 --test True
               ```
           * Description of the arguments:
               * `filename` - File name for training dataset file. Default is `spanish_emotion.tsv`.
               * `learningrate` - Learning rate for the model training. Default value is 1e-5.
               * `batchsize` - Batch size for model training. Default is 16.
               * `epochs` - Number of epochs for model training. Default value is 10.
               * `oversample` - Whether to oversample the data. Default is True.
               * `test` - Whether to train and test the models using a subset of the original data. Default is False.
       * `bert_binary_results.txt` - This file contains the results from running `bert_binary.py` with default parameters. It contains the average loss, accuracy, f1 score, and confusion matrix for each emotion model.
       * `get_stats.py` - This script will get the emotion counts from the results of labeling tweets with our best models and create a visualization using this data.
       * `label_tweets.py` - This script will label the unlabled Twitter data using our best model.
       * `spanish_emotion.tsv` - The training data for our models.
       * `predicted_labels_sample.csv` - A sample of 2,000 tweets labeled with our best models using a softmax threshold of 0.85.
   * DEPENDENCIES:
       * Install the following packages prior to running the programs. These are all the dependencies for all of the scripts.
       ```
       pip install transformers pandas torch emoji scikit-learn tqdm argparse imbalanced-learn matplotlib glob2
       ```
