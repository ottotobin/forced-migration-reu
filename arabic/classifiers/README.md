
# code-examples

# authors: Rich, Toby, Lena, Grace

# Directory and File Descriptors
- llmClassifier: directory that contains the large language model files
    - llm.py: python file that contains code for all llms (GLoVE, MUSE, BERT, GTP2, BLOOM), sensitivity testing for any given model, and code to save the best parameters of that model
    - label_data.py: python file that labels our unlabeled data w/ a the best version of a model for each emotion
    - best_models: directory contains the model's best version for each emotion
    - data: pre-labeled data set, unlabaled data (cloud-share)
    - data/tuning_params.json: the json dictionary w/ the parameters choices to sensitivity test on. to conduct your own sensitivity testing you can manually change the json dictionary to test for parameters as you see fit
    - output: broken down into 1) labels that contains the labeling done with a certain model, and 2) the paramTuning for a given model
- mlClassifier: directory that holds the files for the classic machine learning classifiers
- DictClassifier: directory that holds the files for the dictionary (lexicon) classifer code

# Running llm Script
- terminal command: python3 llm.py -m <GLovE, BERT, MUSE> -s <[0,1]>

# Running label_data. Script
- terminal command: python3 label_data -m <GLoVE, BERT, MUSE>

# Running  lexiconClassifier Script
- terminal command: python3 lexiconClassifier.py --threshold <threshold_value>

# Running mlClassifiers Script
- terminal command: python3 mlClassifiers.py

# Output llm Script
- {model}_{emotion}.pt: the best version of each model is saved as a Pytorch file in best_models
- {model}.csv: the best parameters for a given model is saved to output/paramTuning

# Output label_data Script
- output/labels: {model}.csv contains the labeling done for w/ that model on the cloud-share unlabeled data

# Output of lexiconClassifier Script
- ouputs a csv file, outfile.csv
- this file contains evaluation metrics for the dictionary model for each emotion both including and excluding emojis

# Output of mlClassifiers Script
- outputs .csv files for the evaluation metrics of all the different ML models
- the files are organized by column with the metrics we were evaluating with and by row with the emotion and model for that
- then prints to the command line the predictions using GridSearch hyperparameter optimization along with the best parameters for that model
