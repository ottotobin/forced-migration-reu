#########
# pca_range.py: conducting principle component analysis on the 
# monthly 5 year Google Trends data

# authors: Rich, Toby, Lina, Grace

# last modified: 7/13/23
# - created file

#########
import os
import matplotlib.pyplot as plt
import pickle
import numpy as np
import matplotlib.dates as mdates
import pandas as pd

# reads in data for each catergory for a given ISO code
def load_data(iso, categories):
    retDict = {}  

    for cat in categories:
        if os.path.exists(f"results/{cat}"):
            retDict[cat] = pd.read_csv(f"results/{cat}/{cat}_{iso}_2018-01-01-to-2023-05-31_adjusted.csv")

    return retDict

# aggregates all intensities of words in each dataframe
def aggregate(df_dict):
    retDict = {}
    for key in df_dict:
        df = df_dict[key]
        key_words = df.columns[3:]
        for column in key_words:
            if len(df[column].unique()) == 1 and df[column].unique()[0] == 0:
                df = df.drop(column, axis=1)

        df["aggregate"] = 0
        if len(df.columns) > 3:
            for column in df.columns[3:]:
                df["aggregate"] += df[column] / len(df.columns[3:])
        
        # normalize aggregates
        df["aggregate"] = (df["aggregate"] - np.mean(df["aggregate"])) / np.std(df["aggregate"])
        
        retDict[key] = df

    return retDict

def concat_df(df_dict):
    columns_of_interest = ["date","aggregate"]
    retDf = pd.DataFrame()
    for key in df_dict:
        if retDf.empty:
            retDf = df_dict[key][columns_of_interest]
            retDf = retDf.rename(columns={"aggregate":key})
        else:
            retDf[key] = df_dict[key]["aggregate"]
        
    return retDf

def time_series(df):


    
def main():
    iso = "SD"
    categories = ["generic_terms","geography","safe_places","travel"]

    df = concat_df(aggregate(load_data(iso, categories)))
    # print(df)


if __name__ == '__main__':
    main()