"""
authors: Toby, Rich, Lina, Grace


"""

import os
import argparse
import pandas as pd
import numpy as np

def read_acled(path):
    with open(path, 'r') as acled:
        acled_df = pd.read_csv(acled)
        
        print(acled_df.columns)
        exit()

def read_iom(path):
    for file in os.listdir(path):
        df = pd.read_csv(os.path.join(path, file))
        print(df.columns)
    
    exit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-acled", default='data/acled/2018-07-2023-07-25-Northern_Africa-Sudan.csv')
    args = parser.parse_args()
    
    #read_iom("data/iom/")

    acled_path = args.acled 

    read_acled(acled_path)

if __name__ == "__main__":
    main()