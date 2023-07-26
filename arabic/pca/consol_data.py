"""
authors: Toby, Rich, Lina, Grace


"""

import os
import argparse
import pandas as pd
import numpy as np

def read_acled(path):
    with open(path, 'r') as acled:
        df = pd.read_csv(acled)
        ret_df = pd.DataFrame({'date': [], 'event_count': [], 'fatalities' : []})
        fatalities = 0
        event_count = 0
        event_dates = df['event_date']
        prev_date = event_dates[0]

        for index, row in df.iterrows():
            print(row['event_date'])
            if row['event_date'] != prev_date:
                ret_df.loc[len(df.index)] = [prev_date, event_count, fatalities]
                event_count = 0
                fatalities = 0
            else:
                event_count += 1
                fatalities += row['fatalities']
            
        print(ret_df)
                
def read_iom():
    return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-acled", default='data/acled.csv')
    parser.add_argument("-iom", default='data/iom.csv')
    args = parser.parse_args()
    
    read_iom(args.iom)

    acled_path = args.acled 

    read_acled(acled_path)

if __name__ == "__main__":
    main()