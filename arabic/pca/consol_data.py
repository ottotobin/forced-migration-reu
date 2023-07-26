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
                
def read_iom(file):
    with open(file, "r") as f:
        df = pd.read_csv(f)

    # Helper function to fill missing pcode values
    def fill_missing_pcode(row, pcode_dict):
        if pd.isna(row["STATE PCODE OF ORIGIN"]):
            state = row["STATE OF ORIGIN"]
            code = pcode_dict[state]
            return code
        return row["STATE PCODE OF ORIGIN"]

    # Get dictionary of mappings from each state to pcode to pass to apply function
    pcode_dict = df[["STATE OF ORIGIN","STATE PCODE OF ORIGIN"]].dropna(subset=["STATE PCODE OF ORIGIN"])\
        .set_index("STATE OF ORIGIN")["STATE PCODE OF ORIGIN"].to_dict()
    # Use apply to replace missing values
    df["STATE PCODE OF ORIGIN"] = df.apply(fill_missing_pcode, axis=1, args=(pcode_dict,))

    # List of columns that we will be using from the iom data
    # Some of the week-datasets did not have camp/neighborhood data,
    # so we are using locality as the smallest granularity
    cols_of_interest = ["Date","STATE OF AFFECTED POPULATION","LOCALITY OF AFFECTED POPULATION","# IDP INDIVIDUALS"]
    
    # Convert columns to appropriate data types for summing/grouping
    df[cols_of_interest[-1]] = pd.to_numeric(df[cols_of_interest[-1]].str.replace(",",""), errors="coerce")
    df[cols_of_interest[0]] = pd.to_datetime(df[cols_of_interest[0]])

    # Drop missing values
    df = df[cols_of_interest].dropna()

    # Group rows by date, state, and locality and sum their IDP counts
    df = df.groupby(cols_of_interest[:-1], as_index=False, axis=0)[cols_of_interest[-1]].sum()

    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-acled", default='data/acled.csv')
    parser.add_argument("-iom", default='data/iom.csv')
    args = parser.parse_args()
    
    iom_df=read_iom(args.iom)

    acled_path = args.acled 

    read_acled(acled_path)

if __name__ == "__main__":
    main()