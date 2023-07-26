"""
authors: Toby, Rich, Lina, Grace


"""

import os
import argparse
import pandas as pd
import numpy as np

# reads in an ACLED csv file and converts it into a datafram
# containing dates, total fatalities on that day, and the number of events per day
def read_acled(path):
    with open(path, 'r') as acled:
        df = pd.read_csv(acled)
        df["event_count"] = 1
        # columns to do a combined search by
        cols_interest = ['event_date', 'admin1', 'admin2', 'location']

        # groups the dataframe by event date and sums their fatalities and events
        ret_df = df.groupby(cols_interest[:-3], as_index=False, axis=0).agg({"fatalities":"sum","event_count":"sum"})

        return ret_df
  
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
    
    # consolidate iom and acled data into smaller dataframes
    iom_df = read_iom(args.iom)
    acled_df = read_acled(args.acled)

if __name__ == "__main__":
    main()