"""
authors: Toby, Rich, Lina, Grace


"""

import os
import argparse
import pandas as pd
import numpy as np

  
# reads in IOM data and consolidates it into a dataframe containing the 
# dates, locations, and internally displacement persons count, and leaving person
# count in Sudan
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
    cols_of_interest = ["Date","STATE OF AFFECTED POPULATION","# IDP INDIVIDUALS", "STATE OF ORIGIN"]
    
    # Convert columns to appropriate data types for summing/grouping
    df[cols_of_interest[cols_of_interest.index("# IDP INDIVIDUALS")]] = pd.to_numeric(df[cols_of_interest[cols_of_interest.index("# IDP INDIVIDUALS")]].str.replace(",",""), errors="coerce")
    df[cols_of_interest[cols_of_interest.index("Date")]] = pd.to_datetime(df[cols_of_interest[cols_of_interest.index("Date")]])

    # Drop missing values
    df = df[cols_of_interest].dropna()

    # Group rows by date, state, and locality and sum their IDP counts
    df = df.groupby(cols_of_interest, as_index=False, axis=0)[cols_of_interest[cols_of_interest.index("# IDP INDIVIDUALS")]].sum()

    date_list = df["Date"].unique()
    row_list = []
    for i in range(len(date_list)-1):
        
        current_date = date_list[i]
        next_date = date_list[i+1]
        subset = df[df["Date"]==current_date]

        for index, row in subset.iterrows():
            date = current_date + pd.Timedelta(days=1)
            new_row = row.copy()
            while date < next_date:
                new_row["Date"] = date
                row_list.append(new_row.tolist())
                date += pd.Timedelta(days=1)
    
    new_df = pd.DataFrame(row_list, columns = df.columns)
    df = pd.concat([df,new_df]).sort_values(by=["STATE OF AFFECTED POPULATION","Date"]).reset_index().drop(columns=["index"])
    df.columns = ["date","location","ipd", "orig_state"]
    df["location"] = df["location"].str.lower()

    return df

# reads in an ACLED csv file and converts it into a dataframe
# containing dates, total fatalities on that day, and the number of events per day
def read_acled(path):
    with open(path, 'r') as acled:
        df = pd.read_csv(acled)
        df["event_date"] = pd.to_datetime(df["event_date"], format='%d %B %Y')
        df['event_date'] = df['event_date'].dt.strftime('%Y-%m-%d')

        df["event_count"] = 1
        # columns to do a combined search by
        cols_interest = ['event_date', 'admin1']

        # groups the dataframe by event date and sums their fatalities and events
        ret_df = df.groupby(cols_interest, as_index=False, axis=0).agg({"fatalities":"sum","event_count":"sum"})

    ret_df.columns = ["date","location","fatalities","event_count"]
    ret_df["date"] = pd.to_datetime(ret_df["date"])
    ret_df["location"] = df["location"].str.lower()

    return ret_df

# reads in emotion labelling data and consolidates it into a dataframe containing
# dates, location and emotion percentage for that date and location
def read_labels(file, name_pop_df):
    with open(file, "r") as f:
        df = pd.read_csv(f)
    # converts dataframe into just date, city, and emotion data
    df = df[["date","city"]+df.columns[6:].tolist()]

    # helper func to replace Arabic text w/ English
    def replace_arabic(row, trans_dict):
        city = row["city"]
        if city in trans_dict:
            city = trans_dict[city]
        return city

    # translating the Arabic city names w/ English
    city_translations = name_pop_df[["name_translated","name_en"]].set_index("name_translated")["name_en"].to_dict()
    df["city"] = df.apply(replace_arabic, axis=1, args=(city_translations,))
    
    # aggregates the emotion data in to be sum of emotion for that day and city
    agg_dict = {}
    for emotion in df.columns[2:]:
        agg_dict[emotion] = "sum"
    df = df.groupby(["date","city"], as_index=False, axis=0).agg(agg_dict)

    # for emotion in df.columns[2:]:
    #     df[emotion] = (df[emotion] - df[emotion].mean()) / df[emotion].std()

    df.rename(columns={"city":"location"}, inplace=True)
    df["location"] = df["location"].str.lower()
    df["date"] = pd.to_datetime(df["date"])

    return df

# merges the iom data first w/ the city population data
# and then w/ the data from the inputted organic datafile
def merge_df(iom_df, df2, org_cols, name_pop_df):
    # just Sudan city name and population
    name_pop_df = name_pop_df.loc[name_pop_df['country'] == 'Sudan']
 
    # merge iom w/ just intersecting cities and population
    interest_cols = ['name_en', 'population']
    name_pop_df = name_pop_df[interest_cols].dropna()
    name_pop_df = name_pop_df.rename(columns={'name_en': 'location'})
    pop_merge_df = pd.merge(iom_df, name_pop_df, on=['location'], how="outer")

    # merge base dataframe w/ organic data
    merged_df = pd.merge(pop_merge_df, df2, on=org_cols, how="outer")
    iom_locations = iom_df["location"].unique().tolist()
    merged_df = merged_df[merged_df["location"].isin(iom_locations)]
    merged_df = merged_df.fillna(0)
    
    return merged_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-acled", default='data/acled.csv')
    parser.add_argument("-iom", default='data/iom.csv')
    parser.add_argument("-labels", default="data/labels.csv")
    parser.add_argument("-namePop", default="data/names-and-populations.csv")
    args = parser.parse_args()
    
    name_pop_df = pd.read_csv(args.namePop)

    # consolidate iom and acled data into smaller dataframes
    iom_df = read_iom(args.iom)
    acled_df = read_acled(args.acled)
    label_df = read_labels(args.labels, name_pop_df)

    # merging iom data first w/ city populations and then the inputted
    # organic data
    acled_merged = merge_df(iom_df, acled_df, ["date", "location"], name_pop_df)
    label_merged = merge_df(iom_df, label_df, ["date", "location"], name_pop_df)

    # outputting to csv
    acled_merged.to_csv("acled_outfile.csv", index=False)
    label_merged.to_csv("label_outfile.csv", index=False)


if __name__ == "__main__":
    main()