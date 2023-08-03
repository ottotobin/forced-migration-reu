"""
authors: Toby, Rich, Lina, Grace

description:
    program consolidates data from ACLED, Google Trends, and emotion labeling 
    into a MultiIndex dataframe with Internally Displaced Person (IDP) data 
    from IOM

USAGE:
    python3 consol_data.py [-iom] [-acled] [-labels] [-namePop] [-gtrKey] [-gtrLoc]
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

    # List of columns that we will be using from the iom data
    # Some of the week-datasets did not have camp/neighborhood data,
    # so we are using locality as the smallest granularity
    cols_of_interest = ["Date","STATE OF AFFECTED POPULATION","# IDP INDIVIDUALS","STATE OF ORIGIN"]
    df = df[cols_of_interest]

    # Only interested in rows with non-zero IDP
    df["# IDP INDIVIDUALS"] = pd.to_numeric(df["# IDP INDIVIDUALS"].str.replace(",","").replace("-","0"))
    df = df[df["# IDP INDIVIDUALS"]>0]

    # Convert columns to appropriate data types for summing/grouping
    df["Date"] = pd.to_datetime(df["Date"])

    # Group rows by date, state, and locality and sum their IDP counts
    arrive_df = df.groupby(["Date","STATE OF AFFECTED POPULATION"], as_index=False, axis=0)["# IDP INDIVIDUALS"].sum()
    arrive_df.columns = ["Date","location","arriving_IDP"]
    exit_df = df.groupby(["Date","STATE OF ORIGIN"], as_index=False, axis=0)["# IDP INDIVIDUALS"].sum()
    exit_df.columns = ["Date","location","leaving_IDP"]
    df = pd.merge(arrive_df, exit_df, on=["Date","location"], how="outer").fillna(0)

    ss_df = df[df["location"]=="South Sudan"]
    df = df[df["location"]!="South Sudan"]
    date_list = df["Date"].unique()
    loc_list = df["location"].unique()
    
    row_list = []
    for loc in loc_list:

        subset = df[df["location"]==loc]
        subset = subset.sort_values(by="Date", ascending=True)
        date_vals = subset["Date"].unique()
        date = date_vals[0]
        last_date = date_vals[-1]

        last_val_idp_arriving = 0
        last_val_idp_leaving = 0

        while date <= last_date:
            if date in date_vals:
                last_val_idp_arriving = subset[subset["Date"]==date]["arriving_IDP"].iloc[0]
                last_val_idp_leaving = subset[subset["Date"]==date]["leaving_IDP"].iloc[0]
            else:
                row_list.append([date, loc, last_val_idp_arriving, last_val_idp_leaving])
            
            date += pd.Timedelta(days=1)

    new_df = pd.DataFrame(row_list, columns = df.columns)
    df = pd.concat([df,new_df,ss_df]).sort_values(by=["Date", "location"]).reset_index().drop(columns=["index"])
    df.columns = ["date","location","arriving_IDP", "leaving_IDP"]
    df["location"] = df["location"].str.lower()

    df.to_csv("output/iom_outfile.csv", index=False)

    return df

SS_CITIES = []
def get_SS_cities(filename):
    global SS_CITIES
    with open(filename, "r") as f:
        df = pd.read_csv(f)
        df = df[df["country"]=="South Sudan"]
        SS_CITIES = df["name_en"].tolist() + df["name_translated"].tolist()
        SS_CITIES = [x.lower() for x in SS_CITIES]
def replace_SouthSudan_cities(city):
    if city in SS_CITIES:
        return "south sudan"
    else:
        return city
def replace_hyphen(city):
    return city.split("-")[0].rstrip()

# reads in an ACLED csv file and converts it into a dataframe
# containing dates, total fatalities on that day, and the number of events per day
def read_acled(path):
    with open(path, 'r') as acled:
        df = pd.read_csv(acled)

    df["event_date"] = pd.to_datetime(df["event_date"], format='%d %B %Y')
    df['event_date'] = df['event_date'].dt.strftime('%Y-%m-%d')
    df["event_count"] = 1

    df["admin1"] = df["admin1"].apply(replace_SouthSudan_cities).apply(replace_hyphen)

    # columns to do a combined search by
    cols_interest = ['event_date', 'admin1']
    # groups the dataframe by event date and sums their fatalities and events
    df = df.groupby(cols_interest, as_index=False, axis=0).agg({"fatalities":"sum","event_count":"sum"})

    df.columns = ["date","location","fatalities","event_count"]
    df["date"] = pd.to_datetime(df["date"])
    df["location"] = df["location"].str.lower()

    df.to_csv("output/acled_outfile.csv", index=False)

    return df

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
    

    df["city"] = df["city"].apply(replace_SouthSudan_cities).apply(replace_hyphen)

    df = df.groupby(["date","city"], as_index=False, axis=0).agg(agg_dict)

    # for emotion in df.columns[2:]:
    #     df[emotion] = (df[emotion] - df[emotion].mean()) / df[emotion].std()

    df.rename(columns={"city":"location"}, inplace=True)
    df["location"] = df["location"].str.lower()
    df["date"] = pd.to_datetime(df["date"])

    df.to_csv("output/label_outfile.csv", index=False)

    return df

def read_gtrKey(file):
    with open(file, "r") as f:
        df = pd.read_csv(f)
    
    df.columns = ["date","location","topic","search_count"]
    df["date"] = pd.to_datetime(df["date"])
    df["location"] = df["location"].str.lower()
    df = df.pivot(index=["date","location"], columns = "topic", values="search_count").reset_index()
    df = df.fillna(0)

    df.to_csv("output/gtrKey_outfile.csv", index=False)

    df["location"] = df["location"].apply(replace_SouthSudan_cities).apply(replace_hyphen)

    return df

def read_gtrLoc(file):
    with open(file, "r") as f:
        df = pd.read_csv(f)
    df["date"] = pd.to_datetime(df["date"])
    df["location"] = df["location"].str.lower()

    df["location"] = df["location"].apply(replace_SouthSudan_cities).apply(replace_hyphen)
    
    df.to_csv("output/gtrLoc_outfile.csv", index=False)

    return df

# merges the iom data first w/ the city population data
# and then w/ the data from the inputted organic datafile
def merge_df(iom_df, df2, org_cols, other_df):

    # merge base dataframe w/ organic data
    merged_df = pd.merge(iom_df, df2, on=org_cols, how="outer")
    
    # trim merged_df to contain only IOM locations
    iom_locations = iom_df['location'].unique().tolist()
    merged_df = merged_df[merged_df['location'].isin(iom_locations)]
    merged_df = merged_df.fillna(0)

    # change to Multiindex dataframe
    merged_df = merged_df.set_index(['date', 'location'])

    merged_df.to_csv(f"output/M_iom_{other_df}.csv")
    return merged_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-acled", default='data/acled.csv')
    parser.add_argument("-iom", default='data/iom.csv')
    parser.add_argument("-labels", default="data/labels.csv")
    parser.add_argument("-namePop", default="data/names-and-populations.csv")
    parser.add_argument("-gtrKey", default="data/gtr_SD_keyword_data.csv")
    parser.add_argument("-gtrLoc", default="data/gtr_location_data.csv")

    args = parser.parse_args()
    
    name_pop_df = pd.read_csv(args.namePop)
    get_SS_cities("data/names-and-populations.csv")

    # consolidate iom and acled data into smaller dataframes
    iom_df = read_iom(args.iom)
    acled_df = read_acled(args.acled)
    label_df = read_labels(args.labels, name_pop_df)
    gtrKey_df = read_gtrKey(args.gtrKey)
    gtrLoc_df = read_gtrLoc(args.gtrLoc)

    # merging iom data first w/ city populations and then the inputted
    # organic data
    acled_merged = merge_df(iom_df, acled_df, ["date", "location"], "acled")
    label_merged = merge_df(iom_df, label_df, ["date", "location"], "label")
    gtrKey_merged = merge_df(iom_df, gtrKey_df, ["date", "location"], "gtrKey")
    gtrLoc_merged = merge_df(iom_df, gtrLoc_df, ["date", "location"], "gtrLoc")


if __name__ == "__main__":
    main()