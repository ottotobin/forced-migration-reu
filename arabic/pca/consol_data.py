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

    df["# IDP INDIVIDUALS"] = pd.to_numeric(df["# IDP INDIVIDUALS"].str.replace(",","").replace("-","0"))
    df = df[df["# IDP INDIVIDUALS"]>0]

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
    cols_of_interest = ["Date","STATE OF AFFECTED POPULATION"]
    
    # Convert columns to appropriate data types for summing/grouping
    df["Date"] = pd.to_datetime(df["Date"])

    # Drop missing values
    df = df.dropna()

    # Group rows by date, state, and locality and sum their IDP counts
    arrive_df = df.groupby(["Date","STATE OF AFFECTED POPULATION"], as_index=False, axis=0)["# IDP INDIVIDUALS"].sum()
    arrive_df.columns = ["Date","location","arriving_IDP"]
    exit_df = df.groupby(["Date","STATE OF ORIGIN"], as_index=False, axis=0)["# IDP INDIVIDUALS"].sum()
    exit_df.columns = ["Date","location","leaving_IDP"]
    df = pd.merge(arrive_df, exit_df, on=["Date","location"], how="outer").fillna(0)

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
    df = pd.concat([df,new_df]).sort_values(by=["Date", "location"]).reset_index().drop(columns=["index"])
    df.columns = ["date","location","arriving_IDP", "leaving_IDP"]
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

def read_gtrKey(file):
    with open(file, "r") as f:
        df = pd.read_csv(f)
    
    df.columns = ["date","location","topic","search_count"]
    df["date"] = pd.to_datetime(df["date"])
    df["location"] = df["location"].str.lower()
    df = df.pivot(index=["date","location"], columns = "topic", values="search_count").reset_index()
    df = df.fillna(0)

    return df

def read_gtrLoc(file):
    with open(file, "r") as f:
        df = pd.read_csv(f)
    df["date"] = pd.to_datetime(df["date"])
    df["location"] = df["location"].str.lower()
    df.to_csv("o.csv", index=False)

    return df
    

# merges the iom data first w/ the city population data
# and then w/ the data from the inputted organic datafile
def merge_df(iom_df, df2, org_cols, name_pop_df):

    # merge base dataframe w/ organic data
    merged_df = pd.merge(iom_df, df2, on=org_cols, how="outer")
    merged_df.to_csv("o2.csv",index=False)
   
    # trim merged_df to contain only IOM locations
    iom_locations = iom_df['location'].unique().tolist()
    merged_df = merged_df[merged_df['location'].isin(iom_locations)]
    merged_df = merged_df.fillna(0)


    # name_pop_df = name_pop_df[name_pop_df['country'] == 'Sudan']
    # name_pop_dict = dict(zip(name_pop_df["name_en"].str.lower(),name_pop_df["population"]))
    # def fill_pop(name):
    #     if len(name.split()) == 1 and name in name_pop_dict:
    #         return name_pop_dict[name]
    #     else:
    #         for key in name_pop_dict:
    #             if all(word in key for word in name.split()):
    #                 return name_pop_dict[name]
    #         return 0
    # merged_df["population"] = merged_df["location"].apply(fill_pop)

    # change to Multiindex dataframe
    merged_df = merged_df.set_index(['date', 'location'])
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

    # consolidate iom and acled data into smaller dataframes
    iom_df = read_iom(args.iom)
    acled_df = read_acled(args.acled)
    label_df = read_labels(args.labels, name_pop_df)
    gtrKey_df = read_gtrKey(args.gtrKey)
    gtrLoc_df = read_gtrLoc(args.gtrLoc)

    # merging iom data first w/ city populations and then the inputted
    # organic data
    acled_merged = merge_df(iom_df, acled_df, ["date", "location"], name_pop_df)
    label_merged = merge_df(iom_df, label_df, ["date", "location"], name_pop_df)
    gtrKey_merged = merge_df(iom_df, gtrKey_df, ["date", "location"], name_pop_df)
    gtrLoc_merged = merge_df(iom_df, gtrLoc_df, ["date", "location"], name_pop_df)

    # outputting to csv
    acled_merged.to_csv("output/acled_outfile.csv", index=False)
    label_merged.to_csv("output/label_outfile.csv", index=False)
    gtrKey_merged.to_csv("output/gtrKey_outfile.csv", index=False)
    gtrLoc_merged.to_csv("output/gtrLoc_outfile.csv", index=False)


if __name__ == "__main__":
    main()