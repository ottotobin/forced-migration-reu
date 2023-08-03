"""
authors: Toby, Rich, Lina, Grace

description:
    program reads in csv files consolidated w/ organic data (ACLED, Google Trends, emotion labeling)
    and IOM IDP data.
    creates unnormalized and normalized time series of the data and then creates a correlation matrix
    between the different variables in the organic data.
    finally, visualizes this w/ plots for the time series and a heat map for the correlation matrix

USAGE:
    python3 stats.py
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np
import itertools
import warnings
import statsmodels.api as sm

from math import floor

def regression(merged, week_offset, multiple=False):
    if type(merged) == str:
        with open(merged, "r") as f:
            df = pd.read_csv(f)
            df["date"] = pd.to_datetime(df["date"])
    else:
        df = merged

    row_list = []
    for loc in df["location"].unique():
        sub_df = df[df["location"]==loc]

        weeks = floor((sub_df["date"].unique()[-1] - sub_df["date"].unique()[0]).days / 7)

        iom_df = sub_df[["date","location","arriving_IDP","leaving_IDP"]]
        indic_df = sub_df[[col for col in df.columns if col not in ["arriving_IDP","leaving_IDP"]]].copy()

        for offset in range(-week_offset, week_offset):
            indic_df["date"] += pd.Timedelta(weeks = offset)
            merged_df = pd.merge(iom_df, indic_df, on=["date", "location"], how="inner")
            merged_df.dropna()

            agg_dict = {}
            for col in indic_df.columns:
                if col not in iom_df.columns:
                    agg_dict[col] = "sum"
            agg_dict["date"] = agg_dict["location"] = "first"
            agg_df = merged_df.groupby(["arriving_IDP","leaving_IDP"], as_index=False, axis=0).agg(agg_dict)
            
            y_list = [col for col in iom_df.columns if col not in indic_df.columns]
            x_list = [col for col in indic_df.columns if col not in iom_df.columns]

            for y in y_list:
                Y = agg_df[y]

                try:
                    if multiple:
                        X = agg_df[x_list]
                        lmfit = sm.OLS(Y, sm.add_constant(X)).fit()
                        r2 = lmfit.rsquared
                        row_list.append([loc, y, x_list, len(x_list), r2, offset])
                    else:
                        for x in x_list:
                            X = agg_df[x]
                            lmfit = sm.OLS(Y, sm.add_constant(X)).fit()
                            r2 = lmfit.rsquared
                            row_list.append([loc, y, x, 1, r2, offset])
                except ValueError:
                    pass
       
    df_out = pd.DataFrame(row_list, columns = ["location","Y","X(s)","covariate num.","r^2","offset"])
    
    #df_out.to_csv(f"output/corr_iom_{merged_file.split('.')[0].split('_')[-1]}.csv", index=False)
    return df_out.dropna()

def allstar_model(datafiles, week_offset):
    df = pd.DataFrame()
    col_dict = {}
    for key in datafiles:
        with open(datafiles[key], "r") as f:
            add_df = pd.read_csv(f)
            col_dict[key] = add_df.columns[4:].tolist()
            if df.empty:
                df = add_df
            else:
                df = pd.merge(df, add_df, how="inner")
    
    df["date"] = pd.to_datetime(df["date"])
    df_out = pd.DataFrame()
    combos = list(itertools.product(*col_dict.values()))
    for combo in combos:
        reg_df = df[df.columns[:4].tolist()+list(combo)]
        if df_out.empty:
            df_out = regression(reg_df, week_offset, multiple=True)
        else:
            df_out = pd.concat([df_out, regression(reg_df, week_offset, multiple=True)])
        
    df_out.to_csv("output/full_corr.csv", index=False)
    return df_out

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("-location", required=False)
    parser.add_argument("-acled", default='output/M_iom_acled.csv')
    parser.add_argument("-label", default="output/M_iom_label.csv")
    parser.add_argument("-keyword", default="output/M_iom_gtrKey.csv")
    parser.add_argument("-loc", default="output/M_iom_gtrLoc.csv")
    args = parser.parse_args()

    datafiles = args.__dict__
    
    allstar_model(datafiles, 4)

    # for file in datafiles:''
    #     regression(file)
    

if __name__ == "__main__":
    main()
