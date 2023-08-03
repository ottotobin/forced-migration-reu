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
from tqdm import tqdm

def regression(merged, week_offset, multiple=False):
    if type(merged) == str:
        with open(merged, "r") as f:
            df = pd.read_csv(f)
            df["date"] = pd.to_datetime(df["date"])
    else:
        df = merged

    row_list = []
    for loc in tqdm(df["location"].unique(), desc="locations"):
        sub_df = df[df["location"]==loc]
        
        IOM_df = sub_df[["date","location","arriving_IDP","leaving_IDP"]]
        INDIC_df = sub_df[[col for col in df.columns if col not in ["arriving_IDP","leaving_IDP"]]].copy()
        offset_list = list(range(-week_offset, week_offset))

        if multiple:
            offsets = []
            non_iom_cols = [col for col in INDIC_df.columns if col not in IOM_df.columns]
            for i in range(len(non_iom_cols)):
                offsets.append(offset_list.copy()) 
            offset_list = list(itertools.product(*offsets))

        for offset in tqdm(offset_list, desc="offsets"):
            if multiple:
                indic_dfs = []
                for col in non_iom_cols:
                    indic_dfs.append(INDIC_df[["date","location", col]].copy())
                indic_df = pd.DataFrame()
                for i in range(len(offset)):
                    indic_dfs[i]["date"] += pd.Timedelta(days = offset[i])
                    if indic_df.empty:
                        indic_df = indic_dfs[i]
                    else:
                        indic_df = pd.merge(indic_df, indic_dfs[i], on=["date","location"], how="inner")
            else:
                indic_df = INDIC_df.copy()
                indic_df["date"] += pd.Timedelta(days = offset)
            
            merged_df = pd.merge(IOM_df, indic_df, on=["date", "location"], how="inner")
            merged_df.dropna()

            agg_dict = {}
            for col in indic_df.columns:
                if col not in IOM_df.columns:
                    agg_dict[col] = "sum"
            agg_dict["date"] = agg_dict["location"] = "first"
            agg_df = merged_df.groupby(["arriving_IDP","leaving_IDP"], as_index=False, axis=0).agg(agg_dict)
            
            y_list = [col for col in IOM_df.columns if col not in indic_df.columns]
            x_list = [col for col in indic_df.columns if col not in IOM_df.columns]

            for y in y_list:
                Y = agg_df[y]
                if not (Y == 0.0).all():
                    if multiple:
                        X = agg_df[x_list]
                        lmfit = sm.OLS(Y, sm.add_constant(X)).fit()
                        r2 = lmfit.rsquared
                        if r2 >=0 and r2 <= 1:
                            row_list.append([loc, y, x_list, len(x_list), r2, list(offset)])
                    else:
                        for x in x_list:
                            X = agg_df[x]
                            if not (X == 0.0).all():
                                lmfit = sm.OLS(Y, sm.add_constant(X)).fit()
                                r2 = lmfit.rsquared
                                if r2 >=0 and r2 <= 1:
                                    row_list.append([loc, y, x, 1, r2, offset])
                # except ValueError:
                #     pass
    
    df_out = pd.DataFrame(row_list, columns = ["location","Y","X(s)","covariate num.","r^2","offset"]).dropna()
    
    if type(merged) == str:
        df_out.to_csv(f"output/corr_iom_{merged.split('.')[0].split('_')[-1]}.csv", index=False)
    
    return df_out

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
    for combo in tqdm(combos, desc="allstar configs"):
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
    exit()

    for file in datafiles:
        regression(datafiles[file], 4)


if __name__ == "__main__":
    main()
