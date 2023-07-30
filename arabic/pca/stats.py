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
import matplotlib.dates as mdates
import numpy as np

# creates raw and normalized time series for the consolidated files 
# of IOM and organic data
def time_series(file, location=None):
    org_type = file.split('_')[2].split('.')[0]
    trends_df = pd.read_csv(file).set_index(['date', 'location'])

    regions = list(set(trends_df.index.get_level_values('location')))
    days = list(set(trends_df.index.get_level_values('date')))
    vars = trends_df.columns[0:]

    # time series of organic data vs IOM idp
    if location != None:
        r = location
        for v in vars:
            plt.plot(trends_df.xs(r,level=1).loc[:,v], label = v)
        plt.legend()
        plt.savefig(f"visuals/{org_type}/{r}_unnorm.pdf")
        plt.close()


        # normalize data
        norm_df = trends_df.copy()
        for v in vars:
            norm_df[v] = (norm_df[v] - np.mean(norm_df[v])) / np.std(norm_df[v])

        for v in vars:
            plt.plot(norm_df.xs(r,level=1).loc[:,v], label = v)
        ax = plt.gca()
        # ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        plt.legend()
        plt.savefig(f"visuals/{org_type}/{r}_norm.pdf")
        plt.close()

        return trends_df
    else:
        for r in regions:
            for v in vars:
                plt.plot(trends_df.xs(r,level=1).loc[:,v], label = v)
            plt.legend()
            plt.savefig(f"visuals/{org_type}/{r}_unnorm.pdf")
            plt.close()

        # normalize data
        norm_df = trends_df.copy()
        for v in vars:
            norm_df[v] = (norm_df[v] - np.mean(norm_df[v])) / np.std(norm_df[v])

        # Plot normalized data.
        for r in regions:
            for v in vars:
                plt.plot(norm_df.xs(r,level=1).loc[:,v], label = v)
            ax = plt.gca()
            # ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
            plt.legend()
            plt.savefig(f"visuals/{org_type}/{r}_norm.pdf")
            plt.close()

        return trends_df

# Calculate and viz correlation
def corr_matrix(trends_df, file, location=None):
    org_type = file.split('_')[2].split('.')[0]

    # extract unique, days, and variables from the dataframe
    regions = list(set(trends_df.index.get_level_values('location')))
    days = list(set(trends_df.index.get_level_values('date')))
    vars = trends_df.columns[0:]

    # create a correlation matrix for each region
    if location != None:
        r = location
        corr = np.zeros([len(vars),len(vars)])
        for i,v1 in enumerate(vars):
            for j,v2 in enumerate(vars):
                corr[i,j] = np.corrcoef(trends_df[v1], trends_df[v2])[0,1]

        # fig = plt.figure()
        plt.imshow(corr, cmap = plt.get_cmap('Greens'), vmin = 0, vmax = 1)
        plt.colorbar()
        ax = plt.gca()
        #ax.set_xticklabels(vars)
        ax.set_yticks(np.arange(len(vars)))
        ax.set_yticklabels(vars)
        ax.set_xticks(np.arange(len(vars)))
        ax.set_xticklabels(vars)
        plt.xticks(rotation=90)
        #ax.xaxis.set_xticks(vars)
        plt.tight_layout()
        plt.savefig(f"visuals/{org_type}/{r}_cov_im.pdf")
        plt.close()

    else:
        for r in regions:
            corr = np.zeros([len(vars),len(vars)])
            for i,v1 in enumerate(vars):
                for j,v2 in enumerate(vars):
                    corr[i,j] = np.corrcoef(trends_df[v1], trends_df[v2])[0,1]

            # fig = plt.figure()
            plt.imshow(corr, cmap = plt.get_cmap('magma'), vmin = 0, vmax = 1)
            plt.colorbar()
            ax = plt.gca()
            #ax.set_xticklabels(vars)
            ax.set_yticks(np.arange(len(vars)))
            ax.set_yticklabels(vars)
            ax.set_xticks(np.arange(len(vars)))
            ax.set_xticklabels(vars)
            plt.xticks(rotation=90)
            #ax.xaxis.set_xticks(vars)
            plt.tight_layout()
            plt.savefig(f"visuals/{org_type}/{r}_cov_im.pdf")
            plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-location", required=False)
    parser.add_argument("-acled", default='output/M_iom_acled.csv')
    parser.add_argument("-label", default="output/M_iom_label.csv")
    parser.add_argument("-keyword", default="output/M_iom_gtrKey.csv")
    parser.add_argument("-loc", default="output/M_iom_gtrLoc.csv")
    args = parser.parse_args()

    datafiles = [arg for arg in args.__dict__.values()][1:]

    print(args.location)

    if args.location != None:
        for file in datafiles:
            trends_df = time_series(file, location=args.location)
            corr_matrix(trends_df, file, location=args.location)
    else: 
        for file in datafiles:
            trends_df = time_series(file, location=args.location)
            corr_matrix(trends_df, file, location=args.location)


if __name__ == "__main__":
    main()
