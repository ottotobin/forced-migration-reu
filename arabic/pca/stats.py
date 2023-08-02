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


    #     # fig = plt.figure()
    #     plt.imshow(corr, cmap = plt.get_cmap('Greens'), vmin = 0, vmax = 1)
    #     plt.colorbar()
    #     ax = plt.gca()
    #     #ax.set_xticklabels(vars)
    #     ax.set_yticks(np.arange(len(vars)))
    #     ax.set_yticklabels(vars)
    #     ax.set_xticks(np.arange(len(vars)))
    #     ax.set_xticklabels(vars)
    #     plt.xticks(rotation=90)
    #     #ax.xaxis.set_xticks(vars)
    #     plt.tight_layout()
    #     plt.savefig(f"visuals/{org_type}/{r}_cov_im.pdf")
    #     plt.close()

    # else:
    #     for r in regions:
    #         corr = np.zeros([len(vars),len(vars)])
    #         for i,v1 in enumerate(vars):
    #             for j,v2 in enumerate(vars):
    #                 corr[i,j] = np.corrcoef(trends_df[v1], trends_df[v2])[0,1]

    #         # fig = plt.figure()
    #         plt.imshow(corr, cmap = plt.get_cmap('magma'), vmin = 0, vmax = 1)
    #         plt.colorbar()
    #         ax = plt.gca()
    #         #ax.set_xticklabels(vars)
    #         ax.set_yticks(np.arange(len(vars)))
    #         ax.set_yticklabels(vars)
    #         ax.set_xticks(np.arange(len(vars)))
    #         ax.set_xticklabels(vars)
    #         plt.xticks(rotation=90)
    #         #ax.xaxis.set_xticks(vars)
    #         plt.tight_layout()
    #         plt.savefig(f"visuals/{org_type}/{r}_cov_im.pdf")
    #         plt.close()

    return corr


## Compute PCA
def pca(datafiles, location):
    trends_df = pd.read_csv(datafiles[0])
    trends_df = trends_df.set_index(['date', 'location'])
    vars = trends_df.columns[0:]
    corr = corr_matrix(trends_df, datafiles[0], location)
    
    norm_df = trends_df.copy()
    for v in vars:
        norm_df[v] = (norm_df[v] - np.mean(norm_df[v])) / np.std(norm_df[v])
        
    #corr - 
    #norm_df.T @ norm_df / norm_df.shape[0]
    #np.linalg.svd(norm_df)[2][0,:]
    #np.mean(norm_df)

    ed = np.linalg.eigh(corr)

    # ed[1] @ np.diag(ed[0]) @ ed[1].T

    ## ed[0] - EIGENVALUES: how important is each combination?
    ## ed[1] - EIGENVECTORS: what is the composition of each combination

    ## Eigenvectors == Principal Components

    ed[0]

    # First two principal components
    ev1 = pd.Series(ed[1][:,-1], index = vars)
    ev2 = pd.Series(ed[1][:,-2], index = vars)

    norm_df.iloc[0,:]

    # low dimensional plot of each day/region
    V = np.stack([-ev1,-ev2]).T
    low_d_df = norm_df @ V

    # # Relative importance of principal components (given by eigenvalues)
    # fig = plt.figure()
    # plt.scatter(np.arange(len(vars)), np.flip(ed[0]))
    # plt.ylabel('Eigenvalue')
    # ax = plt.gca()
    # ax1 = ax.twinx()
    # ax1.plot(np.cumsum(np.flip(ed[0]))/np.sum(ed[0]), linestyle = '--')
    # ax1.set_ylim(0,1)
    # ax1.set_ylabel('Cumulative Proportion')
    # plt.savefig("evals.pdf")
    # plt.close()

    # ck = {'Center' : 'Greens', 'South' : 'Reds', 'North' : 'Blues', 'East' : 'Purples', 'Kyiv' : 'Oranges','West':'Greys'}

    # fig = plt.figure()
    # for r in regions:
    #     cmap = plt.get_cmap(ck[r])
    #     Xr = low_d_df.xs(r,level=1)
    #     col_inds = np.flip(0.2+0.6*np.arange(Xr.shape[0]) / Xr.shape[0])
    #     cols = cmap(col_inds)
    #     plt.scatter(Xr.loc[:,0], Xr.loc[:,1], label = r, c = cols)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig("lowd.pdf")
    # plt.close()

    ## Look at variable contribution to top vectors
    contrib = np.square(ev1) + np.square(ev2)
    contrib.sort_values(ascending=False)

    # That was for K = 2, this is general K.
    # top_K = 8
    # contrib = pd.Series(np.sum(np.square(ed[1][:,-top_K:]), axis = 1), index = ev1.index)
    # contrib.sort_values(ascending=False)

    ### Combine flow with indicators to make Predictions
    with open("output/iom_outfile.csv", 'rb') as f:
        [dfo, dfd] = pickles.load(f)
        # [dfo, dfd] = 

    X = pd.DataFrame(np.zeros([dfo.shape[0], trends_df.shape[1]]))
    X.columns = trends_df.columns


    for i in range(dfo.shape[0]):

        #dfrom = dfo['date_from'][i] 
        #dto = dfo['date_to'][i] 

        dfrom = dfo['date_from'][i] + pd.Timedelta(days=offset)
        dto = dfo['date_to'][i] + pd.Timedelta(days=offset)

        in_date_range = trends_df.loc[(slice(dfrom,dto)),:]
        if in_date_range.shape[0] > 0:
            in_region_too = in_date_range.xs(dfo['Macro-region'][i], level = 1)
            X.iloc[i,:] = np.mean(in_region_too,axis=0)
        else:
            X.iloc[i,:] = np.repeat(np.nan, X.shape[1])

    missing = np.any(X.isna(), axis = 1)
    #dfo = dfo.loc[~missing,:]

    X = X.loc[~missing,:]
    y = dfo.loc[~missing,'Percent']

    import statsmodels.api as sm

    for v in X.columns:
        lmfit = sm.OLS(y, sm.add_constant(X.loc[:,v])).fit()
        lmfit.summary()
        r2 = lmfit.rsquared
        print(v)
        print(r2)
        np.square(np.corrcoef(y,X.loc[:,v]))

def regression(merged_file, multiple=False):
    with open(merged_file, "r") as f:
        df = pd.read_csv(f)

    for loc in df["location"].unique():
        sub_df = df[df["location"]==loc]
        iom_df = sub_df["date","location","arriving_IDP","leaving_IDP"]
        indicat_df = sub_df[[col for col in df.columns if col not in ["arriving_IDP","leaving_IDP"]]]

        for offset in range(-4, 4):
            
        

    
        


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-location", required=False)
    parser.add_argument("-acled", default='output/M_iom_acled.csv')
    parser.add_argument("-label", default="output/M_iom_label.csv")
    parser.add_argument("-keyword", default="output/M_iom_gtrKey.csv")
    parser.add_argument("-loc", default="output/M_iom_gtrLoc.csv")
    args = parser.parse_args()

    datafiles = [arg for arg in args.__dict__.values()][1:]

    pca(datafiles, args.location)
    exit()

    if args.location != None:
        for file in datafiles:
            trends_df = time_series(file, location=args.location)
            corr_matrix(trends_df, file, location=args.location)
    else: 
        for file in datafiles:
            trends_df = time_series(file)
            corr_matrix(trends_df, file)


if __name__ == "__main__":
    main()
