#########
# pca_range.py: conducting principle component analysis on the 
# monthly 5 year Google Trends data

# authors: Rich, Toby, Lina, Grace

# last modified: 7/13/23
# - created file

#########
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
import pandas as pd

# reads in data for each catergory for a given ISO code
def load_data(iso, categories):
    retDict = {}  

    for cat in categories:
        if os.path.exists(f"results/{cat}"):
            retDict[cat] = pd.read_csv(f"results/{cat}/{cat}_{iso}_2018-01-01-to-2023-05-31_adjusted.csv")

    return retDict

# aggregates all intensities of words in each dataframe
def aggregate(df_dict):
    retDict = {}
    for key in df_dict:
        df = df_dict[key]
        key_words = df.columns[3:]
        for column in key_words:
            if len(df[column].unique()) == 1 and df[column].unique()[0] == 0:
                df = df.drop(column, axis=1)

        df["aggregate"] = 0
        if len(df.columns) > 3:
            for column in df.columns[3:]:
                df["aggregate"] += df[column] / len(df.columns[3:])


        retDict[key] = df

    return retDict

# concatenates the dataframe into just dates and aggregates
# for each category
def concat_df(df_dict):
    columns_of_interest = ["date","aggregate"]
    retDf = pd.DataFrame()
    for key in df_dict:
        if retDf.empty:
            retDf = df_dict[key][columns_of_interest]
            retDf = retDf.rename(columns={"aggregate":key})
        else:
            retDf[key] = df_dict[key]["aggregate"]

    return retDf

def time_series(df, iso):
    # plot unnormalized time series
    r = iso
    fig = plt.figure()
    df.plot(x="date", y=df.columns[1:])
    plt.legend()
    plt.savefig(f"{r}_unnormed.pdf")
    plt.close()

    # normalize aggregates
    norm_df = df.copy()
    for column in df.columns[1:]:
        norm_df[column] = (norm_df[column] - np.mean(norm_df[column])) / np.std(norm_df[column])
    # df["aggregate"] = (df["aggregate"] - np.mean(df["aggregate"])) / np.std(df["aggregate"])

    # plot normalized time series
    fig = plt.figure()
    norm_df.plot(x="date", y=df.columns[1:])
    # ax = plt.gca()
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    plt.legend()
    plt.savefig(f"{r}_norm.pdf")
    plt.close()

    return norm_df

# Calculate and viz correlation
def corr_matrix(df):    
    vars = df.columns[1:]
    corr = np.zeros([len(vars),len(vars)])
    for i,v1 in enumerate(vars):
        for j,v2 in enumerate(vars):
            # print(df[v1], df[v2])
            corr[i,j] = np.corrcoef(df[v1], df[v2])[0,1]

    fig = plt.figure()
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
    plt.savefig("cov_im.pdf")
    plt.close()

    return corr

## Compute PCA
def pca(corr, norm_df):
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

    # Relative importance of principal components (given by eigenvalues)
    fig = plt.figure()
    plt.scatter(np.arange(len(vars)), np.flip(ed[0]))
    ax = plt.gca()
    ax1 = ax.twinx()
    ax1.plot(np.cumsum(np.flip(ed[0]))/np.sum(ed[0]))
    ax1.set_ylim(0,1)
    plt.savefig("evals.pdf")
    plt.close()

    ck = {'Center' : 'Greens', 'South' : 'Reds', 'North' : 'Blues', 'East' : 'Purples', 'Kyiv' : 'Oranges','West':'Greys'}

    fig = plt.figure()
    for r in regions:
        cmap = plt.get_cmap(ck[r])
        Xr = low_d_df.xs(r,level=1)
        col_inds = np.flip(0.2+0.6*np.arange(Xr.shape[0]) / Xr.shape[0])
        cols = cmap(col_inds)
        plt.scatter(Xr.loc[:,0], Xr.loc[:,1], label = r, c = cols)
    plt.legend()
    plt.tight_layout()
    plt.savefig("lowd.pdf")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Google Trend PCA")
    parser.add_argument("--iso", required=True, help="ISO code for country PCA is applied to")
    args = parser.parse_args()

    iso = str(args.iso)
    categories = ["generic_terms","geography","safe_places","travel"]

    df = concat_df(aggregate(load_data(iso, categories)))

    norm_df = time_series(df, iso)
    corr = corr_matrix(df)
    # pca(corr, norm_df)
    
if __name__ == '__main__':
    main()