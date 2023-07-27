######
# authors: Lina, Rich, Toby, Grace

# last modified: 7/25/23

######
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
import argparse

def time_series(file, location):
    with open(file, 'r') as f:
        trends_df = pd.read_csv(f)
        local_df = trends_df.copy().loc[trends_df['location'] == location]
        local_df['date'] = sorted(local_df['date'])

    regions = list(trends_df['location'].unique())
    days = sorted(list(trends_df['date']))
    vars = trends_df.columns[5:]

    ## Plot to start out with.
    local_df.plot(x='date', y=vars)
    plt.legend()
    plt.savefig(f"{location}_unnorm.pdf")
    plt.close()

    ## Normalize overall
    norm_df = local_df.copy()
    for v in vars:
        norm_df[v] = (norm_df[v] - np.mean(norm_df[v])) / np.std(norm_df[v])
    
    norm_df.plot(x='date', y=vars)
    ax = plt.gca()
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    plt.legend()
    plt.savefig(f"{location}_norm.pdf")
    plt.close()

    return norm_df

def corr_matrix(trends_df, location):
    ## Calculate and viz correlation
    vars = trends_df.columns[5:]
    corr = np.zeros([len(vars),len(vars)])
    for i,v1 in enumerate(vars):
        for j,v2 in enumerate(vars):
            corr[i,j] = np.corrcoef(trends_df[v1], trends_df[v2])[0,1]

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
    plt.savefig(f"{location}_cov_im.pdf")
    plt.close()

    return corr


def pca(corr, norm_df):
    ## Compute PCA

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

    # Relative importance of principal components (given by eigenvalues)
    fig = plt.figure()
    plt.scatter(np.arange(len(vars)), np.flip(ed[0]))
    plt.ylabel('Eigenvalue')
    ax = plt.gca()
    ax1 = ax.twinx()
    ax1.plot(np.cumsum(np.flip(ed[0]))/np.sum(ed[0]), linestyle = '--')
    ax1.set_ylim(0,1)
    ax1.set_ylabel('Cumulative Proportion')
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

def cur(corr_mat):
    ## Look at variable contribution to top vectors
    contrib = np.square(ev1) + np.square(ev2)
    contrib.sort_values(ascending=False)

    # That was for K = 2, this is general K.
    top_K = 8
    contrib = pd.Series(np.sum(np.square(ed[1][:,-top_K:]), axis = 1), index = ev1.index)
    contrib.sort_values(ascending=False)

    ### Combine flow with indicators to make Predictions
    with open("pickles/iom.pkl", 'rb') as f:

        [dfo, dfd] = pickle.load(f)

    X = pd.DataFrame(np.zeros([dfo.shape[0], trends_df.shape[1]]))
    X.columns = trends_df.columns

    offset = 1
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-location", required=True)
    parser.add_argument("-acled", default='output/acled_outfile.csv')
    parser.add_argument("-label", default="data/label_outfile.csv")
    args = parser.parse_args()

    norm_df = time_series(args.acled, args.location)

    corr = corr_matrix(norm_df, args.location)

    pca(corr, norm_df)



    

if __name__ == "__main__":
    main()