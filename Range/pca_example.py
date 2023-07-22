import matplotlib.pyplot as plt
import pickle
import numpy as np
import matplotlib.dates as mdates
import pandas as pd

## TODO: Load in your data here.
#with open("./pickles/reg_trends.pkl",'rb') as f:
#    trends_df = pickle.load(f)

regions = list(set(trends_df.index.get_level_values('macro')))
days = list(set(trends_df.index.get_level_values('date')))
vars = trends_df.columns

## Plot to start out with.
r = 'East'
fig = plt.figure()
for v in vars:
    plt.plot(trends_df.xs(r,level=1).loc[:,v], label = v)
plt.legend()
plt.savefig("trends_unnorm.pdf")
plt.close()

## Normalize overall
norm_df = trends_df.copy()
for v in vars:
   norm_df[v] = (norm_df[v] - np.mean(norm_df[v])) / np.std(norm_df[v])

#norm_df = trends_df.copy()
#for v in vars:
#   norm_df[v] = (norm_df[v] - np.min(norm_df[v])) /  (np.max(norm_df[v]) - np.min(norm_df[v]))

### Normalize by region
#norm_df = trends_df.copy()
### Normalize by region
#for v in vars:
#    for r in regions:
#        norm_df.loc[(days,r),v] = (norm_df.loc[(days,r),v] - np.mean(norm_df.loc[(days,r),v])) / np.std(norm_df.loc[(days,r),v])

## Plot normalized data.
r = 'East'
fig = plt.figure()
for v in vars:
    plt.plot(norm_df.xs(r,level=1).loc[:,v], label = v)
ax = plt.gca()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
plt.legend()
plt.savefig("trends_norm.pdf")
plt.close()

## Calculate and viz correlation
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
plt.savefig("cov_im.pdf")
plt.close()

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
