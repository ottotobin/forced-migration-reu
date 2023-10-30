from cmd import Cmd
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

df = pd.read_csv('/Users/bernardmedeiros/Desktop/ToRun/FinalData/FinalFinal/monthly_aggregated_data_all.csv', lineterminator='\n')

grouped1 = df.groupby(['country']).get_group('Colombia').reset_index(drop=True)
grouped2 = df.groupby(['country']).get_group('Brazil').reset_index(drop=True)

#df_flow = pd.concat([grouped1, grouped2], axis=0)

def corrMat(covars, responses, df):
    corrMat = pd.DataFrame(columns=covars, index=responses)
    for c2 in covars:
        for c1 in responses:
            corrMat.at[c1, c2] = df[c2].corr(df[c1])
    return corrMat

responses = [12, 13, 14, 15] # EDIT ME
covars = [18] # EDIT ME

cvs = {i:var for i, var in enumerate(list(grouped1.columns[2:]))}

response_cols = [cvs[i] for i in responses]
covar_cols = [cvs[i] for i in covars]

cm = corrMat(covar_cols, response_cols, grouped1)

plt.figure(figsize=(8, 6))
sns.heatmap(cm.astype(float), annot=False, cmap='coolwarm', vmin=-1, vmax=1)
#cm["location"] = "all"
plt.show() 

row_n = len(response_cols)
fig, axes = plt.subplots(nrows=row_n, ncols=1, figsize=(4, row_n*3))
for i, r in enumerate(response_cols):
    #response = scaler.fit_transform(grouped1[r].values.reshape(-1, 1))
    ax = axes[i] if row_n > 1 else axes
    ax.plot(grouped1["month"], grouped1[r],label = r,linestyle='--')
    ax.plot(grouped2["month"], grouped2[r],label = r,linestyle='--')
    for cv in covar_cols:
        ax.plot(grouped1["month"], grouped1[cv], label=cv)
        ax.plot(grouped2["month"], grouped2[cv], label=cv)
    ax.set_xlabel("Date")
    ax.set_ylabel(r)
    ax.set_xticks(grouped1["month"])
    ax.set_xticklabels(grouped1["month"], rotation=45)
    ax.set_title(f"average {r} and covariates over time")
    ax.legend(loc="upper right")
plt.tight_layout()
plt.show()

