import pandas as pd

models = ["bert2","GloVe","muse"]

bigDf = pd.DataFrame()
for m in models:
    with open(f"{m}.csv","r") as f:
        df = pd.read_csv(f)
    
    if bigDf.empty:
        bigDf = df
    else:
        bigDf = pd.concat([bigDf, df], ignore_index=True)
    
bigDf.to_csv("test.csv", index = False)

