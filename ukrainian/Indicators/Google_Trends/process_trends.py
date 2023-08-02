#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  process_trends.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 11.22.2022

import pandas as pd
from os import listdir
import geopandas as gpd
import numpy as np
import pickle

root = "./trends_data_2022"
dirnames = [
        'travel', 
        'oblast',
        'internal_city', 
        'external_city'
        ]

trends = {}
for v in dirnames:
    dirr = root+'/'+ v
    files = listdir(dirr)
    adj_files = sorted([x for x in files if x.split('_')[-1] == 'adjusted.csv'])
    print(adj_files)
    if v == 'travel':
        dates = [pd.to_datetime(x.split('TransportationWords')[1].split('_adjusted.csv')[0].split('-to-')) for x in adj_files]
    else:
        dates = [pd.to_datetime(x.split('_')[1].split('-to-')) for x in adj_files]

    print(dates)
    dd = [(x[1] - x[0]).days for x in dates]
    is1 = [x==1 for x in dd]
    adj_files = [x for i,x in enumerate(adj_files) if is1[i]]
    dates = [x for i,x in enumerate(dates) if is1[i]]

    dfs = []
    for fi,f in enumerate(adj_files):
        df = pd.read_csv(dirr+'/'+f)
        ind = np.sum(df.iloc[:,10:], axis = 1)
        date = dates[fi][0]
        ind.index = pd.MultiIndex.from_tuples([(oblast, date) for oblast in df['geoName']], names = ['Oblast','Date'])
        dfs.append(ind)
    this_series = pd.concat(dfs)

    this_series.index.duplicated()

    trends[v] = pd.DataFrame(this_series, columns = [v])

trends_df = pd.concat(list(trends.values()), axis = 1)
trends_df.columns = list(trends.keys())
trends_df.index.names = ['oblast','date']

trends_df['ob'] = trends_df.index.get_level_values('oblast')
trends_df['date'] = trends_df.index.get_level_values('date')

## Attach macroregions
macro_regions = {
     "Cherkas'ka" : 'Center',
     "Chernihivs'ka" : 'North',
     'Chernivtsi' : 'West',
     "Chernivets'ka" : 'West',
     'Crimea' : 'Crimea',
     "Dnipropetrovsk" : 'East',
     "Donetsk" : 'East',
     "Ivano-Frankivs'ka": 'West',
     'Kharkiv' : 'East',
     "Khersons'ka" :'South',
     "Khmel'nyts'ka":'West',
     'Kyiv city':'Kyiv',
     "Kyivs'ka" :'North',
     "Kirovohrads'ka":'Center',
     "Lviv":'West',
     "Luhans'ka":'East',
     "Mykolaivs'ka" :'South',
     'Odessa':'South',
     "Poltavs'ka" :'Center',
     "Rivnens'ka" :'West',
     "Sevastopol' city":'Crimea',
     "Sums'ka":'North',
     "Ternopil's'ka'":'West',
     'Transcarpathia':'West',
     "Vinnyts'ka":'Center',
     "Volyns'ka":'West',
     "Zaporiz'ka" :'East',
     "Zakarpats'ka" : 'West',
     "Zhytomyrs'ka" :'North'
     }

macro_pd = pd.DataFrame([list(macro_regions.keys()),list(macro_regions.values())]).T
macro_pd.columns = ['name','macro']

ob_diff = ['oblast', 'Oblast']
trends_df['name'] = [(x.split(' ')[0]) if any(o in x for o in ob_diff) else x for x in trends_df['ob']]
tdf = trends_df.merge(macro_pd, on = 'name')

iscrimean = tdf['macro']=='Crimea'
tdf = tdf.loc[~iscrimean,:]

#deal with NaN values
for c in dirnames:
    tdf[c] = tdf[c].fillna(0)


tdf = tdf.groupby(['date','macro']).sum()
tdf.index.names = ['date','macro']
all_dates = pd.date_range(np.min(tdf.index.get_level_values('date')), np.max(tdf.index.get_level_values('date')))
all_places = list(set(tdf.index.get_level_values('macro')))
ind2 = pd.MultiIndex.from_product([all_dates,all_places], names = ['date','macro'])
tdf = tdf.reindex(ind2, fill_value = 0) 
tdf.index.sortlevel('date')
tdf.drop(columns=['ob', 'name'], inplace=True)

#pd.concat([dfh['food'],dfh['health']], axis = 1)
with open("trends_2022.pkl",'wb') as f:
    pickle.dump(tdf, f)

