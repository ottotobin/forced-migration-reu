'''
This file is messy, and a lot of it is hard-coded, but it'll get the job done.
'''

from tkinter import W
from scipy import stats
import matplotlib.pyplot as plt
import pickle
import numpy as np
import matplotlib.dates as mdates
import pandas as pd
import statsmodels.api as sm
import datetime
import argparse
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
import tqdm
import pylab as py
import matplotlib.ticker as mtick
from matplotlib import style
from dateutil.relativedelta import relativedelta

# set up parser
parser = argparse.ArgumentParser(description="PCA Stuff")
parser.add_argument("--country", type=str, default='VEN', help='String, three-letter ISO code for GTrends data, default is VEN.')
parser.add_argument("--normalize", type=str, default='False', help='Boolean, whether or not to normalize GTrends data, default is False.')
global args
args = parser.parse_args()

file_list = ['./TrendsData/' + args.country + '/general_' + args.country + '_2018-01-01-to-2023-05-31_adjusted.csv', './TrendsData/' + args.country + '/travel_' + args.country + '_2018-01-01-to-2023-05-31_adjusted.csv', './TrendsData/' + args.country + '/geography_' + args.country + '_2018-01-01-to-2023-05-31_adjusted.csv', './TrendsData/' + args.country + '/destination_' + args.country + '_2018-01-01-to-2023-05-31_adjusted.csv']
key_list = ['general', 'travel', 'geography', 'destination']
month_list_span = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]
month_list_eng = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
year_list = [2018, 2019, 2020, 2021, 2022]

def getRegressionPlotMultiOffset(indicator1_df, indicator2_df, df, var_list):

    ''' # for offsetting - i didn't use date time object i just removed the back few values and front few values from each dataframe which should have the same effect
    indicator_df = indicator_df.drop(47)
    indicator_df = indicator_df.drop(46)
    indicator_df = indicator_df.drop(45)
    indicator_df = indicator_df.reset_index(drop=True)

    df = df.drop(0)
    df = df.drop(1)
    df = df.drop(2)
    df = df.reset_index(drop=True) '''

    # date indexing (not needed for our code)
    ''' X = pd.DataFrame(np.zeros([df.shape[0], trends_df.shape[1]]))
    X.columns = trends_df.columns
    for i in range(df.shape[0]):

        # dfrom = df['date_from'][i] + pd.Timedelta(days=offset)
        # dto = df['date_to'][i] + pd.Timedelta(days=offset)
        
        in_date_range = trends_df.loc[(slice(dfrom,dto)),:]
        if in_date_range.shape[0] > 0:
            in_region_too = in_date_range.xs(df[i])
            X.iloc[i,:] = np.mean(in_region_too,axis=0)
        else:
            X.iloc[i,:] = np.repeat(np.nan, X.shape[1]) '''

    # missing = np.any(indicator_df.isna(), axis = 1)
    # dfo = dfo.loc[~missing,:]

    offset_range_1 = np.arange(-10, 11)
    offset_range_2 = np.arange(-10, 11)
    r_squared_list = []
    offset_list = []

    indicator1_df['date'] = pd.to_datetime(indicator2_df['date'])
    indicator2_df['date'] = pd.to_datetime(indicator2_df['date'])

    df['date'] = pd.to_datetime(df['date'])
    for x in offset_range_1:
        for y in offset_range_2:

            new_indicator1_df = indicator1_df.copy()
            new_indicator2_df = indicator2_df.copy()
            new_df = df.copy()

            new_indicator1_df = indicator1_df.shift(x)
            '''for i, row in indicator1_df.iterrows():
                date = row['date'] + relativedelta(months=x)
                new_indicator1_df.at[i, 'date'] = date'''

            new_indicator2_df = new_indicator2_df.shift(y)
            '''for i, row in indicator2_df.iterrows():
                date = row['date'] + relativedelta(months=y)
                new_indicator2_df.at[i, 'date'] = date'''

            #new_indicator1_df = new_indicator1_df[new_indicator1_df['date'].isin(df['date'].tolist())].reset_index(drop=True)
            #new_indicator2_df = new_indicator2_df[new_indicator2_df['date'].isin(df['date'].tolist())].reset_index(drop=True)
            merge_indicator_df = pd.concat([new_indicator1_df, new_indicator2_df.reset_index(drop=True)], axis=1)
            #new_df = df[df['date'].isin(new_indicator1_df['date'].tolist())].reset_index(drop=True)

            vars = var_list
            lmfit = sm.OLS((new_df['Total']), sm.add_constant(merge_indicator_df.loc[:,vars]), missing='drop').fit()
            lmfit.summary()
            r2 = lmfit.rsquared
            print("Offsets : " + str(x) + " acled" + ", " + str(y) + " trends")
            print(r2)
            r_squared_list.append(r2)
            offset_list.append([x, y])
    
    print(max(r_squared_list))
    i = r_squared_list.index(max(r_squared_list))
    print(offset_list[i])
    return var_list, max(r_squared_list), offset_list[i]

# meant for use with emotion_counts.csv or emotion_counts_normalized.csv
def varianceTimeSeries(df):

    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by=['date'], inplace=True)
    print(df)

    emotion_list = ['fear', 'anger', 'sadness', 'joy']
    temp = 0
    f_list = []
    a_list = []
    s_list = []
    j_list = []
    for emotion in emotion_list:
        df[emotion] += 1
        for i in range(len(df)):
            if i < (len(df) - 1):
                temp = ((df[emotion][i+1] - df[emotion][i]) / df[emotion][i]) * 100
                print(temp)
                ''' if df[emotion][i] > df[emotion][i+1]:
                    temp = -1
                elif df[emotion][i] < df[emotion][i+1]:
                    temp = 1
                elif df[emotion][i] == df[emotion][i+1]:
                    temp = 0 '''
                if emotion == 'fear':
                    f_list.append(temp)
                elif emotion == 'anger':
                    a_list.append(temp)
                elif emotion == 'sadness':
                    s_list.append(temp)
                elif emotion == 'joy':
                    j_list.append(temp)
    
    df = df.drop(52)
    fig,ax = plt.subplots(figsize=(3, 2))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    #ln1 = plt.plot(df['date'], j_list, label='joy')
    ln2 = plt.plot(df['date'], s_list, label='sadness')
    ln3 = plt.plot(df['date'], a_list, label='anger')
    ax2=ax.twinx()
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
    ln4 = ax2.plot(df['date'], f_list, label='fear', color='red')
    plt.title("Daily Percent Change in Sadness, Anger, and Fear Values \nfor Tweets from Venezuela")
    plt.xlabel('Date')
    ax.set_ylabel('Percent Change')
    plt.ylabel('Percent Change')
    '''x_ticks = range(52)
    dates = df['date'].tolist()
    plt.xticks(x_ticks, dates)'''
    plt.grid(True)
    plt.gcf().autofmt_xdate()

    lns = ln2+ln3+ln4
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0)

    #plt.legend()
    print(style.available)
    plt.show()               

# meant for use with ./acled_all_2018_2023.csv, aggregates data by day
def aggregateACLEDByDay(df):

    df['event_date'] = pd.to_datetime(df['event_date'], format='%d-%b-%y')
    df.sort_values(by='event_date', inplace = True)

    sum_df = pd.DataFrame(columns=['Date', '# of Events'])

    offset = 1
    date = pd.to_datetime('2023-04-22')
    end_date = pd.to_datetime('2023-06-06')
    
    while date < end_date:
        count = 0
        for ev_date in df['event_date']:
            if ev_date == date:
                count += 1
        
        to_add = { 'Date' : [date], '# of Events' : [count] }
        temp_df = pd.DataFrame(to_add)
        sum_df = pd.concat([sum_df, temp_df.reset_index(drop=True)])

        date += pd.Timedelta(days=offset)

    ''' for year in year_list:
        for month in month_list_eng:
            fatality_count = 0
            for i in range(0, len(df)):
                if df['year'][i] == year:
                    if month in df['event_date'][i].split(" "):
                        fatality_count += df['fatalities'][i]

            to_add = { 'Year' : [year], 'Month' : [month], 'Fatalities' : [fatality_count] }
            temp_df = pd.DataFrame(to_add)
            sum_acled_df = pd.concat([sum_acled_df, temp_df.reset_index(drop=True)]) ''' 
    
    sum_df.to_csv('acled_daily-sum_Apr-2023_Jun-2023_event-count.csv', index=False)

# ignore
def convertLatLong():

    # this is the script I wrote to convert the Colombia csv. Will need retooling to apply to the ACLED data.

    df = pd.read_csv('ven_to_col_2012-2022.csv')
    geolocator = Nominatim(user_agent="test")
    df = df.rename(columns={"Latitud - Longitud" : "State"})
    checked_list = []
    for i in tqdm.tqdm(range(0, len(df))):
        temp = df.iloc[i]['State']
        print(temp)
        if temp.replace(' ', '').replace(',', '').isalpha() != True:
            temp_parse = temp.replace('(','').replace(')','')
            lat, long = temp_parse.split(',')
            location = geolocator.reverse(lat + "," + long, timeout=100000)
            address = location.raw['address']
            state = address.get('state', '')
            if len(state) == 0:
                state = 'Not Available'
            df['State'] = df['State'].replace([temp], [state])
            checked_list.append(temp)

    df.to_csv('ven_to_col_locations.csv', index=False)

# meant for use with ./acled_all_2018_2023.csv, aggregates data by month
def aggregateACLEDByMonth(df):

    df = df[df.year != 2023]
    df.reset_index(inplace=True)
    sum_acled_df = pd.DataFrame(columns=['Year', 'Month', 'Fatalities'])

    for year in year_list:
        for month in month_list_eng:
            fatality_count = 0
            for i in range(0, len(df)):
                if df['year'][i] == year:
                    if month in df['event_date'][i].split(" "):
                        fatality_count += df['fatalities'][i]

            to_add = { 'Year' : [year], 'Month' : [month], 'Fatalities' : [fatality_count] }
            temp_df = pd.DataFrame(to_add)
            sum_acled_df = pd.concat([sum_acled_df, temp_df.reset_index(drop=True)])
    sum_acled_df.to_csv('acled_monthly-sum_2018-2022_fatalities.csv', index=False)


    ''' print(df['sub_event_type'].unique())
    topic_list = ['Attack', 'Other', 'Armed clash', 'Peaceful protest', 'Abduction/forced disappearance', 'Looting/property destruction', 'Mob violence', 'Sexual violence', 'Disrupted weapons use', 'Violent demonstration', 'Agreement', 'Grenade', 'Protest with intervention', 'Arrests', 'Shelling/artillery/missile attack', 'Remote explosive/landmine/IED', 'Change to group/activity', 'Non-violent transfer of territory', 'Suicide bomb', 'Excessive force against protesters', 'Headquarters or base established', 'Air/drone strike', 'Non-state actor overtakes territory']
    df.reset_index(inplace=True)
    sum_acled_df = pd.DataFrame(columns=['Year', 'Month', 'Attack', 'Other', 'Armed clash', 'Peaceful protest', 'Abduction/forced disappearance', 'Looting/property destruction', 'Mob violence', 'Sexual violence', 'Disrupted weapons use', 'Violent demonstration', 'Agreement', 'Grenade', 'Protest with intervention', 'Arrests', 'Shelling/artillery/missile attack', 'Remote explosive/landmine/IED', 'Change to group/activity', 'Non-violent transfer of territory', 'Suicide bomb', 'Excessive force against protesters', 'Headquarters or base established', 'Air/drone strike', 'Non-state actor overtakes territory'])

    for year in year_list:
        for month in month_list_eng:
            sum_list = [0 for i in range(len(topic_list))]
            for i in range(0, len(df)):
                if df['year'][i] == year:
                    if month in df['event_date'][i].split(" "):
                        for j, topic in enumerate(topic_list):
                            if topic in df['sub_event_type'][i]:
                                sum_list[j] += 1
                        
            to_add = { 'Year' : [year], 'Month' : [month],  'Attack' : [sum_list[0]], 'Other' : [sum_list[1]], 'Armed clash' : [sum_list[2]], 'Peaceful protest' : [sum_list[3]], 'Abduction/forced disappearance' : [sum_list[4]], 'Looting/property destruction' : [sum_list[5]], 'Mob violence' : [sum_list[6]], 'Sexual violence' : [sum_list[7]], 'Disrupted weapons use' : [sum_list[8]], 'Violent demonstration' : [sum_list[9]], 'Agreement'  : [sum_list[10]], 'Grenade' : [sum_list[11]], 'Protest with intervention' : [sum_list[12]], 'Arrests' : [sum_list[13]], 'Shelling/artillery/missile attack' : [sum_list[14]], 'Remote explosive/landmine/IED' : [sum_list[15]], 'Change to group/activity' : [sum_list[16]], 'Non-violent transfer of territory' : [sum_list[17]], 'Suicide bomb' : [sum_list[18]], 'Excessive force against protesters' : [sum_list[19]], 'Headquarters or base established' : [sum_list[20]], 'Air/drone strike' : [sum_list[21]], 'Non-state actor overtakes territory' : [sum_list[22]] }
            temp_df = pd.DataFrame(to_add)
            sum_acled_df = pd.concat([sum_acled_df, temp_df.reset_index(drop=True)])
    sum_acled_df.to_csv('acled_monthly-sum_2018-2022_sub-event-type.csv', index=False) '''

# meant for processing ven_to_col.csv (the full one, whatever that file name is)
def processColombiaData(df):

    sum_flow_df = pd.DataFrame(columns=df.columns)
    sum_flow_df = sum_flow_df.drop('State', axis=1).drop('Nacionalidad', axis=1).drop('Codigo Iso 3166', axis=1).drop('Femenino', axis=1).drop('Masculino', axis=1).drop('Indefinido', axis=1)
    for year in year_list:
        for month in month_list_span:
            month_sum = 0
            for i in range(0, len(df)):
                if df['Año'][i] == year and df['Mes'][i] == month:
                    month_sum += int(df['Total'][i])
            to_add = { 'Año' : [year], 'Mes' : [month], 'Total' : [month_sum] }
            temp_df = pd.DataFrame(to_add)
            sum_flow_df = pd.concat([sum_flow_df, temp_df.reset_index(drop=True)])
    sum_flow_df.to_csv('ven_to_col_2018-2022_sum.csv', index=False)
     
    df = df[df.Año != 2012]
    df = df[df.Año != 2013]
    df = df[df.Año != 2014]
    df = df[df.Año != 2015]
    df = df[df.Año != 2016]
    df = df[df.Año != 2017]

# indicator df should be sum_trends_df or acled_df
def getRegressionPlot(indicator_df, df):

    ''' offset_range = np.arange(-10, 11)
    r_squared_list = []
    offset_list = []

    indicator_df['date'] = pd.to_datetime(indicator_df['date'])
    df['date'] = pd.to_datetime(df['date'])
    for x in offset_range:

        new_indicator_df = indicator_df.copy()
        new_df = df.copy()

        new_indicator_df = indicator_df.shift(x)
        for i, row in indicator_df.iterrows():
            date = row['date'] + relativedelta(months=x)
            new_indicator_df.at[i, 'date'] = date

        #new_indicator_df = new_indicator_df[new_indicator_df['date'].isin(df['date'].tolist())].reset_index(drop=True)
        #print(new_indicator_df)
        #new_df = df[df['date'].isin(new_indicator_df['date'].tolist())].reset_index(drop=True)
        #print(new_df)

        vars = ['Violence against civilians']
        lmfit = sm.OLS((new_df['Total']), sm.add_constant(new_indicator_df.loc[:,vars]), missing='drop').fit()
        lmfit.summary()
        r2 = lmfit.rsquared
        print("Offset: " + str(x))
        print(r2)
        r_squared_list.append(r2)
        offset_list.append(x)

    plt.scatter(offset_list, r_squared_list)
    plt.show() '''

    # get regression model for each column in indicator_df and df for
    for v in indicator_df.columns:
        lmfit = sm.OLS(df['Total'], sm.add_constant(indicator_df.loc[:,v])).fit()
        lmfit.summary()
        r2 = lmfit.rsquared
        print(v)
        print(r2)
        np.square(np.corrcoef(df, indicator_df.loc[:,v]))

        # graph regression plot
        fig, ax = plt.subplots(figsize=(3,2))
        plt.scatter(indicator_df.loc[:,v], df)
        plt.xlabel('Venezuelan Google Trends \'Destination\' Statistic (monthly)')
        plt.ylabel('Number of Venezuelan Migrants Entering Colombia (monthly)')
        
        # this number is very finicky
        ax.annotate("r-squared = {:.3f}".format(r2), (150,125000))
        plt.grid(True)
        ax.set_title('\'Destination\' Indicator Values vs. \nNumber of Venezuelan Migrants Entering COL (2018-2022)\n (3 month offset)', pad=10)
        fig = sm.graphics.abline_plot(model_results=lmfit, color='red', ax=plt.gca())

        plt.show()

# sums each phrase in indicator categories together and puts all categories into a dataframe together, for use w/ Google Trends csvs
def sumDataframes():
    columns = ['date', 'general', 'travel', 'geography', 'destination']
    sum_df = pd.DataFrame(columns=columns)
    for file, label in zip(file_list, key_list):
        df = pd.read_csv(file, index_col=False)

        # remove rows not in flow data
        
        df = df[df.date != '2023-01-01']
        df = df[df.date != '2023-02-01']
        df = df[df.date != '2023-03-01']
        df = df[df.date != '2023-04-01']
        df = df[df.date != '2023-05-01']

        # offsetting
        ''' df = df[df.date != '2022-12-01']
        df = df[df.date != '2022-11-01']
        df = df[df.date != '2022-10-01'] '''


        for date in df.date:
            if '2018' in date:
                df = df[df.date != date]
            if '2019' in date:
                df = df[df.date != date]
            if '2020' in date:
                df = df[df.date != date]
            if '2021' in date:
                df = df[df.date != date]

        # remove extraneous columns
        #df = df.drop('date', axis=1)
        df = df.drop('country', axis=1)
        df = df.drop('topic', axis=1)

        # normalize data
        if args.normalize == 'True':
            df = normDataframe(df)
        #df['Sum'] = df.sum(axis=1)
        df['Sum'] = df.iloc[:, 1:].sum(axis=1)
        #df['Sum'] = np.log(df['Sum'])
        sum_df[label] = df['Sum']
        sum_df['date'] = df['date']

    return sum_df.reset_index(drop=True)

# i don't really remember what this one does i don't think i've use it in a while
def getLineGraph(df):
    r = 'East'
    fig = plt.figure()
    for k in key_list:
        plt.plot(df.loc[:,k], label=k)
    ax = plt.gca()
    x_ticks = [0, 12, 24, 36, 48, 60]
    dates = ['2018', '2019', '2020', '2021', '2022', '2023']
    plt.xticks(x_ticks, dates)
    plt.legend()
    plt.show()
    # plt.savefig("ven_trends_unnorm.pdf")
    plt.close()

# normalizes dataframe
def normDataframe(df):
    #key_list = []
    norm_df = df.copy()
    #for col in norm_df.columns:
    #    key_list.append(str(col))
    for key in norm_df.columns:
        norm_df[key] = (norm_df[key] - np.mean(norm_df[key])) / np.std(norm_df[key])
    return norm_df

# calculates correlation matrix for df (meant to be one indicator) and merge_flow_df (flow data from different countries/places merged into one dataframe)
def calculateCorrelation(df, merge_flow_df):

    # offsetting
    merge_flow_df = merge_flow_df.drop(0)
    merge_flow_df = merge_flow_df.drop(1)
    #merge_flow_df = merge_flow_df.drop(2)
    merge_flow_df = merge_flow_df.reset_index(drop=True)

    df = df.drop(47)
    df = df.drop(46)
    #df = df.drop(45)
    df = df.reset_index(drop=True)

    corr = np.zeros([len(merge_flow_df.columns), len(df.columns)])
    for i,v1 in enumerate(merge_flow_df):
        for j,v2 in enumerate(df):
            corr[i,j] = np.corrcoef(merge_flow_df[v1], df[v2])[0,1]
            print(v1 + " and " + v2 + "corrcoef: " + str(corr[i,j]))
    visualizeCorrelation(df, corr)

    # computePCA(sum_df, corr, sum_df.columns)
    return corr

# visualize correlation matrix
def visualizeCorrelation(df, corr):
    stat_key_list = df.columns.tolist()
    state_key_list = ['Brazil', 'Colombia']
    fig = plt.figure(figsize=(4,2))
    plt.imshow(corr, cmap = plt.get_cmap('Reds'), vmin = 1, vmax = 0)
    plt.colorbar()
    ax = plt.gca()
    #ax.set_xticklabels(vars)
    ax.set_xticks(np.arange(len(stat_key_list)))
    ax.set_xticklabels(stat_key_list)
    ax.set_yticks(np.arange(len(state_key_list)))
    ax.set_yticklabels(state_key_list)
    plt.xticks(rotation=10)
    # ax.set_title('BRA and COL Flow Data vs. VEN Google Trends Stats \nCorrelation Matrix', pad=20)
    ax.set_title('BRA and COL Flow Data vs. VEN Google\n Trends Stats Correlation Matrix (2 month offset)', pad=20)
    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    plt.savefig("cov_im.jpeg")
    plt.show()
    plt.close()

# comput elbow graph (i haven't used this one in a while)
def computePCA(sum_df, corr, key_list):
    ed = np.linalg.eigh(corr)
    ed[0]
    ev1 = pd.Series(ed[1][:,-1], index = key_list)
    ev2 = pd.Series(ed[1][:,-2], index = key_list)
    sum_df.iloc[0,:]
    V = np.stack([-ev1,-ev2]).T
    low_d_df = sum_df @ V

    fig = plt.figure()
    plt.scatter(np.arange(len(key_list)), np.flip(ed[0]))
    plt.ylabel('Eigenvalue')
    ax = plt.gca()
    ax1 = ax.twinx()
    ax1.plot(np.cumsum(np.flip(ed[0]))/np.sum(ed[0]), linestyle = '--')
    ax1.set_ylabel('Cumulative Proportion')
    ax1.set_ylim(0,1)
    # plt.show()
    # plt.savefig("evals_test.pdf")
    plt.close()

    # Look at variable contribution to top vectors
    contrib = np.square(ev1) + np.square(ev2)
    contrib.sort_values(ascending=False)

    # That was for K = 2, this is general K.
    top_K = 4
    contrib = pd.Series(np.sum(np.square(ed[1][:,-top_K:]), axis = 1), index = ev1.index)
    contrib.sort_values(ascending=False)

# plots two dataframes in a time series
def plotTimeSeries(df1, df2):

    ''' df1 = df1.drop('Year', axis=1).drop('Month', axis=1)
    # df2 = normDataframe(df2.drop('Año', axis=1).drop('Mes', axis=1))
    # df3 = normDataframe(df3.drop('Year', axis=1).drop('Month', axis=1))
    df3 = df3[df3.Year != 2023]
    df1 = df1.reset_index() '''

    # df1 = df1.drop('index', axis=1)

    plt.style.use(['seaborn-v0_8-poster'])
    fig,ax = plt.subplots(figsize=(3, 2))
    #for v in df1.columns:
    ax.plot(df1.index.values, df1['Violence against civilians'].to_list(), label='Violence against civilians')
    ax.plot(df1.index.values, df1['Strategic developments'].to_list(), label='Strategic developments')
    ax.plot(df1.index.values, df1['Battles'].to_list(), label='Battles')
    ax.plot(df1.index.values, df1['Protests'].to_list(), label='Protests')
    ax.plot(df1.index.values, df1['Riots'].to_list(), label='Riots')
    ax.plot(df1.index.values, df1['Explosions/Remote violence'].to_list(), label='Explosions/Remote violence')
    ax2=ax.twinx()
    ax.legend(loc="upper center")
    plt.xticks([0, 12, 24, 36, 48], [2019, 2020, 2021, 2022, 2023])
    # plt.grid(True)
    #ax.legend(loc="upper center")
    plt.xticks(rotation=90)
    ax2.scatter(df2.index.values, df2['Total #'], label='Brazil')
    ax2.scatter(df2.index.values, df2['Total'], label='Colombia')
    ax2.legend(loc="upper right")
    ax.set_xlabel('Year')
    ax.set_ylabel('ACLED Event Occurences (monthly)')
    ax2.set_ylabel('# of Venezuelan Migrants \n Entering BRA and COL (monthly, normalized)')
    ax.set_title('Time Series of ACLED Event Indicators \nand Migration to BRA and COL (2019-2022)', pad=10)
    plt.savefig("TimeSeriesBR.png")

def main():

    emotion_df = pd.read_csv('./emotion_counts.csv')
    emotion_df['date'] = pd.to_datetime(emotion_df['date'])
    emotion_df.sort_values(by='date', inplace = True)
    emotion_df = emotion_df.reset_index(drop=True)
    emotion_df.index = emotion_df['date']
    emotion_df.drop('date', inplace=True, axis=1)
    
    # use this one
    monthly_emotions_df = emotion_df.resample('M').sum()
    #varianceTimeSeries(df)

    # initialize ACLED dataframe
    acled_df = pd.read_csv('/Users/bernardmedeiros/Desktop/ColombiaData/acled_monthly-sum_2018-2022_event-type.csv')
    #acled_df = acled_df[acled_df.Year != 2018]
    acled_df = acled_df.drop('Year', axis=1).drop('Month', axis=1).reset_index(drop=True)

    # initialize GTrends dataframe
    sum_trends_df = sumDataframes()
    getRegressionPlot(monthly_emotions_df.reset_index(), sum_trends_df.reset_index())

    merge_indicator_df = pd.concat([sum_trends_df, acled_df.reset_index(drop=True)], axis=1)

    # initialize brazil flow data dataframe
    df_brazil = pd.read_csv('./Brazil_Monthly_Data_2019-2023.csv')
    df_brazil = df_brazil[df_brazil.Year != 2023]
    df_brazil = df_brazil.drop('Year', axis=1).drop('Month', axis=1)

    # initialize colombia flow data dataframe
    df_col = pd.read_csv('./ven_to_col_2018-2022_sum.csv')
    #df_col = df_col[df_col.Año != 2018]
    df_col = df_col.drop('Año', axis=1).drop('Mes', axis=1)
    #df_col['Total'] = np.log(df_col['Total'])

    # create merged brazil and colombia flow data dataframe
    merge_df = pd.concat([df_brazil, df_col.reset_index(drop=True)], axis=1)
    merge_df['sum'] = merge_df.sum(axis=1)
    merge_df['date'] = merge_indicator_df['date']
    df_col['date'] = merge_df['date']
    df_brazil['date'] = merge_df['date']
    #merge_df = normDataframe(merge_df)

    # plot time series for acled data, and google trends data
    #plotTimeSeries(acled_df, merge_df)
    #plotTimeSeries(sum_trends_df, merge_df)

    # potential transformation options ('Total' is Colombia and 'Total #' is Brazil -- confusing, i know)
    #acled_df = np.sqrt(acled_df)
    #sum_trends_df = np.log(sum_trends_df)
    #merge_df['Total'] = np.log(merge_df['Total'])
    #merge_df['Total #'] = np.log(merge_df['Total #'])

    # colombia regression plots

    # for testing all combinations -- destination x violence against civilians is best
    '''trends_list = ['geography', 'destination', 'travel', 'general']
    ACLED_list = ['Violence against civilians', 'Strategic developments', 'Battles', 'Protests', 'Riots', 'Explosions/Remote violence']
    output_list = []

    for trends in trends_list:
        for acled in ACLED_list:
    
            var_list = [trends, acled]

            vars, r2, offset_vals = getRegressionPlotMultiOffset(acled_df.reset_index(drop=True), sum_trends_df.reset_index(drop=True), df_col.reset_index(drop=True), var_list)
            output_list.append([vars, r2, offset_vals])
    
    print(output_list)'''

    #getRegressionPlotMultiOffset(acled_df.reset_index(drop=True), sum_trends_df.reset_index(drop=True), df_col.reset_index(drop=True), ['destination', 'Violence against civilians'])

    #getRegressionPlot(merge_indicator_df.reset_index(drop=True), df_col.reset_index(drop=True))
    
    #getRegressionPlot(sum_trends_df.reset_index(drop=True), df_brazil.reset_index(drop=True))

    #acled_df['date'] = sum_trends_df['date']
    #getRegressionPlot(acled_df.reset_index(drop=True), df_brazil.reset_index(drop=True))

    # get correlation matrices
    #calculateCorrelation(acled_df, merge_df)
    #calculateCorrelation(sum_trends_df, merge_df)

    return 0

if __name__ == "__main__":
    main()