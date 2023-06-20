
"""
Author: Grey, Yiqing

Last Update: Jan 28 2023

Important Note:
  Including a lynchWord in parameters is discouraged, because the dynamically selected lynchWords that the program uses are almost certainly
    better. The program selects a lynchWord based on the data it has at the moment and will change the moment it finds a better one. This
    means that it can select different lynchWords for different days (or even the same days) based on the data of each day, unlike a static
    lynchWord that must be used for all days and within all days.
  That being said, The words in the words parameter should be ordered roughly by importance/expected number of searches, so that the program
    will immediately select an ideal lynchWord that can hopefully be used for the entire day. The first group of five is the most important
    in this regard. The words in the words parameter are ordered before any words in files, so keep that in mind.

countryCode: Official two letter iso code of country. Can get codes by searching 'iso codes' in google or here:
    https://www.nationsonline.org/oneworld/country_code_list.htm#:~:text=The%20ISO%20country%20codes%20are,a%20country%20or%20a%20state.

lynchWord: Word to base proportionality adjustment off of. Best to be something in the upper middle in terms of data values.
    This will theoretically result in more accurate proportinality adjustment, but if the lynchWord is bad it could make things worse.
    
wait: multiplier for wait times. Try increasing this if you keep getting error 429. If the errors persist irregularly, you likely need a 
    less crowded network. If they are regular, Google is just angry at you and all you can do is wait.

USAGE:
python gtrAPI_Daily.py 
    --topic economics 
    --wordsfile /home/airflow/dags/googleTrends/parameter/words_general /home/airflow/dags/googleTrends/parameter/words_economics 
    --wait 2
    --output /home/airflow/dags/googleTrends/result/

"""
import os 
import sys
import warnings
import argparse
import pandas as pd
import datetime
import time
from pytrends.request import TrendReq

def get_wordslist(wordsFile):
    global words
    words = []
    try:
        for file in wordsFile:
            infile = open(file, 'r')
            for line in infile:
                if line.strip() != '':
                    words.append(line.strip())
    except IOError:
        sys.exit('Invalid wordsFile or no space')
    infile.close()

    #Delete duplicates
    words = sorted(set(words), key=lambda x: words.index(x))

    #Split words parameter into groups of 5
    wordsList = []
    for i in range(0,len(words),4):
        wordsList.append(words[i:min(i+5,len(words))])

    if len(wordsList[-1]) == 1:
        wordsList = wordsList[:-1]

    #Check if there is a lynchWord, if so, modify the wordList accordingly
    '''lynchWord = []
    if 'lynchWord':
        idx = 0
        wlLen = len(wordsList) - 1
        for i in wordsList:
            #Move first index to new location in wordList
            if idx < wlLen:
                wordsList[idx][min(4,len(i)-1)] = i[0]
            else:
                if len(i) < 5:
                    wordsList[idx].append(i[0])
                else:
                    wordsList.append([i[0]])

            #Move lynchWord to first location in wordList[idx]
            wordsList[idx][0] = lynchWord[0]
            idx += 1'''

    return wordsList
    
#Function to request and return the response.
#Note: Catch mostly for exception 429 (too many requests). Sometimes waiting 60 sec will allow requests again, sometimes it must wait a day.
#Note: Code 400 error means that something is wrong with the request, which probably means a problem with the parameters.
#Note: Every now and then theres another code (in the 430s or 440s) that pops up. There is no reason for it on our end.
#      Just ignore it, the program should start up again after a minute.
def getReq(w5,cc,timefr,mult):
    #Inputs: w5 is list of words (max 5 per request), cc is country code, timefr is start and end dates
    #Outputs: pandas data frame with relevant data (directly from response)
    #How it works: magic
    try:
        pytrends = TrendReq(tz=360)
        time.sleep(0.1*mult)
        #Build payload for request
        pytrends.build_payload(w5, cat=0, geo=cc, timeframe=timefr)
        time.sleep(0.2*mult)
        #Return response
        return pytrends.interest_by_region(inc_low_vol=True, inc_geo_code=True).reset_index()
    except Exception as e:
        # continue to request every 30 minutes
        print('Warning:', e, file=sys.stderr)
        while True:
            print('Sleeping for 30 mins', file=sys.stderr)
            print('Error time: ' + datetime.datetime.now().strftime("%H:%M:%S") +
                  " on " + datetime.date.today().strftime("%m/%d/%y"))
            time.sleep(1800)
            try:  # Try again after 30 mins if 429
                pytrends = TrendReq(tz=360)
                pytrends.build_payload(w5, cat=0, geo=cc, timeframe=timefr)
                time.sleep(0.2*mult)
                return pytrends.interest_by_region(inc_low_vol=True, inc_geo_code=True).reset_index()
            except Exception as e2:
                print('Another failure:', e2, file=sys.stderr)
                # sys.exit(e2)

# runs an API request with the given inputs, and creates a csv file for the data
def run(date, wordsList, countryCode, wait, outputDirectory):
    #Put date in query format
    dt = [date.strftime('%Y-%m-%d ') + (date+datetime.timedelta(days=1)).strftime('%Y-%m-%d'),
        (date+datetime.timedelta(days=-1)).strftime('%Y-%m-%d ') + (date+datetime.timedelta(days=1)).strftime('%Y-%m-%d')]

    for i in [0,1]:
        #Get data for first five words
        frame = getReq(wordsList[0], countryCode, dt[i], wait)

        #Initialize backup, which will keep the data directly from google
        backup = frame
        backIter = 0
        for fiveW in wordsList[1:]:
            #Dynamically select lynchword based on lowest number of 0s and 100s
            ## Solve multiobjective optimization problem via scalarization:
            #    a) Count the number of times a term has value 100
            #    b) Count the number of times a term has a nonzero value
            #    c) Count the number of ideal terms (check between 90 and 10 and 70 and 30)
            #    d) Select as lynchword the word which maximizes the quantity:
            #         nonzero_times + ideal_times - hundred_times*scalarizer
            # For now, just using scalarizer = 1/(number of regions), because we want to avoid zeros more than we want to avoid 100s. 
            scalarizer = 1/frame.shape[0]
            #scalarizer  = 2
            #scalarizerh = 1 / frame.shape[0]
            if args.lynchWord is None:
                hundreds = (frame[wordsList[backIter]] == 100).sum() * scalarizer
                nonzeros = (frame[wordsList[backIter]] == 0).sum() + (frame[wordsList[backIter]] == 1).sum()
                ideal = (frame[wordsList[backIter]] > 10).sum() - (frame[wordsList[backIter]] > 89).sum() 
                ideal2 = (frame[wordsList[backIter]] > 30).sum() - (frame[wordsList[backIter]] > 69).sum() 
                fiveW[0] = (nonzeros + ideal + ideal2 - hundreds).idxmax()
            backIter+=1

            #Get reqs
            data = getReq(fiveW, countryCode, dt[i], wait)

            #Add data to backup frame
            backup = pd.merge(backup, data, how='outer', on=['geoName','geoCode'], suffixes=("",str(backIter)))

            #Divide columns to get ratio between old data and new data
            if (data[fiveW[0]]==0).any():
                warnings.warn("Replaced zero in lynchword with 1.")
            dv = frame[fiveW[0]].replace([0],1) / data[fiveW[0]].replace([0],1)

            #Multiply ratio by new data to make new data proportional to old data
            data[fiveW[1:]] = data[fiveW[1:]].multiply(dv,axis='index')

            #Merge new data and old data
            frame = pd.merge(frame, data.drop(columns=fiveW[0]), how='outer', on=['geoName','geoCode'])

        #Make all values in frame decimals so we know the max (0 < x < 1)
        frame[words] = frame[words].divide(frame[words].max(axis='columns').replace([0],1),axis='index')

        #Write to <output file><date>-to-<date + 1 day>.csv
        try:
            filename = "{}_{}_{}_adjusted.csv".format(args.topic, countryCode, dt[i].replace(' ','-to-'))
            f = open(outputDirectory+filename, 'w+')
            getattr(frame,'to_csv')(f)
            f.close()
            filename = "{}_{}_{}_raw.csv".format(args.topic, countryCode, dt[i].replace(' ','-to-'))
            bF = open(outputDirectory+filename, 'w+')
            getattr(backup,'to_csv')(bF)
            bF.close()
        except IOError:
            sys.exit('Invalid output file or no space to write to output file')

        #Print so that you know how much data has been collected
        print(dt[i].replace(' ','-to-') + ' data collection successful')

def main():
    parser = argparse.ArgumentParser(description="Google Trend Data")
    parser.add_argument("--topic", required=True, help="topic")
    parser.add_argument("--worddir", required=True, nargs="+",
                        help="wordsFile directory: fm-google-trends/Daily")
    parser.add_argument("--lang", required=True)
    parser.add_argument("--wait", required=True, help="multiplier for wait times")
    parser.add_argument("--output", required=True, help="output directory")
    parser.add_argument("--date", help="date for the data to run on, format YYYY-MM-DD")
    parser.add_argument("--lynchWord", help="optional lynchWord")
    global args
    args = parser.parse_args()

    dir = os.fsencode(str(args.worddir))

    for file in os.listdir(dir):
        fn = os.fsdecode(file)
        if fn.endswith(".txt"):    
            wordsList = get_wordslist(file)

            if args.date:
                date = datetime.datetime.strptime(args.date, '%Y-%m-%d').date()
            else:
                date  = datetime.date.today()

            run(date-datetime.timedelta(days=1), wordsList, args.lang, int(args.wait), args.output)
    

if __name__ == "__main__":
    main()