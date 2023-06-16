
"""
Author: Grey, Yiqing, Helge Marahrens, Nathan Wycoff, Lina Laghzaoui, Grace Magny-Fokam, Toby Otto, Rich Pihlstrom

Last Update: 16 June 2023
    Ran into an issue with the read_csv pandas function not handling arabic script. 
    As a solution, I just copied the wordslist function from the previous version to 
    make it more manual and allow for arabic script. Should work with other texts.
    New code didn't allow for the redirection of output files to a directory so I 
    just added that in as well. 
    -Rich

Necessary Packages:
    You will need to install pytrends in order to use the API in this code:
        pip3 install pytrends

Important Note:
  Including a pivotalWord in parameters is discouraged, because the dynamically selected pivotalWords that the program uses are almost certainly
    better. The program selects a pivotalWord based on the data it has at the moment and will change the moment it finds a better one. This
    means that it can select different pivotalWords for different days (or even the same days) based on the data of each day, unlike a static
    pivotalWord that must be used for all days and within all days.
  That being said, The words in the words parameter should be ordered roughly by importance/expected number of searches, so that the program
    will immediately select an ideal pivotalWord that can hopefully be used for the entire day. The first group of five is the most important
    in this regard. The words in the words parameter are ordered before any words in files, so keep that in mind.

countryCode: Official two letter iso code of country. Can get codes by searching 'iso codes' in google or here:
    https://www.nationsonline.org/oneworld/country_code_list.htm#:~:text=The%20ISO%20country%20codes%20are,a%20country%20or%20a%20state.

pivotalWord: Word to base proportionality adjustment off of. Best to be something in the upper middle in terms of data values.
    This will theoretically result in more accurate proportinality adjustment, but if the pivotalWord is bad it could make things worse.
    
wait: multiplier for wait times. Try increasing this if you keep getting error 429. If the errors persist irregularly, you likely need a 
    less crowded network. If they are regular, Google is just angry at you and all you can do is wait.

USAGE:
    python3 gtrAPI_Range.py --topic MeansofTravel --wordsfile words_MeansOfTravel --wait 20 --output results/ --iso IQ

"""

TIME_FRAME = '2018-01-01 2023-05-31'

import sys
import warnings
import argparse
import pandas as pd
import datetime
import time
from pytrends.request import TrendReq
import numpy as np

#Function to read in the list of search terms from the wordsfile
#Cannot use pandas read_csv function as it does not work
#with non-ASCII Arabic script
def get_wordslist(wordsfile):
    words = []

    try:
        for file in wordsfile:
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

    return wordsList

#Function to request and return the response.
#Note: Catch mostly for exception 429 (too many requests). Sometimes waiting 60 sec will allow requests again, sometimes it must wait a day.
#Note: Code 400 error means that something is wrong with the request, which probably means a problem with the parameters.
#Note: Every now and then theres another code (in the 430s or 440s) that pops up. There is no reason for it on our end.
#      Just ignore it, the program should start up again after a minute.
def getReq(w5,cc,mult):
    #Inputs: w5 is list of words (max 5 per request), cc is country code, wait is wait time in seconds
    #Outputs: pandas data frame with relevant data (directly from response)
    def get():
        pytrends = TrendReq(tz=360)
        time.sleep(0.1*mult)
        #Build payload for request
        pytrends.build_payload(w5, cat=0, geo=cc, timeframe=TIME_FRAME)
        time.sleep(0.2*mult)
        #Return response
        df = pytrends.interest_over_time().reset_index()
        #If all 5 words do not have enough data, df will be empty
        #dataframe, so we need to check if the column exists
        if "isPartial" in df.columns:
            return df.drop('isPartial', axis = 1)
        else:
            return df

    try:
        return get()
    except Exception as e:
        # continue to request every 30 minutes
        print('Warning:', e, file=sys.stderr)
        while True:
            print('Sleeping for 30 mins', file=sys.stderr)
            print('Error time: ' + datetime.datetime.now().strftime("%H:%M:%S") +
                  " on " + datetime.date.today().strftime("%m/%d/%y"))
            time.sleep(1800)
            try:  # Try again after 30 mins if 429
                return get()
            except Exception as e2:
                print('Another failure:', e2, file=sys.stderr)

#Function to run and output the API for specified country code
def run(wordsList, countryCode, wait, outputDirectory, pivotalWord, topic):
    #Get data for first five words
    frame = getReq(wordsList[0], countryCode, wait)

    #Initialize backup, which will keep the data directly from google
    backup = frame
    backIter = 0
    for fiveW in wordsList[1:]:
        #Dynamically select pivotalword based on most average value.
        if pivotalWord is None:
            fiveW[0] = np.sum(np.square(frame[wordsList[backIter]] - np.mean(np.mean(frame[wordsList[backIter]], axis=0)))).idxmin()
        backIter+=1

        #Get reqs
        data = getReq(fiveW, countryCode, wait)

        #If all 5 words do not return data, then the "data" dataframe will
        #be empty. This prevents the data from being merged below. Given that
        #all of the words have no data, we can just merge zero-columns for
        #each word. We create that zero-dataframe here.
        if len(data) == 0:
            data = pd.DataFrame(0, index=np.arange(len(backup)), columns=fiveW)
            data["date"] = backup["date"]

        #Add data to backup frame
        backup = pd.merge(backup, data, how='outer', on='date', suffixes=("",str(backIter)))

        #Divide columns to get ratio between old data and new data
        if (data[fiveW[0]]==0).any():
            warnings.warn("Replaced zero in pivotalword with 1.")
        dv = frame[fiveW[0]].replace([0],1) / data[fiveW[0]].replace([0],1)

        #Multiply ratio by new data to make new data proportional to old data
        data[fiveW[1:]] = data[fiveW[1:]].multiply(dv,axis='index')

        #Merge new data and old data
        frame = pd.merge(frame, data.drop(columns=fiveW[0]), how='outer', on='date')

        #Add column with country ISO
        frame["ISO"] = countryCode
        frame["topic"] = topic
        firstCols = ["date", "ISO", "topic"]
        frame = frame[firstCols+[ elem for elem in frame.columns if elem not in firstCols]]

        #Write to <output file><date>-to-<date + 1 day>.csv
        try:
            filename = "{}_{}_{}_adjusted.csv".format(topic, countryCode, TIME_FRAME.replace(' ','-to-'))
            frame.to_csv(outputDirectory+filename, index = False)
            filename = "{}_{}_{}_raw.csv".format(topic, countryCode, TIME_FRAME.replace(' ','-to-'))
            backup.to_csv(outputDirectory+filename, index = False)
        except IOError:
            sys.exit('Invalid output file or no space to write to output file')

        #Print so that you know how much data has been collected
        print('Data collection successful')

def main():

    parser = argparse.ArgumentParser(description="Google Trend Data")
    parser.add_argument("--topic", required=True, help="topic")
    parser.add_argument("--wordsfile", required=True, nargs="+",
                        help="wordsFile directory: /googleTrends/parameterFile/words_economics")
    parser.add_argument("--iso", required=True)
    parser.add_argument("--wait", required=True, help="multiplier for wait times")
    parser.add_argument("--output", required=True, help="output directory")
    parser.add_argument("--pivotalWord", help="optional pivotalWord")
    args = parser.parse_args()

    wordsList = get_wordslist(args.wordsfile)

    run(wordsList, args.iso, int(args.wait), args.output, args.pivotalWord, args.topic)

if __name__ == "__main__":
    main()