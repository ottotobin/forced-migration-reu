
"""
Author: Grey, Yiqing, Helge Marahrens, Nathan Wycoff, Lina Laghzaoui, Grace Magny-Fokam, Toby Otto, Rich Pihlstrom

Last Update: 22 June 2023
    Cleaned up some of the remaining errors. Mostly minor case errors where the first 5 words wouldn't
    return data or something.
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
    python3 gtrAPI_Range.py --wordsDir keywords/ --wait 20 --output results/

"""

TIME_FRAME = '2018-01-01 2023-05-31' #hard-coded time frame to collect data over

import sys
import argparse
import pandas as pd
import datetime
import time
from pytrends.request import TrendReq
import numpy as np
import json
import csv
import os

#Function to read in the list of search terms from the wordsfile
#Cannot use pandas read_csv function as it does not work
#with non-ASCII Arabic script
def get_wordslist(words):

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
def run(wordsList, countryCode, wait, outputDirectory, topic):
    #Get data for first five words
    frame = getReq(wordsList[0], countryCode, wait)
    
    #Sometimes the 5 words present in wordsList[0] would all not have any GTR data about them.
    #In this case, frame would be empty. This causes errors later. Because the terms don't have
    #data, we can just manually fill frame with 0-columns for each word. We also need a date
    #column to merge dfs. This chunk does all of this.
    if len(frame) == 0:
        #Create df of just date column based on global var
        now = datetime.datetime.strptime(TIME_FRAME.split()[1], "%Y-%m-%d")
        ctr = datetime.datetime.strptime(TIME_FRAME.split()[0], "%Y-%m-%d")
        date = [ctr.strftime('%Y-%m-%d')]
        while ctr <= now:
            ctr += datetime.timedelta(days=32)
            date.append(datetime.datetime(ctr.year, ctr.month, 1))
        date=date[:-1]
        frame = pd.DataFrame(data=date, columns=["date"])

        #Create column for each of initial words and set to 0
        frame[wordsList[0]] = 0

    #Initialize backup, which will keep the data directly from google
    backup = frame
    backIter = 0
    for i, fiveW in enumerate(wordsList[1:]):
        #Dynamically select pivotalword based on most average value.
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
        if 0 in data[fiveW[0]].values:
            print("        !! Warning: Replaced zero in pivotalword with 1.")
        dv = frame[fiveW[0]].replace([0],1) / data[fiveW[0]].replace([0],1)

        #Multiply ratio by new data to make new data proportional to old data
        data[fiveW[1:]] = data[fiveW[1:]].multiply(dv,axis='index')

        #Merge new data and old data
        frame = pd.merge(frame, data.drop(columns=fiveW[0]), how='outer', on='date')

        #Add columns for ISO and topic
        firstCols = ["date", "ISO", "topic"]
        if "ISO" and "topic" not in frame:
            frame["ISO"] = countryCode
            frame["topic"] = topic
            frame = frame[firstCols+[ elem for elem in frame.columns if elem not in firstCols]]
        if "ISO" and "topic" not in backup:
            backup["ISO"] = countryCode
            backup["topic"] = topic
            backup = backup[firstCols+[ elem for elem in backup.columns if elem not in firstCols]]

        #Write to <output file><date>-to-<date + 1 day>.csv
        try:
            #Structuring output such that the output directory will contain
            #a sub-directory for each of topic of keyword.

            #Make new directory if it does not exist
            if not os.path.exists(outputDirectory+topic):
                os.makedirs(outputDirectory+topic)

            filename = "{}_{}_{}_adjusted.csv".format(topic, countryCode, TIME_FRAME.replace(' ','-to-'))
            frame.to_csv(outputDirectory+topic+"/"+filename, index = False)
            filename = "{}_{}_{}_raw.csv".format(topic, countryCode, TIME_FRAME.replace(' ','-to-'))
            backup.to_csv(outputDirectory+topic+"/"+filename, index = False)

        except IOError:
            sys.exit('Invalid output file or no space to write to output file')
       
       #Print so that you know how much data has been collected
        print(' '*4+'Data collection successful ('+str(i+1)+"/"+str(len(wordsList[1:]))+")")

#Function that reads in 4 csv "basefiles" and the keyword files and converts them all into a single json.
#The reasoning for this method is to allow users to edit the foundational files and see how they are
#being collected and labelled. We then loop over this json when we run our code on all the countries and topics.
def csvToJson(directory):
    dict = {}

    # add all keyword files to list
    categories=[]
    for filename in os.listdir(directory):
        categories.append(open(os.path.join(directory, filename), "r"))

    neighbors = open("basefiles/neighboringCountries.csv", "r") #neighboring countries
    translations = open("basefiles/translation.csv") #translations

    # helper func that checks if a phrase needs a destination
    # or a origin appended to it
    def checkPlaceType(phrase):
        with open("basefiles/appendPhrases.csv", "r") as appends:
            for line in csv.reader(appends):
                if line[0] == phrase:
                    return line[1]
            return "none"

    # arranges the 'topics' subdictionary
    dict["topics"] = {}

    #For each topic, we want to read the keywords, check if they need
    #either a destination or origin appended (or no appending)
    for topic in categories:
        tpc = topic.name.split("/")[-1]
        dict["topics"][tpc] = {
            "destination":[],
            "origin":[],
            "none":[]
        }
        for line in topic.readlines():
            l=line.strip()
            if l not in dict["topics"][tpc][checkPlaceType(l)]:
                dict["topics"][tpc][checkPlaceType(l)] += [l]
        topic.close()

    # helper func that returns the 2-char and 3-char ISO codes
    # for a given country name
    def checkISO(name):
        with open("basefiles/ISO.csv", "r") as iso_codes:
            for line in csv.reader(iso_codes):
                if line[0] == name:
                    return [line[1],line[2]]
        return []
            
    # arranges the "countries" subdictionary
    dict["countries"] = {}

    #For each conutry, populate it's dictionary entry
    #with information such as ISO, name, neighboring countries, and translation
    for line in translations.readlines()[1:]:
        name = line.split(",")[0]
        ISO_3 = checkISO(name)[1]
        dict["countries"][ISO_3]={}
        dict["countries"][ISO_3]["name"] = name
        dict["countries"][ISO_3]["2-char ISO"] = checkISO(name)[0]
        
        #Append neighboring countries (if applicable)
        neighbor_reader=csv.reader(neighbors)
        for lne in neighbor_reader:
            if lne[0] == name:
                neighbor_countries = lne[1].split("|")
                dict["countries"][ISO_3]["neighbor ISO"]=[]
                for neighbor in neighbor_countries:
                    if len(checkISO(neighbor)) > 0:
                        dict["countries"][ISO_3]["neighbor ISO"].append(checkISO(neighbor)[1])
        neighbors.seek(0)

        translated_name = line.split(",")[1]
        dict["countries"][ISO_3]["translation"] = translated_name.strip()
    translations.close()
    neighbors.close()

    return dict

#This function loops over the json created earlier and returns a
#list of dictionaries where each dictionary contains the parameters
#for a unique iteration
def getArgsList(jsonName):
    retList = []

    with open(jsonName, "r") as f:

        jsonObj = json.load(f) #load json

        #Every county in the neighboring country file is present as a dictionary entry in
        #the json, but only a subset of those countries function as an origin country. Any
        #origin country will have a list of neighboring countries, so we can check this way
        for originISO in jsonObj["countries"]:
            if "neighbor ISO" in jsonObj["countries"][originISO]:

                for topic in jsonObj["topics"]:
                    keywordList=jsonObj["topics"][topic]["none"]
                    
                    #If the keyword needs an origin appended, we append that here
                    for originPhrase in jsonObj["topics"][topic]["origin"]:
                        originTranslation = jsonObj["countries"][originISO]["translation"]
                        keywordList.append(originPhrase +" "+ originTranslation)

                    #If the keyword needs a destination appended, we append that here
                    for destinationPhrase in jsonObj["topics"][topic]["destination"]:
                        for neighborISO in jsonObj["countries"][originISO]["neighbor ISO"]:
                            destinationTranslation = jsonObj["countries"][neighborISO]["translation"]
                            keywordList.append(destinationPhrase +" "+ destinationTranslation)

                    #Strip each word
                    for i,word in enumerate(keywordList):
                        keywordList[i] = word.strip()

                    #Pass the keyword list to the function that produces the
                    #list that we use in our actual function run
                    keywordList = get_wordslist(keywordList)

                    argsDict = {
                        "wordList":keywordList,
                        "ISO":jsonObj["countries"][originISO]["2-char ISO"],
                        "topic":topic
                    }

                    retList.append(argsDict)
    
    return retList
                    
def main():
    parser = argparse.ArgumentParser(description="Google Trend Data")
    parser.add_argument("--wordsDir", required=True, nargs="+",
                        help="wordsDir directory: keywords/")
    parser.add_argument("--wait", required=True, help="multiplier for wait times")
    parser.add_argument("--output", required=True, help="output directory")
    args = parser.parse_args()

    # Create consolidating json and output
    with open("dataDict.json", "w+") as o:
        json.dump(csvToJson(args.wordsDir[0]), o, indent=2, ensure_ascii=False)
    
    # Get list of arguments to be passed
    argsList = getArgsList("dataDict.json")

    # Run for each iteration
    t0 = time.time()
    for i, argDict in enumerate(argsList):
        print("Country: "+ argDict["ISO"]+"; Topic: "+argDict["topic"]+\
              "; Iteration: "+str(i+1)+"/"+str(len(argsList)))
        t1 = time.time()

        if i >= 51:
            run(argDict["wordList"], argDict["ISO"], int(args.wait), args.output, argDict["topic"])

        print("Iteration Run Time: "+str(time.time()-t1)[:5]+" sec")
        print("Total Run Time: "+str((time.time()-t0)/60)[:5]+" min\n")

if __name__ == "__main__":
    main()