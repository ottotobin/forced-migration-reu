# fm-google-trends-arabic

# Authors: Lina Laghzaoui, Grace Magny-Fokam, Toby Otto, Rich Pihlstrom

# Directory and File Descriptors
- Daily: directory for running the daily version of the GTR code.
    - gtrAPI_Daily.py: original API requester that worked on a daily granularity. (aside from comments, this file has not been modified by our branch)
    - generic_terms, geography, info_of_destination, means_of_travel: keyword files with origins and destinations included relative to our country of focus, Iraq.
- Range: directory for running the time range version of the GTR code.
    - basefiles: directory of csv files. Users should only have to edit these csv files for the code to run for their respective country/language of focus.
        - ISO.csv: a file containing a list of every country and their associated 2-letter and 3-letter ISO codes. This is a universal file, so nothing should need to be changed except for if a smaller territory (e.g. W. Sahara) is referenced and one needs to manually add it's ISO because it is not already included. This was produced using ChatGPT.
        - neighboringCountries: a file containing a list of neighboring countries for each "origin" country that we are focusing on for our data collection. This file was created by copy-pasting the Google Sheets document into ChatGPT.
        - translation.csv: a file containing the translation of each "destination" and "origin" country that would need to be appended onto phrases such as "move to." This list allows us to just concatenate a phrase with an "origin" or "destination" country rather than needing to translate every permutation of the phrase with the country.
        - appendPhrases.csv: a file containing a list of phrases that need either an "origin" country (e.g. "flee from") or a "destination" country (e.g. "move to") appended.
    - keywords: includes 4 .txt. files.
        - generic_terms: generic term keywords in Arabic
        - geography: geography term keywords in Arabic
        - safe_places: safe place keywords in Arabic
        - travel: travel term keywords in Arabic
    - results: includes a subdirectory with the same names as the files in the keywords directory.
        - generic_terms: directory of results for each country for the generic_terms topic.
        - geography: directory of results for each country for the geography topic.
        - safe_places: directory of results for each country for the safe_places topic.
        - travel: directory of results for each country for the travel topic.
    - dataDict.json: contains a dictionary produced by gtrAPI_Range.py that has all of the phrases (without appendings) for each topic and relevant country information for all of the "origin" and "destination" countries considered. Outputting this file allows the user to confirm that their phrases and countries are being organized correctly and makes it clear how the gtrAPI_Range.py program loops through each country and topic.

# Running Daily Script
- terminal command:
    python3 gtrAPI_Daily.py --topic generic --wordsfile generic_terms --iso IQ --wait 2 --output results
- paramenter definitions
    --topic: topic of interest
    --wordsfile: input file name
    --iso: 2-char ISO code for country of interest
    --wait: time multiplier between API calls
    --output: output directory name 

# Running Range Script
- terminal command:
    python3 gtrAPI_Range.py --wordsDir keywords/ --wait 20 --output results/
- paramenter definitions
    --wordsDir: name of directory that contains the keyword files
    --wait: time multiplier between API calls
    --output: output directory name 
