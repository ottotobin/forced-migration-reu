# fm-google-trends-arabic

# authors: Rich, Toby, Lena, Grace

# Directory and File Descriptors
- results: directory of outputed .csv files from gtrAPI_Daily.py and gtrAPI_Range.py
- basefiles: includes 4 .csv files
    - appendPhrases.csv: (phrase, label); holds all the keyword phrases that need an origin or destination appended to it
    - ISO.csv: (country, 2-char, 3-char); contains all countries with their 2-char and 3-char ISO codes
    - neighboringCountries.csv: (country, (neighor1|neighbor2|..)); contains all arabic speaking countries and their neighboring countries
    - translation.csv: (country, translation); contains a list of countries and their translated name
- gtrAPI_Range.py: API requester modified to work over a 5 year period w/ a monthly granularity
- dicFormat.json: json file describing our mass data dictionary
- gtrAPI_Daily.py: original API requester that worked on a daily granularity. (aside from comments, this file has not been modified by our branch)
- keywords: includes 4 .txt. files
    - generic_terms: generic term keywords in arabic
    - geography: geography term keywords in arabic
    - safe_places: safe place keywords in arabic
    - travel: travel term keywords in arabic

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
    python3 gtrAPI_Range.py --topic generic --wordsfile generic_terms --iso IQ --wait 2 --output results