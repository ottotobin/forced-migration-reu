# fm-google-trends
gtrAPI_Daily.py - API request code that works with a Daily granularity. Only added clarifying comments

gtrAPI_Range.py - API request code that works over a five year period at a monthly granularity. Main changes from Daily to Range are:
- fixing 'isParitial' return issues from how getRequest worked
- added a columed to the data frame populate with Iraq