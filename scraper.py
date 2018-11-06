"""
Requires beautiful soup module
pip install beautifulsoup4
"""

from bs4 import BeautifulSoup
from collections import namedtuple
import requests

"""
Creates a named tuple that acts more like a class for clarity
A tuple of 3 values, each value can be referenced by index or by name
e.g. rating[0] === rating.approval
for people who haven't used before
"""
ratings = namedtuple('rating_percents', ['approval', 'disapproval', 'no_opinion'])

"""
getApproval Dict
returns a dictionary of strings of weeks mapped to rating named tuple
"""
def getApprovalList():
    #url, beautiful soup object, and the correct table
    page = requests.get("https://news.gallup.com/poll/203198/presidential-approval-ratings-donald-trump.aspx")
    soup = BeautifulSoup(page.content, "html.parser")
    table = soup.find("table", attrs={"aria-labelledby":"caption-20181105122501"})
    
    approvalList = []
    
    rows = table.find_all('tr')
    for row in rows:      
        #find the week from the row header
        week = row.find('th')
        #filters out the rows that are either year breakpoints or column headers
        if week != None and len(week.string) > 7:
            """
            list of all cells in row relating to approval, disapproval, and no 
            opinion ratings respectively
            """
            data = row.find_all('td')
            weeks_rating = ratings(data[0].string,data[1].string,data[2].string)
            approvalList.insert(0,(week.string,weeks_rating))
    return approvalList

#takes approvalList entry, converts to string to store in file
def tupleToString(lEntry):
    retStr = lEntry[0] + "\t"
    for item in lEntry[1]:
        retStr += item + "\t"
    return retStr
    
def generateRatingDataFile():
    approvalList = getApprovalList()
    
    #converted every entry to list format
    listOfStrings = [tupleToString(x) + "\n" for x in approvalList]
    
    fh = open("data/rating_data.txt", "w")    
    fh.writelines(listOfStrings)
    fh.close()