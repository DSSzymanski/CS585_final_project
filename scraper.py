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

#url, beautiful soup object, and the correct table
page = requests.get("https://news.gallup.com/poll/203198/presidential-approval-ratings-donald-trump.aspx")
soup = BeautifulSoup(page.content, 'html.parser')
table = soup.find('table', attrs={"aria-labelledby":"caption-20181105122501"})

"""
getApproval Dict
returns a dictionary of strings of weeks mapped to rating named tuple
"""
def getApprovalDict():
    approvalDict = {}
    
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
            approvalDict[week.string] = weeks_rating
    return approvalDict

print(getApprovalDict())