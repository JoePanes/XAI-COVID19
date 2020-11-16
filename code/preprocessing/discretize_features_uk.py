import csv
import sys

SEVERITY = {
    "low" : 1,
    "moderate" : 2,
    "high" : 3,
}

CLOSURE = {
    "opened" : 0,
    "closed" : 1,
}

REGIONS = {
    "east midlands" : 1,
    "east of england" : 2,
    "london" : 3,
    "north east" : 4,
    "north west" : 5,
    "south east" : 6,
    "south west" : 7,
    "west midlands" : 8,
    "yorkshire and the humber" : 9,
    "northern ireland" : 10,
    "scotland" : 11,
    "wales" : 12,
}

fileName = "../../data/uk_data.csv"
optFileName = "../../data/uk_mlTable_0_0.csv"

def main():
    """
    Converts the UK dataset into a format that is usuable in the next stage

    INPUT:
        NONE
    
    OUTPUT:
        returns nothing, but creates writes the converted UK dataset to another .csv file
    """
    newLabel = [
        'Day', 'Date', 'Regions', 'Cases', 'Cumulative cases',
        'Deaths', 'Cumulative Deaths', 'Tests', 'Cumulative Tests',
        'Meeting Friends/Family', 'Domestic Travel', 'Cafes and restaurants',
        'Pubs and bars', 'Sports and leisure', 'Schools Closure'
    ]
    #readLabel = True
    
    rawData = []


    with open(fileName, "r") as dataFile:
        myReader = csv.DictReader(dataFile)

        for row in myReader:
            """if readLabel:
                readLabel = False
                
                newLabel = list(row.keys())
                continue"""

            rawData.append(row)
            

if __name__ == '__main__':
    sys.exit(main()) 