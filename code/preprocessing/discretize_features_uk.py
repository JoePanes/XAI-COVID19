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
        returns nothing, but, writes the converted UK dataset to the output .csv file
    """
    newLabel = {
        'Day' : "Day",
        'Date' : "Date",
        'Regions' : "Regions",
        'Cases' : "Cases",
        'Cumulative Cases' : "Cumulative Cases",
        'Deaths' : "Deaths", 
        'Cumulative Deaths' : "Cumulative Deaths", 
        'Tests' : "Tests", 
        'Cumulative Tests' : "Cumulative Tests", 
        'Meeting Friends/Family' : "Meeting Friends/Family",
        'Domestic Travel' : "Domestic Travel", 
        'Cafes and Restaurants' : "Cafes and Restaurants",
        'Pubs and Bars' : "Pubs and Bars", 
        'Sports and Leisure' : "Sports and Leisure", 
        'Schools Closure' : "Schools Closure",
    }
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

    with open(optFileName, "w") as optFile:
        myWriter = csv.DictWriter(optFile, list(newLabel.keys()))
        #Put the column labels in
        myWriter.writerow(newLabel)

        for row in rawData:
            #Copy over info from the original, and discretize where relevant
            newLine = {}

            newLine["Day"] = row["Day"]

            """
            TO DO:
                Discretize the date?
            """
            newLine["Date"] = row["Date"]

            newLine["Regions"] = REGIONS[row["Regions"].lower()]

            newLine["Cases"] = row["Cases"]
            newLine["Cumulative Cases"] = row["Cumulative Cases"]
            
            newLine["Deaths"] = row["Deaths"]
            newLine["Cumulative Deaths"] = row["Cumulative Deaths"]
            
            newLine["Tests"] = row["Tests"]
            newLine["Cumulative Tests"] = row["Cumulative Tests"]

            newLine["Meeting Friends/Family"] = SEVERITY[row["Meeting Friends/Family"].lower()]

            newLine["Domestic Travel"] = SEVERITY[row["Domestic Travel"].lower()]

            newLine["Cafes and Restaurants"] = SEVERITY[row["Cafes and Restaurants"].lower()]

            newLine["Pubs and Bars"] = SEVERITY[row["Pubs and Bars"].lower()]

            newLine["Sports and Leisure"] = SEVERITY[row["Sports and Leisure"].lower()]

            newLine["Schools Closure"] = CLOSURE[row["Schools Closure"].lower()]

            myWriter.writerow(newLine)
            

if __name__ == '__main__':
    sys.exit(main()) 