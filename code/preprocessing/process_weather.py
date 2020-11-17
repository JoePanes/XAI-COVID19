import csv
import sys

filePath = "../../data/weather/raw/"
optFilePath = "../../data/weather/processed/"

def main():
    """
    Goes through the weather files present within the raw folder, and outputs the
    avg Temp and Humidity for each day.

    INPUTS:
        NONE
    
    OUTPUTS:
        returns nothing, but, it converts the .csv files in raw into the expected format
            and outputs them in processed
    """
    rawData = []
    with open(filePath + "uk.csv","r") as dataFile:
        #Skip over rows that are not part of the database
        for _ in range(6):
            next(dataFile)

        myReader = csv.DictReader(dataFile)
        
        for row in myReader:
            #Reverse the order of the data to match the EU dataset
            rawData.insert(0, row)
        
    

if __name__ == '__main__':
    sys.exit(main()) 