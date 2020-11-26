import sys
import csv
from info.general import FILE_PATH_CORE

class computeRt():
        
    INPUT_FILE = None
    OUTPUT_FILE = None
    OUTPUT_ERROR = None

    CONFIRMED_THRESHOLD = 20

    PAST_DATE_LIST = [0, 6, 12, 18]

    #Retrieve dataset being used from command line
    DATASET = sys.argv[0][-5:-3]

    def readFile(self):
        """
        Takes in the contents of the file, and compiles it into
        a usable format

        INPUT:
            NONE
        
        OUTPUT:
            returns a list of dictionaries, containing the contents of the dataset
        """
        processedData = []

        with open(f"{FILE_PATH_CORE}{self.DATASET}{self.INPUT_FILE}", "r") as dataFile:
            myReader = csv.DictReader(dataFile)

            for row in myReader:
                processedData.append(row)

        return processedData
    
    def getRegion(self, currRow):
        return -1
    
    def main(self):
        """
        Over the course of its functions, it will calculate the Rt for each day,
        then add them as a new column to the dataset, and save it to a new .csv file.

        INPUT:
            NONE
        
        OUTPUT:
            returns nothing, but creates a new .csv file in the folder /Rt/ of the given
            dataset

        """
        optDataList = []
        oldLabel = None

        processedData = self.readFile()

        newData = []
        currRegion = self.getRegion(processedData[0])

        for currRowIndex in range(1, len(processedData)-1):
            lineRegion = self.getRegion(processedData[currRowIndex])