import sys
import csv
from shared.sharedVariables import FILE_PATH_CORE
from shared.sharedFunctions import removeBrackets
from copy import deepcopy

class computeRt():
        
    INPUT_FILE = None
    OUTPUT_FILE = None
    OUTPUT_ERROR = None

    CONFIRMED_THRESHOLD = 20

    PAST_DATE_LIST = [0, 6, 12, 18]

    #Retrieve dataset being used from command line
    DATASET = sys.argv[0][-5:-3]

    CONTROL_MEASURES = {
        "Meeting Friends/Family (Indoor)" : "Trinary",
        "Meeting Friends/Family (Outdoor)" : "Trinary",
        "Domestic Travel": "Trinary",
        "International Travel": "Trinary",
        "Cafes and Restaurants": "Trinary",
        "Pubs and Bars": "Trinary",
        "Sports and Leisure" : "Trinary",
        "School Closure" : "Binary",
    }

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

    def writeFile(self, dataset):
        pass

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

            #When a new regions occurs, replace and skip over the first instance
            if lineRegion != currRegion:
                currRegion = lineRegion
                continue
            
            #Look at previous dates increasingly further back
            for currDateReduction in self.PAST_DATE_LIST:
                previousDateIndex = currRowIndex - currDateReduction

                pastDateRegion = self.getRegion(processedData[previousDateIndex])
                
                if previousDateIndex > 0 and pastDateRegion == currRegion:
                    for currControlMeasure in self.CONTROL_MEASURES:
                        val = int(processedData[previousDateIndex].get(currControlMeasure))
                        typeOfControlMeasure = self.CONTROL_MEASURES.get(currControlMeasure)
                        
                        if typeOfControlMeasure is "Binary" and val >= 1:
                            processedData[currRowIndex][currControlMeasure] = str(self.PAST_DATE_LIST.index(currDateReduction) + val)            
            nextRegion = self.getRegion(processedData[currRowIndex + 1])

            if currRegion != nextRegion:
                continue
            

            newRow = deepcopy(processedData[currRowIndex])

            prevCaseCount = int(processedData[currRowIndex - 1].get("Cases"))
            currCaseCount = int(processedData[currRowIndex].get("Cases"))
            nextCaseCount = int(processedData[currRowIndex + 1].get("Cases"))

            newRow["Cases"] = int((prevCaseCount + currCaseCount + nextCaseCount) / 3)

            newData.append(newRow)

        self.writeFile(newData)