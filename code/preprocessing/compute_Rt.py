"""
Computes the Rt throughout the given dataset
"""

import sys
import csv
import statistics

from copy import deepcopy

from scipy.stats import gamma
from shared.sharedVariables import FILE_PATH_CORE
from shared.sharedFunctions import createErrorFile
from shared.sharedFunctions import printProgressBar


class computeRt():
        
    INPUT_FILE = None
    OUTPUT_FILE = None
    OUTPUT_ERROR = None

    CONFIRMED_THRESHOLD = 20

    PAST_DATE_LIST = [0, 6, 12, 18]

    #In accordance to reports by multiple papers
    RZERO = 2.3

    #Retrieve dataset being used from command line
    DATASET = sys.argv[0][-5:-3]

    CONTROL_MEASURES = {
        "Meeting Friends/Family (Indoor)" : ("Trinary", ["Low", "Moderate", "High"]),
        "Meeting Friends/Family (Outdoor)" : ("Trinary", ["Low", "Moderate", "High"]),
        "Domestic Travel": ("Trinary", ["Low", "Moderate", "High"]),
        "International Travel": ("Trinary", ["Low", "Moderate", "High"]),
        "Cafes and Restaurants": ("Trinary", ["Low", "Moderate", "High"]),
        "Pubs and Bars": ("Trinary", ["Low", "Moderate", "High"]),
        "Sports and Leisure" : ("Trinary", ["Low", "Moderate", "High"]),
        "School Closure" : ("Binary", ["Opened", "Closed"]),
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

    def writeFile(self, dataset, fileName):
        pass
    
    def computeGs(self, s):
        """
        Computes the Gamma distribution with mean 7 and standard deviation of 4.5

        INPUT:
            :param s: Integer, the result of (number of days since records began) - (number in range 1 -> the first number)
        
        OUTPUT:
            returns the resultant value of Gamma distribution, or, if an error occurs, nothing and the program ends

        Code by Xiuyi Fan
        """
        # These two values gives Gamma distribution with mean 7, standard deviation 4.5
        a = 2.412
        scale = 2.898

        #    a = 10.84
        #    scale = 0.64

        if s == 0:
            print("Error: s == 0 in computeGs")
            sys.exit(-1)

        if s == 1:
            return gamma.cdf(1.5, a, 0, scale)
        else:
            returnVal = gamma.cdf(s+0.5, a, 0, scale) - gamma.cdf(s-0.5, a, 0, scale)
            return returnVal

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
                        
                        if typeOfControlMeasure[0] is "Binary" and val >= 1:
                            processedData[currRowIndex][currControlMeasure] = str(self.PAST_DATE_LIST.index(currDateReduction) + val)
                        
                        elif typeOfControlMeasure[0] is "Trinary":
                            trinaryValues = deepcopy(typeOfControlMeasure[1])

                            val-=1

                            #print(trinaryValues[val])                   
                            processedData[currRowIndex].update({f"{currControlMeasure} ({trinaryValues[val]})" : str(self.PAST_DATE_LIST.index(currDateReduction) + 1)})
                            trinaryValues.pop(val)

                            for currOption in trinaryValues:
                                processedData[currRowIndex][f"{currControlMeasure} ({currOption})"] = 0   

            nextRegion = self.getRegion(processedData[currRowIndex + 1])

            if currRegion != nextRegion:
                continue
            

            newRow = deepcopy(processedData[currRowIndex])

            for currControlMeasure in self.CONTROL_MEASURES:
                if self.CONTROL_MEASURES.get(currControlMeasure)[0] is "Trinary":
                    newRow.pop(currControlMeasure)

            prevCaseCount = int(processedData[currRowIndex - 1].get("Cases"))
            currCaseCount = int(processedData[currRowIndex].get("Cases"))
            nextCaseCount = int(processedData[currRowIndex + 1].get("Cases"))

            newRow["Cases"] = int((prevCaseCount + currCaseCount + nextCaseCount) / 3)

            newData.append(newRow)
        #Log the current dataset before further processing
        self.writeFile(newData, "/Rt/after_control_measures.csv")

        prevRegion = None
        prevConfirmed = 0
        tauZeroRow = 0
        prevRt = 0
        #Calculate the Rt
        for rowInd in range(len(newData)):
            Rt = self.RZERO
            row = newData[rowInd]           

            currRegion = self.getRegion(row)
            currConfirmed = row.get("Cases")
            
            printProgressBar(rowInd, len(newData))

            if currRegion != prevRegion:
                tauZeroRow = rowInd
                prevConfirmed = 0
                prevRt = 0

            # t - Days since first record (for region)   
            currDayNo = rowInd - tauZeroRow

            if currDayNo > 1:
                # ct
                newInfections = currConfirmed - prevConfirmed
                
                temp = 0
                
                for tau in range(1, currDayNo):

                    casesTau = int(newData[tauZeroRow + tau].get('Cases')) - int(newData[tauZeroRow + tau - 1].get("Cases"))

                    """if casesTau < 0:
                        errorList = []
                        errorList.append("----Current----")
                        errorList.append(f"Day: {newData[rowInd]['Day']}")
                        errorList.append(f"Region: {self.getRegion(newData[rowInd])}")
                        errorList.append(f"Cases: {newData[rowInd]['Cases']}")
                        errorList.append(f"Case Tau: {casesTau}")
                        errorList.append(f"Temp: {temp}")

                        errorList.append("\n----Previous----")
                        errorList.append(f"Day: {newData[rowInd-1]['Day']}")
                        errorList.append(f"Region: {self.getRegion(newData[rowInd-1])}")
                        errorList.append(f"Cases: {newData[rowInd-1]['Cases']}")

                        errorList.append("\n----TauZeroRow + Tau----")
                        errorList.append(f"Day: {newData[tauZeroRow + tau]['Day']}")
                        errorList.append(f"Region: {self.getRegion(newData[tauZeroRow + tau])}")
                        errorList.append(f"Cases: {newData[tauZeroRow + tau]['Cases']}")

                        errorList.append("\n----TauZeroRow + Tau -1 ----")
                        errorList.append(f"Day: {newData[tauZeroRow + tau - 1]['Day']}")
                        errorList.append(f"Region: {self.getRegion(newData[tauZeroRow + tau - 1])}")
                        errorList.append(f"Cases: {newData[tauZeroRow + tau - 1]['Cases']}")



                        createErrorFile(FILE_PATH_CORE, self.DATASET, 1, errorList)
                        sys.exit(0)"""

                    gs = self.computeGs(currDayNo - tau)
                    
                    temp += casesTau * gs

                if temp > 0:
                    Rt = newInfections / temp
                else:
                    Rt = prevRt
            else:
                Rt = 2.3
            
            #Since the R number can't go below 0, correct
            if Rt < 0:
                Rt = 0
                prevRt = 0

            #Add the Rt to the new dataset
            row.update({"Rt" : str(Rt)})    
            newRow = deepcopy(row)
            newRow["Cases"] = currConfirmed - prevConfirmed
            optDataList.append(newRow)

            prevRegion = currRegion
            prevConfirmed = currConfirmed

            #Go through dataset and convert Rt to the middle value in relation to its surrounding days
            for rowInd in range(1, len(optDataList) - 1):
                prevRt = float(optDataList[rowInd - 1].get("Rt"))
                currRt = float(optDataList[rowInd].get("Rt"))
                nextRt = float(optDataList[rowInd + 1].get("Rt"))

                median = statistics.median([prevRt, currRt, nextRt])
                optDataList[rowInd]["Rt"] = str(median)
            
        filteredData = []

        for rowInd in range(len(optDataList)):
            if int(newData[rowInd].get("Cases")) < self.CONFIRMED_THRESHOLD:
                continue
            filteredData.append(optDataList[rowInd])

        optDataList = filteredData

        self.writeFile(optDataList, self.OUTPUT_FILE)