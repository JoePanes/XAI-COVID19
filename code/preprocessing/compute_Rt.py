"""
Parent class of compute_Rt_uk and compute_Rt_eu

Can currently only take in the UK dataset due to that being at a completed level, the EU dataset is
missing a lot of data, therefore is unusable.

It is able to take in a dataset, that has been processed by process_features and its corresponding child class.
Then, it processes the control measures and breaks down any non-binary control measures by splitting them up so that they are binary (each level is its own column).
Next, it calculates the Rt based upon the change in cases numbers rather than the case numbers themselves.
Finally, it writes the results of the new dataset with all the extra processing, into a new .csv file.
"""

import sys
import csv
import statistics
import multiprocessing
import os

from copy import deepcopy
from scipy.stats import gamma
from itertools import chain

from shared.sharedVariables import FILE_PATH_CORE
from shared.sharedFunctions import createErrorFile
from shared.sharedFunctions import printProgressBar
from shared.sharedFunctions import readFile


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

    REGIONAL_FIELD_NAME = None

    NON_CONTROL_MEASURE_FIELDS = []
    
    def getRegion(self, currRow):
        return -1

    def writeFile(self, dataset, fileName, containRt = False):
        pass

    def getRegionalLabel(self):
        """
        Simply return what the regional fieldname is called

        INPUT:
           NONE

        OUTPUT:
            returns the the regional fieldname
        """
        
        return self.REGIONAL_FIELD_NAME

    def orderFields(self, labels, containRt):
        """
        After the splitting up of the Trinary fields, it results in a rather oddly ordered dataset,
        this will put everything in the same order as is expected.

        INPUT:
            :param labels: List of Strings, the current order of fields within the dataset
            :param containRt: Boolean, whether Rt needs to be added to the dataset labels

        OUTPUT:
            returns the new order for the dataset to be written to a file with
        """
        
        desiredOrder = ["Cases", "Cumulative Cases", "Deaths", "Cumulative Deaths",  "Tests", "Cumulative Tests", "Temperature", "Humidity"]
        
        #Add fields that are specific to the dataset, but not control measures
        for fieldToAdd, fieldToInsertAfter in self.NON_CONTROL_MEASURE_FIELDS:
            insertionIndex = desiredOrder.index(fieldToInsertAfter)
            desiredOrder.insert(insertionIndex + 1, fieldToAdd)

        regionalLabel = self.getRegionalLabel()

        insertIndexs = []      
        controlMeasureIndex = None
        
        #Find the indexs to insert non-regular measures
        for currIndex in range(len(desiredOrder)):
            if desiredOrder[currIndex] is "Cases":
                insertIndexs.append((regionalLabel, currIndex))
            
            elif desiredOrder[currIndex] is "Temperature":
                insertIndexs.append(("Control Measures", currIndex))

        #Insert unique labels into the labels, adjusting index as it goes 
        adjustment = 0
        for name, index in insertIndexs:
            if name is "Control Measures":
                controlMeasureIndex = index + adjustment
            else:
                desiredOrder.insert(index + adjustment, name)
            adjustment += 1
        
        for currControlMeasure in self.CONTROL_MEASURES:
            controlType = self.CONTROL_MEASURES.get(currControlMeasure)

            if controlType[0] is "Trinary":
                for currLevel in controlType[1]:
                    desiredOrder.insert(controlMeasureIndex, f"{currControlMeasure} ({currLevel})")
                    controlMeasureIndex += 1
            
            elif controlType[0] is "Binary":
                desiredOrder.insert(controlMeasureIndex, currControlMeasure)
                controlMeasureIndex += 1

        if containRt:
            desiredOrder.extend(["Rt"])

        #Based on https://stackoverflow.com/a/52044835        
        reorderedLabels = {k : labels[k] for k in desiredOrder}

        return reorderedLabels

    def orderDatasetSpecificFields(self, desiredOrder):
        return []

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

    def processControlMeasures(self, processedData):
        """
        Look at the Control measures used, and evalutate the length that have been implemented for, 
        and provide a value.

        INPUT:
            :param processedData: List of dictionaries, where each entry is a row from the datasetc

        OUTPUT:
            returns the new altered version of the dataset
        """
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
                        
                        if typeOfControlMeasure[0] is "Binary":
                            if int(processedData[currRowIndex][currControlMeasure]) == 0:
                                continue
                            elif val >= 1:
                                processedData[currRowIndex][currControlMeasure] = str(self.PAST_DATE_LIST.index(currDateReduction) + val)
                            
                        
                        elif typeOfControlMeasure[0] is "Trinary":
                            trinaryValues = typeOfControlMeasure[1]

                            currLevel = int(processedData[currRowIndex][currControlMeasure])
                            
                            #Convert the level of the control measure into 3 binary fields
                            currDateControlLevel = [1]*currLevel + [0]*(3 - currLevel)

                            prevDateControlLevel = [1]*val + [0]*(3 - val)

                            for currIndex in range(len(currDateControlLevel)):

                                currField = f"{currControlMeasure} ({trinaryValues[currIndex]})"
                                
                                if currDateControlLevel[currIndex] == 1 and currDateControlLevel[currIndex] == prevDateControlLevel[currIndex]:    
                                    try:
                                        #Due to the breaking up of fields, the val used in binary is not usable for the same purpose here,
                                        #since we are not overwriting the same field
                                        
                                        prevDateTrinaryVal = int(processedData[previousDateIndex].get(currField))
                                    except:
                                        #If this is the case, then it is early on in the converting of this region
                                        prevDateTrinaryVal = 1
 
                                    processedData[currRowIndex].update({currField : str(self.PAST_DATE_LIST.index(currDateReduction) + prevDateTrinaryVal)})
                                else:
                                    processedData[currRowIndex].update({currField : 0})

                        else:
                            processedData[currRowIndex][currControlMeasure] = str(0)                            

            nextRegion = self.getRegion(processedData[currRowIndex + 1])

            if currRegion != nextRegion:
                continue
            

            newRow = deepcopy(processedData[currRowIndex])

            for currControlMeasure in self.CONTROL_MEASURES:
                if self.CONTROL_MEASURES.get(currControlMeasure)[0] is "Trinary":
                    newRow.pop(currControlMeasure)

            prevCaseCount = int(processedData[currRowIndex - 1].get("Cumulative Cases"))
            currCaseCount = int(processedData[currRowIndex].get("Cumulative Cases"))
            nextCaseCount = int(processedData[currRowIndex + 1].get("Cumulative Cases"))

            newRow["Cumulative Cases"] = int((prevCaseCount + currCaseCount + nextCaseCount) / 3)

            newData.append(newRow)
        
        return newData

    def calculateRt(self, newData):
        """
        Calculates the R value for each day within the dataset

        INPUT:
            :param newData: List of dictionaries, where each entry is a row of the dataset
        
        OUTPUT:
            returns the dataset with the newly added Rt column
        """
        optDataList = []

        prevRegion = None
        prevConfirmed = 0
        tauZeroRow = 0
        prevRt = 0

        print("---------------------")
        print(f"pid      | {os.getpid()}")
        print(f"regionNo | {self.getRegion(newData[0])}")
        print("---------------------")
        
        errorList = [] #remove
        #Calculate the Rt
        for rowInd in range(len(newData)):
            Rt = self.RZERO
            row = newData[rowInd]           

            currRegion = self.getRegion(row)
            currConfirmed = int(row.get("Cumulative Cases"))       
            
            #printProgressBar(rowInd, len(newData))

            if currRegion != prevRegion:
                tauZeroRow = rowInd
                prevConfirmed = 0

            # t - Days since first record (for region)   
            currDayNo = rowInd - tauZeroRow

            if currDayNo > 1:
                # ct
                newInfections = currConfirmed - prevConfirmed
                
                temp = 0
                for tau in range(1, currDayNo):

                    casesTau = int(newData[tauZeroRow + tau].get('Cumulative Cases')) - int(newData[tauZeroRow + tau - 1].get("Cumulative Cases"))
                    
                    #If this is the case, then there is an issue in the data
                    if casesTau < 0:
                        errorList.append("########################")
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

                    gs = self.computeGs(currDayNo - tau)
                    
                    temp += casesTau * gs

                    if temp != 0 and newInfections != 0:
                        Rt = float(newInfections / temp)
                    else:
                        Rt = 0
            else:
                if newData[rowInd]['Cumulative Cases'] == 0:
                    Rt = 0
                else:
                    Rt = 2.3

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
        
        if len(errorList) > 0:
            createErrorFile(FILE_PATH_CORE, self.DATASET, len(errorList), errorList)
        #print("\n")
        return optDataList
            
    def filterDate(self, optDataList):
        """
        Filters the data so that only days that have above the CONFIRMED_THRESHOLD are present
        in the outputted dataset

        INPUTS:
            :param optDataList: List of dictionaries, where each entry is a row from the dataset (post calculateRt version)
        
        OUTPUT:
            returns the dataset containing only the entries that exceed the threshold
        """
        filteredData = []

        for rowInd in range(len(optDataList)):
            
            if int(optDataList[rowInd].get("Cumulative Cases")) < self.CONFIRMED_THRESHOLD:
                continue
            filteredData.append(optDataList[rowInd])

        return filteredData
    
    def getRegionalIndexs(self, dataset):
        """
        From the dataset, get the start and end indexes for each region

        INPUT:
            :param dataset: List of Dictionaries, where each entry is a row from the dataset
        
        OUTPUT:
            returns a list of tuples, which specify the start and end index range of each region in the
            dataset
        """
        currStartIndexNo = None
        currRegion = None
        regionalRangeList = []
        for currIndex in range(len(dataset)):
            
            if currRegion != self.getRegion(dataset[currIndex]):
                if currRegion != None:
                    regionalRangeList.append((currStartIndexNo, currIndex-1))
                
                currRegion = self.getRegion(dataset[currIndex])
                currStartIndexNo = currIndex
            
        
        #It misses (due to the if) one region at the end, therefore, add it on
        currIndex = regionalRangeList[-1][1]
        regionalRangeList.append((currIndex + 1, len(dataset)-1))

        return regionalRangeList

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

        processedData = readFile(self.DATASET, f"{FILE_PATH_CORE}{self.DATASET}{self.INPUT_FILE}")

        newData = self.processControlMeasures(processedData)

        #Log the current dataset before further processing
        self.writeFile(newData, "/2. Rt/after_control_measures.csv")

        regionalIndexList = self.getRegionalIndexs(newData)
        
        splitList = []
        
        for currStartIndex, currEndIndex in regionalIndexList:
            splitList.append(newData[currStartIndex:currEndIndex])

        numProcesses = 10

        with multiprocessing.Pool(numProcesses) as p:
            result = p.map(self.calculateRt, splitList)

        for currRegionalList in result:
            for row in currRegionalList:
                optDataList.append(row)
       #optDataList = self.calculateRt(newData)

        optDataList = self.filterDate(optDataList)
        
        """optRegionalIndex = self.getRegionalIndexs(optDataList)
        results = []
        for currIndex in range(len(optRegionalIndex)):
            optStart, optEnd = optRegionalIndex[currIndex]
            orgStart, orgEnd = regionalIndexList[currIndex]
            results.insert(0, [(33 - currIndex), f"{orgEnd - orgStart}", f"{optEnd - optStart}"])
        
        with open(FILE_PATH_CORE + self.DATASET + "/Rt/test.csv", "w") as optFile:
            labels = ["Region No", "Original Size", "New Size"]

            myWriter = csv.writer(optFile, labels)
            myWriter.writerow(labels)
            for row in results:
                myWriter.writerow(row)"""
        self.writeFile(optDataList, self.OUTPUT_FILE, True)

if __name__ == '__main__':
    #In the event that the user runs this file
    inp = input("Do you wish to process either the EU or UK dataset? ")

    os.system(f"python compute_Rt_{inp.lower()}.py")