import matplotlib.pyplot as plt
import numpy as np
import csv
import sys

from sklearn.preprocessing import KBinsDiscretizer
from datetime import datetime
from shared.sharedFunctions import readFile

FILE_PATH = "../../data/core/"
BIN_NO = 8

def getInformationAboutDiscretizer(discreteData, originalData):
    """
    Using the result of the discretized data, provide some information on:
        1. the total no. of elements within each bin
        2. the range of each bin

    INPUT:
        :param discreteData: List, the result of the discretizer on the 1D array
        :param originalData: List, the values from the data that were made discrete
    OUPUT:
        returns information 1 and 2 in two dictionaries
    """
    currDataAmounts = {}
    for currIndex in range(len(discreteData)):
        currVal = int(discreteData[currIndex])
        
        if currVal in currDataAmounts:
            currDataAmounts[currVal] += 1
        else:
            currDataAmounts[currVal] = 1
    
    valueRangesDis = {}
    for currIndex in range(len(discreteData)):
        currDisVal = int(discreteData[currIndex])
        currOrigVal = int(originalData[currIndex])
        
        if currDisVal in valueRangesDis:
            currMin, currMax = valueRangesDis.get(currDisVal)
            if currOrigVal < currMin:
                valueRangesDis[currDisVal][0] = currOrigVal
            
            elif currOrigVal > currMax:
                valueRangesDis[currDisVal][1] = currOrigVal
        
        else:
            valueRangesDis[currDisVal] = [currOrigVal, currOrigVal]
    
    return currDataAmounts, valueRangesDis

def prepareExplainerText(amount, ranges):
    """
    Present the information in a nice looking format

    INPUTS:
        :param amount: Dictionary, containing the no. elements for each bin
        :param ranges: Dictioanry, containing the range of elements within the bin
    
    OUTPUT:
        returns a string, formatted so that each iteration is on its own line.
    """
    text = "\n"
    for currKey in amount:
        text += f"{currKey}: {ranges[currKey]} | {amount[currKey]}\n"
    text += "\n\n"
    return text
    


if __name__ == "__main__":
    if len(sys.argv) == 2:
        dataset = sys.argv[1]
    else:
        dataset = input("UK or EU dataset? ").lower()

    if dataset == "uk" or dataset == "eu":
        filePath = f"{FILE_PATH}{dataset}/Rt/{dataset}_Rt.csv"
        
        rtData = readFile(dataset, filePath)
        #Get the data needed to perform K-means discretizing on
        cases = []
        casesCumulative = []
        
        deaths = []
        deathsCumulative = []
        
        tests = []
        testsCumulative = []

        for currRow in rtData:
            cases.append(int(currRow["Cases"]))
            casesCumulative.append(int(currRow["Cumulative Cases"]))

            deaths.append(int(currRow["Deaths"]))
            deathsCumulative.append(int(currRow["Cumulative Deaths"]))

            tests.append(int(currRow["Tests"]))
            testsCumulative.append(int(currRow["Cumulative Tests"]))
        
        #Prepare so that the 1D arrays can be used
        cases = np.array(cases).reshape(-1,1)
        casesCumulative = np.array(casesCumulative).reshape(-1,1)
        
        deaths = np.array(deaths).reshape(-1,1)
        deathsCumulative = np.array(deathsCumulative).reshape(-1,1)
        
        tests = np.array(tests).reshape(-1,1)
        testsCumulative = np.array(testsCumulative).reshape(-1,1)

        discretizer = KBinsDiscretizer(n_bins=BIN_NO, encode='ordinal', strategy='kmeans')

        #perform K-Means
        casesDiscrete = discretizer.fit_transform(cases)
        casesCumulativeDiscrete = discretizer.fit_transform(casesCumulative)

        deathsDiscrete = discretizer.fit_transform(deaths)
        deathsCumulativeDiscrete = discretizer.fit_transform(deathsCumulative)

        testsDiscrete = discretizer.fit_transform(tests)
        testsCumulativeDiscrete = discretizer.fit_transform(testsCumulative)

        #Transfer discrete values into dataset
        for rowIndex in range(len(rtData)):
            rtData[rowIndex]["Cases"] = int(casesDiscrete[rowIndex])
            rtData[rowIndex]["Cumulative Cases"] = int(casesCumulativeDiscrete[rowIndex])

            rtData[rowIndex]["Deaths"] = int(deathsDiscrete[rowIndex])
            rtData[rowIndex]["Cumulative Deaths"] = int(deathsCumulativeDiscrete[rowIndex])

            rtData[rowIndex]["Tests"] = int(testsDiscrete[rowIndex])
            rtData[rowIndex]["Cumulative Tests"] = int(testsCumulativeDiscrete[rowIndex])
        
        fieldNames = {}
        for field in list(rtData[0].keys()):
            fieldNames[field] = field
        
        #Save result
        with open(f"{FILE_PATH}{dataset}/final/{dataset}_final.csv", "w") as optFile:
            myWriter = csv.DictWriter(optFile, fieldNames)
            myWriter.writerow(fieldNames)

            for row in rtData:
                myWriter.writerow(row)
        
        #Provide explanation for result
        outputText = ""
        outputText += f"Bin No: {BIN_NO}\n"

        outputText += "---Cases---"
        currAmount, currRange = getInformationAboutDiscretizer(casesDiscrete, cases)
        outputText += prepareExplainerText(currAmount, currRange)

        outputText += "---Cases Cumulative---"
        currAmount, currRange = getInformationAboutDiscretizer(casesCumulativeDiscrete, casesCumulative)
        outputText += prepareExplainerText(currAmount, currRange)

        outputText += "---Deaths---"
        currAmount, currRange = getInformationAboutDiscretizer(deathsDiscrete, deaths)
        outputText += prepareExplainerText(currAmount, currRange)

        outputText += "---Deaths Cumulative---"
        currAmount, currRange = getInformationAboutDiscretizer(deathsCumulativeDiscrete, deathsCumulative)
        outputText += prepareExplainerText(currAmount, currRange)

        outputText += "---Tests---"
        currAmount, currRange = getInformationAboutDiscretizer(testsDiscrete, tests)
        outputText += prepareExplainerText(currAmount, currRange)

        outputText += "---Tests Cumulative---"
        currAmount, currRange = getInformationAboutDiscretizer(testsCumulativeDiscrete, testsCumulative)
        outputText += prepareExplainerText(currAmount, currRange)

        #Write to a text file
        time = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        
        descriptor = open(f"{FILE_PATH}{dataset}/final/description_{time}.txt", "x")
        descriptor.write(outputText)
    exit()