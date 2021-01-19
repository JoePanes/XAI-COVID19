"""
Similar use as lstm_preprocessing, but for the Oxford Dataset.
"""

import csv

from copy import deepcopy

from shared.sharedFunctions import readFile
from shared.sharedVariables import FILE_PATH_CORE

CONTROL_MEASURES = {
    'C1_School closing' : ("Quadruple", [0, 1, 2, 3]), 
    'C2_Workplace closing' : ("Quadruple", [0, 1, 2, 3]),
    'C3_Cancel public events' : ("Trinary", [0, 1, 2]), 
    'C4_Restrictions on gatherings' : ("Quintuple", [0, 1, 2, 3, 4]),
    'C5_Close public transport' : ("Trinary", [0, 1, 2]), 
    'C6_Stay at home requirements' : ("Quadruple", [0, 1, 2, 3]),
    'C7_Restrictions on internal movement': ("Trinary", [0, 1, 2]),
    'C8_International travel controls' : ("Quintuple", [0, 1, 2, 3, 4]), 
    'E1_Income support' : ("Trinary", [0, 1, 2]),
    'E2_Debt/contract relief' : ("Trinary", [0, 1, 2]), 
    'H1_Public information campaigns' : ("Trinary", [0, 1, 2]),
    'H2_Testing policy' : ("Quadruple", [0, 1, 2, 3]), 
    'H3_Contact tracing' : ("Trinary", [0, 1, 2]), 
    'H6_Facial Covering_National' : ("Quintuple", [0, 1, 2, 3, 4]),
    'H6_Facial Covering_Regional' : ("Quintuple", [0, 1, 2, 3, 4]), 
    'H7_Vaccination policy' : ("Sextuple", [0, 1, 2, 3, 4, 5]),
}

PAST_DATE_LIST = [0, 6, 12, 18]

def processControlMeasures(processedData):
    """
    Look at the Control measures used, and evalutate the length that have been implemented for, 
    and provide a value.

    INPUT:
        :param processedData: List of dictionaries, where each entry is a row from the datasetc

    OUTPUT:
        returns the new altered version of the dataset
    """
    newData = []
    
    for currRowIndex in range(1, len(processedData)-1):

        #Look at previous dates increasingly further back
        for currDateReduction in PAST_DATE_LIST:
            previousDateIndex = currRowIndex - currDateReduction
            
            if previousDateIndex > 0:

                for currControlMeasure in CONTROL_MEASURES:
                    try:
                        val = int(processedData[previousDateIndex].get(currControlMeasure))
                    except:
                        continue
                    typeOfControlMeasure =CONTROL_MEASURES.get(currControlMeasure)
                    
                    values = typeOfControlMeasure[1]

                    try:
                        currLevel = int(processedData[currRowIndex][currControlMeasure])
                    except:
                        print(currRowIndex, " | ", currControlMeasure)
                        exit()
                        
                    noZeros = 0

                    if typeOfControlMeasure[0].lower() == "trinary":
                        noZeros = 3
                    elif typeOfControlMeasure[0].lower() == "quadruple":
                        noZeros = 4
                    elif typeOfControlMeasure[0].lower() == "quintuple":
                        noZeros = 5
                    elif typeOfControlMeasure[0].lower() == "sextuple":
                        noZeros = 6

                    currDateControlLevel = [1]*(currLevel + 1) + [0]*(noZeros - currLevel - 1)

                    prevDateControlLevel = [1]*(val + 1) + [0]*(noZeros - val - 1)

                    for currIndex in range(len(currDateControlLevel)):

                        currField = f"{currControlMeasure} ({values[currIndex]})"
                        if currDateControlLevel[currIndex] == 1 and currDateControlLevel[currIndex] == prevDateControlLevel[currIndex]:    
                            try:
                                #Due to the breaking up of fields, the val used in binary is not usable for the same purpose here,
                                #since we are not overwriting the same field
                                prevDateVal = int(processedData[previousDateIndex].get(currField))
                            except:
                                #If this is the case, then it is early on in the converting of this region
                                prevDateVal = 0

                            processedData[currRowIndex].update({currField : str(PAST_DATE_LIST.index(currDateReduction) + prevDateVal)})
                        else:

                            processedData[currRowIndex].update({currField : 0})                            

        newRow = deepcopy(processedData[currRowIndex])

        for currControlMeasure in CONTROL_MEASURES:
            try:
                newRow.pop(currControlMeasure)
            except:
                continue
        
        newData.append(newRow)

    return newData

def writeFile(dataset, fileName, containRt = False):
    """
    Writes the current state of the dataset to a .csv file, (along with reorder the dataset
    inorder to match the desired format).

    INPUTS:
        :param dataset: List of dictionaries, where each entry in the list is a row from the dataset
        :param fileName: String, what the name of the outputted file will be
        :param containRt: Boolean, OPTIONAL, whether the current state of the dataset contains an Rt row

    OUTPUT:
        returns nothing, but creates/overwrites a .csv file of the given filename
    """
    with open(FILE_PATH_CORE + "/ox/lstm/" + fileName + ".csv", "w") as optFile:
        
        labels = {}
        for currFieldName in dataset[0].keys():
            labels[currFieldName] = currFieldName
        
        #reorderedLabels = self.orderFields(labels, containRt)
        myWriter = csv.DictWriter(optFile, labels)
        
        myWriter.writerow(labels)
        
        for row in dataset:
            myWriter.writerow(row)


region = "sz"
filepath = FILE_PATH_CORE + f"ox/raw/{region}.csv"

oxDataset =  readFile("eu", filepath)

keys = oxDataset[0].values()

for currKey in keys:
    print(currKey)


controlUkOk = processControlMeasures(oxDataset)

writeFile(controlUkOk,f"processed_{region}")