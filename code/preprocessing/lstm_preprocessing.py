import multiprocessing
import csv

from process_features_uk import processFeaturesUK
from shared.sharedVariables import FILE_PATH_CORE
from shared.sharedFunctions import readFile
from shared.sharedFunctions import removeBrackets
from compute_Rt_uk import computeRtUK

filePath = FILE_PATH_CORE + processFeaturesUK.INPUT_FILE

procUK = processFeaturesUK()
#Read in the file
rawData = readFile("uk", filePath)

fieldNames = procUK.getFieldNames()

currData = []
#Categorise text data
for currRowInd in range(len(rawData)):
    
    newLine = {}

    for currField in fieldNames:
        data = rawData[currRowInd][currField]
        standardFieldName = removeBrackets(currField).rstrip()
        
        if standardFieldName.lower() in procUK.CATEGORISED_FIELDS:
            categoryConversion = procUK.CATEGORISED_FIELDS.get(standardFieldName.lower())
            newLine[currField] = categoryConversion[data.lower()]
        else:
            try:
                newLine[currField] = float(data)
            except:
                if currField == "Tests":
                    #replace with the case number (due to these being obtained through testing)
                    newLine[currField] = int(rawData[currRowInd]["Cases"])
                    rawData[currRowInd]["Cumulative Tests"] = rawData[currRowInd]["Cumulative Cases"]
                else:
                    raise

    currData.append(newLine)

#Convert numerical control measures into days after they were implemented
rtPart = computeRtUK()

currData = rtPart.processControlMeasures(currData)

#Calculate Rt
splitList = []

regionalIndexList = rtPart.getRegionalIndexs(currData)

for currStartIndex, currEndIndex in regionalIndexList:
    splitList.append(currData[currStartIndex:currEndIndex])

numProcesses = 10

with multiprocessing.Pool(numProcesses) as p:
    result = p.map(rtPart.calculateRt, splitList)

optDataList = []

for currRegionalList in result:
    for row in currRegionalList:
        optDataList.insert(0, row)

#Save file
newDataset = []

#Reorder to match original
for row in optDataList:
    newDataset.insert(0, row)

with open(FILE_PATH_CORE + "uk/lstm/lstm_processed.csv", "w") as optFile:
    
    labels = {}
    for currFieldName in newDataset[0].keys():
        labels[currFieldName] = currFieldName
    
    reorderedLabels = rtPart.orderFields(labels, True)
    myWriter = csv.DictWriter(optFile, reorderedLabels)
    
    myWriter.writerow(labels)
    
    for row in newDataset:
        myWriter.writerow(row)