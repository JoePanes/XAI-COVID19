from csv import DictReader
from pandas import qcut
from pandas import DataFrame
from statistics import mean

QUANTILE_NO = 8
filePath = "../../data/core/uk/Rt/uk_Rt.csv"

def readFile(dataset, filePath):
    """
    Takes in the contents of the file, and compiles it into
    a usable format

    INPUT:
        :param dataset: String, the name of the dataset being read (determines the order)
        :param filePath: String, the location of the .csv file to be read from
    
    OUTPUT:
        returns a list of lists, where each sublist contains the Rt values for each region 
    """
    compiledData = []

    with open(filePath, "r") as dataFile:
        myReader = DictReader(dataFile)
        currInsertionIndex = 0
        firstIteration = True
        prevRegionNo = None
        
        if dataset == "eu":
            regionalColName = "Country"
        else:
            regionalColName = "Regions"

        for row in myReader:
            #If a new region, start a new sublist
            if prevRegionNo != row[regionalColName]:
                
                if firstIteration:
                    firstIteration = False
                else:
                    currInsertionIndex +=1
                prevRegionNo = row[regionalColName]
                compiledData.append([])

            if dataset == "eu":
                compiledData[currInsertionIndex].append(float(row["Rt"]))
            else:
                compiledData[currInsertionIndex].insert(0, float(row["Rt"]))

    return compiledData

#Read in Rt csv file
regionRt = readFile("uk",filePath)
print(len(regionRt))
print(len(regionRt[0]))
rtValueChange = []
currIndex = 0

#Get the Rt value changes
for currRegion in regionRt:
    rtValueChange.append([])
    prevRt = float(0)
    
    for currRow in currRegion:
        currRt = float(currRow)

        rtValueChange[currIndex].append(float(currRt - prevRt))

        prevRt = currRt
    
    currIndex += 1

print(len(rtValueChange))
print(len(rtValueChange[0]))
"""for currIndex in range(len(rtValueChange[0])):
    print(f"{regionRt[0][currIndex]} : {rtValueChange[0][currIndex]}")"""
        

#Convert the Rt values to positive values and remove 0 values
for currRegionIndex in range(len(rtValueChange)):
    newRegionRtValueChange = []
    
    for currIndex in range(len(rtValueChange[currRegionIndex])):
        currVal = rtValueChange[currRegionIndex][currIndex]
        if currVal < 0:
            newRegionRtValueChange.append(-currVal)
        elif currVal == 0:
            continue
        else:
            newRegionRtValueChange.append(currVal)

    rtValueChange[currRegionIndex] = newRegionRtValueChange

print(len(rtValueChange))
print(len(rtValueChange[0]))
regionalPotentialActionLists = []

for currRegionIndex in range(len(rtValueChange)):
    #Convert to a dataframe to easily perform quantile bucketing
    rtValChangeDataFrame = DataFrame({"change" : rtValueChange[0]}, columns=["change"])

    rtValChangeDataFrame["Bucket"] = qcut(rtValChangeDataFrame.change, QUANTILE_NO, labels=list(range(QUANTILE_NO)))

    #The discretized bucket vals for each record
    quantileBuckets = list(rtValChangeDataFrame["Bucket"])

    #Where the Rt val changes will be sorted based on quantileBuckets
    quantileBucketsValues = [[] for _ in range(QUANTILE_NO)]
    
    #Sort the Rt values into their corresponding bucket
    for currIndex in range(len(rtValueChange[0])):
        bucketVal = quantileBuckets[currIndex]
        quantileBucketsValues[bucketVal].append(rtValueChange[0][currIndex])

    actionList = [0,]
    
    #Average each of the buckets to get the action list
    for currBucket in quantileBucketsValues:
        actionList.append(mean(currBucket))


    #For each of the postive potential actions add a negative version
    for currVal in actionList[1:QUANTILE_NO+1]:
        actionList.append(-currVal)
    regionalPotentialActionLists.append(actionList)
    
print(len(regionalPotentialActionLists[0]))

    
