import sys

from csv import DictReader
from pandas import qcut
from pandas import DataFrame
from statistics import mean
from matplotlib import pyplot as plt
from copy import deepcopy


QUANTILE_NO = 8

REGIONS = {
    1 : "East Midlands",
    2 : "East of England",
    3 : "London",
    4 : "North East",
    5 : "North West",
    6 : "South East",
    7 : "South West",
    8 : "West Midlands",
    9 : "Yorkshire and the Humber",
    10 : "Northern Ireland",
    11 : "Scotland",
    12 : "Blaenau gwent",
    13 : "Caerphilly",
    14 : "Monmouthshire",
    15 : "Newport",
    16 : "Torfaen",
    17 : "Conwy",
    18 : "Denbighshire",
    19 : "Flintshire",
    20 : "Gwynedd",
    21 : "Isle of Anglesey",
    22 : "Wrexham",
    23 : "Cardiff",
    24 : "Vale of Glamorgan",
    25 : "Bridgend",
    26 : "Merthyr Tydfil",
    27 :"Rhondda Cynon Taf",
    28 : "Carmarthenshire",
    29 :"Ceredigion",
    30 : "Pembrokeshire",
    31 : "Powys",
    32 : "Neath Port Talbot",
    33 : "Swansea",
}

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

def getRtValueChange(regionRt):
    """
    From the dataset, go through each regional sublist and obtain the Rt value change from
    one day to the next.

    INPUT:
        :param regionRt: List of Lists, containing only the Rt values for each day in each region
    
    OUTPUT:
        returns a new list of lists, in the same structure and lengths as the input, but with the Rt values
                now being the difference in value between one day to the next.
    """
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
    return rtValueChange

def convertRtValues(rtValueChange):
    """
    Converts all negative values to positive and remove 0 from the dataset.

    INPUT:
        :param rtValueChange: List of lists, the day to day value change of Rt
    
    OUTPUT:
        returns the altered rtValueChange, which may have sublists now shorter.
    """
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
    
    return rtValueChange

def getPotentialActionLists(rtValueChange):
    """
    For each region, it performs Quantile Bucketing as a method,
    of splitting a region's Rt values in an equal manner.
    Then, derives the action list using this split data.

    INPUT:
        :param rtValueChange: List of lists, the day to day value change of Rt
    
    OUTPUT:
        returns a list of lists, where each sublist contains the potential actions 
                that can be taken for each region.
    """
    regionalPotentialActionLists = []

    for currRegionIndex in range(len(rtValueChange)):
        #Convert to a dataframe to easily perform quantile bucketing
        rtValChangeDataFrame = DataFrame({"change" : rtValueChange[currRegionIndex]}, columns=["change"])

        rtValChangeDataFrame["Bucket"] = qcut(rtValChangeDataFrame.change, QUANTILE_NO, labels=list(range(QUANTILE_NO)))
        #The discretized bucket vals for each record
        quantileBuckets = list(rtValChangeDataFrame["Bucket"])

        #Where the Rt val changes will be sorted based on quantileBuckets
        quantileBucketsValues = [[] for _ in range(QUANTILE_NO)]
        
        #Sort the Rt values into their corresponding bucket
        for currIndex in range(len(rtValueChange[currRegionIndex])):
            bucketVal = quantileBuckets[currIndex]
            quantileBucketsValues[bucketVal].append(rtValueChange[currRegionIndex][currIndex])

        actionList = [0,]
        
        #Average each of the buckets to get the action list
        for currBucket in quantileBucketsValues:
            actionList.append(mean(currBucket))


        #For each of the postive potential actions add a negative version
        for currVal in actionList[1:QUANTILE_NO+1]:
            actionList.append(-currVal)
        regionalPotentialActionLists.append(actionList)
    return regionalPotentialActionLists

def greedyAgent(currRegion, actionList, firstValue=True, evaluatePoint=False, evauluateIndex=None, evaluateAction=None, startIndex=None, endIndex=None):
    """
    A greedy implementation for the agent to try and mimic as closely the
    movements of the Rt values.

    INPUTS:
        :param currRegion: List of floats, of the current region's Rt values
        :param actionList: List of floats, the changes in value that the agent will be able to use.
        :param evaluatePoint: Boolean, whether the agent is being used to evaluate the impact of a point
        :param evaluateIndex: Integer, the index of the point that is being evaluated
        :param evaluateAction: Float, the action originally taken that needs to be ignored during the decision making
        :param startIndex: Interger, during evaluation, starting index of the small chunk of the dataset being checked
        :param endIndex: Integer, during evaluation, end index of the small chunk of the dataset being checked

    OUTPUTS:
        returns a list of tuples, containing the Rt value and the action that resulted in said Rt value
    """
    agentRtValues = []

    if evaluatePoint:
        iterations = range(startIndex, endIndex)
    else:
        iterations = range(len(currRegion))

    for currRtIndex in iterations:
        if evaluatePoint != True:
            actionListResults = prepareCurrentActionResults(currRegion, currRtIndex, actionList, firstValue)
        
        elif evaluatePoint and currRtIndex == evauluateIndex:
            evaluateActionList = deepcopy(actionList)

            for currIndex in range(len(evaluateActionList)):
                if evaluateActionList[currIndex] == evaluateAction:
                    evaluateActionList.pop(currIndex)
                    break
            
            actionListResults = prepareCurrentActionResults(currRegion, currRtIndex, evaluateActionList, firstValue)        

        if firstValue:
            firstValue = False
            
        #Find the closest result
        currBestIndex = 0
        for currIndex in range(len(actionListResults)):
            if actionListResults[currIndex] < actionListResults[currBestIndex]:
                currBestIndex = currIndex
        
        agentRtValues.append((currRegion[currRtIndex-1] + actionList[currBestIndex], actionList[currBestIndex]))
    
    return agentRtValues

def evaluatePotentialActions(prevPoint, currPoint, potentialActionList):
    """
    Look through each of the potential actions, and return a list of how close they are
    in achieving the current point's value.

    INPUTS:
        :param prevPoint: Float, the Rt value of the previous point in the data
        :param currPoint: Float, the Rt value of the current point in the data
        :param potentialActionList: List of floats, the various alterations that can be done to prevPoint

    OUTPUT:
        returns a list of floats, where it contains the difference in value between each alteration of prevPoint
                and nextPoint
    """
    actionListResults = []
    for currAction in potentialActionList:
        actionListResults.append(prevPoint + currAction - currPoint)
    
    return actionListResults

def prepareCurrentActionResults(currRegion, currRtIndex, actionList, firstValue=False):
    """
    From the action list, provide how close to the original Rt value the actions are able
    to achieve.

    INPUT:
        :param currRegion: List of floats, a list of the Rt values from the dataset
        :param currRtIndex: Integer, where within currRegion the current point is
        :param actionList: List of floats, a list of the possible adjustments that can be made
        :param firstValue: Boolean, whether the current value is the first within the current region
    
    OUTPUT:
        returns a list of the resultant action of applying the current actions to the current point
    """
    if firstValue != True:
        actionListResults = evaluatePotentialActions(currRegion[currRtIndex-1], currRegion[currRtIndex], actionList)
    else:
        actionListResults = evaluatePotentialActions(currRegion[currRtIndex], currRegion[currRtIndex], actionList)  
    
    #To make comparison simpler, make all positive
    for currIndex in range(len(actionListResults)):    
        if actionListResults[currIndex] < 0:
            actionListResults[currIndex] = -actionListResults[currIndex]
    
    return actionListResults

def createGraph(origRt, agentRt, regionNo):
    """
    Create a line graph that provides a clean comparison between the original Rt,
    and the one produced from the current agent

    Code from Xiyui Fan, adjusted for the dataset and modified further for this application

    INPUT:
        :param origRt: List of floats, the Rt values for the current region from the dataset
        :param agentRt: List of floats, the Rt values produced by the agent when trying to mimic the Rt values.
        :param regionNo: Integer, the current regional number within the dataset
    
    OUTPUT:
        returns nothing, but produces a line graph of the two Rt values plotted together for comparison
    """
    width = .4
    m1_t = DataFrame({'Rt' : origRt, 'Agent Rt' : agentRt})
    ax1 = m1_t[['Rt']].plot(figsize=(12,3))
    ax2 = m1_t['Agent Rt'].plot(secondary_y=False, color='red', label='Agent Rt')

    ax2.legend(loc='upper right')

    plt.xlim([-width, len(m1_t['Agent Rt'])-width])

    xticks = [int(len(origRt)/10)*k for k in range(10)]

    ax1.set_xticks(xticks)
    ax1.set_xticklabels([str(x) for x in xticks])

    ax1.set_xlabel('Days')
    ax1.set_ylabel('Rt')
    
    ax1.set_title(f"{REGIONS[regionNo+1]} - Comparison of Rt vs Agent Rt")

    plt.tight_layout()
    plt.savefig(f"../../images/Rt/machine learning/rt/{regionNo+1}.png")
    plt.close()


def main():
    regionRt = readFile("uk",filePath)

    rtValueChange = getRtValueChange(regionRt)

    rtValueChange = convertRtValues(rtValueChange)

    print(len(rtValueChange))
    print(len(rtValueChange[0]))
    regionalPotentialActionLists = getPotentialActionLists(rtValueChange)
    
    for currIndex in range(len(regionRt)):
        currRegionActionList = regionalPotentialActionLists[currIndex]
        currRegionRt = regionRt[currIndex]

        
        agentResults = greedyAgent(currRegionRt, currRegionActionList)

        agentRt = [currRt for currRt, _ in agentResults]

        createGraph(currRegionRt, agentRt, currIndex)

    #Split the Agent Rt values into k sized groups

    #For each group, add the value to the right of it, to be the goal value to be used for evaluation

    #Then iterate through each group:
        #Remove the action for the current point that was used to get to the next point
        
        #Determine how close the agent was able to get to the goal value, and store this value
        
        #Repeat until all values within the group have been re-run by the agent under these conditions
    
    #Using these values from the groups, determine which of the points had the largest impact on achieving the goal value
    #this will be done by the determining which end value is the largest. After all, if the changing of an action at a certain point
    #can be overcome by later actions, then the effect of that point is rather minimal.
    
if __name__ == "__main__":
    sys.exit(main())
