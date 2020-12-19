import sys

from csv import DictReader
from pandas import qcut
from pandas import DataFrame
from statistics import mean
from matplotlib import pyplot as plt
from copy import deepcopy


QUANTILE_NO = 8
GROUP_SIZE = 3

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

def greedyAgent(currRegion, actionList, firstValue=True, evaluatePoint=False, evaluateIndex=None, evaluateAction=None):
    """
    A greedy implementation for the agent to try and mimic as closely the
    movements of the Rt values.

    INPUTS:
        :param currRegion: List of floats, of the current region's Rt values
        :param actionList: List of floats, the changes in value that the agent will be able to use.
        :param evaluatePoint: Boolean, whether the agent is being used to evaluate the impact of a point
        :param evaluateIndex: Integer, the index of the point that is being evaluated
        :param evaluateAction: Integer, the index for the action originally taken that needs to be ignored at the evaluatePoint

    OUTPUTS:
        returns a list of tuples, containing the Rt value and the index for the action that resulted in said Rt value
    """
    agentRtValues = []

    if evaluatePoint:
        iterations = range(1, len(currRegion))
    else:
        iterations = range(len(currRegion))

    for currRtIndex in iterations:
        removedAction = False
        if evaluatePoint != True or currRtIndex-1 != evaluateIndex:
            actionListResults = prepareCurrentActionResults(currRegion, currRtIndex, actionList, firstValue)
        else:
            evalActionList = deepcopy(actionList)
            
            evalActionList.pop(evaluateAction)
            removedAction = True
            
            actionListResults = prepareCurrentActionResults(currRegion, currRtIndex, evalActionList, firstValue)        

        if firstValue:
            firstValue = False
            
        #Find the closest result
        currBestIndex = 0
        for currIndex in range(len(actionListResults)):
            if actionListResults[currIndex] < actionListResults[currBestIndex]:
                currBestIndex = currIndex
        
        #Correct so that compatible with original action list 
        if removedAction and currBestIndex >= evaluateAction:
            currBestIndex += 1

        agentRtValues.append((currRegion[currRtIndex-1] + actionList[currBestIndex], currBestIndex))
        
    
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
        actionListResults = evaluatePotentialActions(currRegion[currRtIndex-1], 
        currRegion[currRtIndex], 
        actionList)
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

    regionalPotentialActionLists = getPotentialActionLists(rtValueChange)
    regionalAgentResults = []
    for currIndex in range(len(regionRt)):
        currRegionActionList = regionalPotentialActionLists[currIndex]
        currRegionRt = regionRt[currIndex]

        
        agentResults = greedyAgent(currRegionRt, currRegionActionList)

        agentRt = [currRt for currRt, _ in agentResults]

        createGraph(currRegionRt, agentRt, currIndex)
        regionalAgentResults.append(agentResults)

    regionEvaluationGroups = [[] for _ in regionRt]

    #For each region, get the start and end index for each group, along with the index of the goal value
    for currRegionIndex in range(len(regionalAgentResults)):
        
        for currGroupIndex in range(int(len(regionalAgentResults[currRegionIndex]) / GROUP_SIZE)):
            currGroup = []
            #Split the Rt values into k sized groups
            startIndex = currGroupIndex * GROUP_SIZE
            goalIndex = currGroupIndex * GROUP_SIZE + GROUP_SIZE
            
            #Gather the current groups data
            currRegionSubset = regionRt[currRegionIndex][startIndex:goalIndex+1]
            
            ignoreCurrentValues = False
            
            for currIndex in range(len(currRegionSubset)-1):
                #Don't run on the last elements of the region if they don't fit neatly
                if goalIndex > len(regionalAgentResults[currRegionIndex])-1:
                    ignoreCurrentValues = True
                    break

                agentRt, agentAction = regionalAgentResults[currRegionIndex][currIndex]
                
                #Due to this being the start val, replace with the agent Rt
                currRegionSubset[0] = agentRt

                agentResults = greedyAgent(currRegionSubset, regionalPotentialActionLists[currRegionIndex], firstValue=False, 
                                            evaluatePoint=True, evaluateIndex=0, evaluateAction=agentAction)
                #Remove the value since it is no longer needed
                currRegionSubset.pop(0)
                
                agentRt, _ = agentResults[-1]

                distanceFromGoal = regionRt[currRegionIndex][goalIndex] - agentRt
                #Add the result to current grouping
                currGroup.append((agentResults, distanceFromGoal))
            
            if ignoreCurrentValues == False:
                regionEvaluationGroups[currRegionIndex].append(((startIndex, goalIndex), currGroup))

    print("----")    

    regionalGroupResults = []
    
    #Compare the original agent action, with the newer attempts
    for currRegionIndex in range(len(regionalAgentResults)):
        currRegionGroupResults = []
        
        for currGroupIndex in range(len(regionEvaluationGroups[currRegionIndex])):
            #Get the information about the current group
            indexs, group = regionEvaluationGroups[currRegionIndex][currGroupIndex]
            startIndex, goalIndex = indexs
            
            #Get the distance of the original attempt
            agentRt, _ = regionalAgentResults[currRegionIndex][goalIndex]
            agentRtDifference = regionRt[currRegionIndex][goalIndex] - agentRt

            if agentRtDifference < 0:
                agentRtDifference = -agentRtDifference
            
            mostImpactfulPoints = None
            prevDifference = 0
            #Go through the eval point results and compare to the original result
            #to find which is the most impactful
            for currIndex in range((goalIndex - 1) - startIndex):
                currGroupDifference = group[currIndex][1]
                if currGroupDifference < 0:
                    currGroupDifference = -currGroupDifference
                
                if prevDifference < currGroupDifference > agentRtDifference:
                    if currGroupDifference == prevDifference:
                        mostImpactfulPoints.append(currIndex)
                    else:
                        prevDifference = currGroupDifference
                        mostImpactfulPoints = [currIndex]

            currRegionGroupResults.append((mostImpactfulPoints, prevDifference))
        
        regionalGroupResults.append(currRegionGroupResults)
                
    for currRegion in regionalGroupResults[32]:
        #for currGroup in currRegion:
        print(currRegion)
    
    #Using these values from the groups, determine which of the points had the largest impact on achieving the goal value
    #this will be done by the determining which end value is the largest. After all, if the changing of an action at a certain point
    #can be overcome by later actions, then the effect of that point is rather minimal.

    """
    When implementing tree agents:

    Have it so that rather than one action is listed for the agentRt value, it is a complete list of the actions it chose to the max depth of the search.
    This way, when it comes to evaluating it, it can occur in the same way as the Greedy Agent. With the twist being that rather than only one action for a given point being prevented,
    it is instead the sequences of actions that lead to it choosing the current action that cannot be done for the current point.
    Meaning that it would be prevented from one action per level it searches ahead.

    """
    
if __name__ == "__main__":
    sys.exit(main())
