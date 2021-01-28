import sys
import multiprocessing
import csv
import os

import numpy as np

#Surpress Tensorflow info
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.optimizers import Adam

from pandas import qcut
from pandas import DataFrame
from pandas import read_csv

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler

from random import randint
from random import shuffle

from statistics import mean
from matplotlib import pyplot as plt
from copy import deepcopy
from datetime import datetime

QUANTILE_NO = 5
GROUP_SIZE = 3
MAX_DEPTH = GROUP_SIZE
NUM_PROCESSES = 10

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
    12 : "Wales",
}

class Node:
    def __init__(self, origRt, action, actionIndex, prevRt, parent = None):
        """
        Create a node within the tree
        
        INPUTS:
            :param origRt: Float, the Rt value for the current point within the original dataset
            :param action: Float, the value being used to tweak the prevRtAgent value
            :param actionIndex: Integer, the index for the current position within the action list
            :param prevRt: Float, the value of the previous point
            :param parent: Node, the previous point that this Node is a potential possibility for
        
        OUTPUT:
            returns nothing
        """
        self.action = action
        self.actionIndex = actionIndex
        self.parent = parent

        self.currVal = prevRt + action
        self.cumulativeDifference = None
        #Just something to make returning to the currently expanded node easier to find
        #when backtracking up the tree
        self.isExpandedPoint = False

        currDifference = self.currVal - origRt
        
        if currDifference < 0:
            currDifference = -currDifference

        if parent != None:
            self.cumulativeDifference = parent.cumulativeDifference + currDifference
        else:
            self.cumulativeDifference = currDifference


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
        myReader = csv.DictReader(dataFile)
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

def greedyAgent(currRegion, actionList, firstValue=True, evaluatePoint=False, evaluateAction=None):
    """
    A greedy implementation for the agent to try and mimic as closely the
    movements of the Rt values.

    INPUTS:
        :param currRegion: List of floats, of the current region's Rt values
        :param actionList: List of floats, the changes in value that the agent will be able to use.
        :param evaluatePoint: Boolean, whether the agent is being used to evaluate the impact of a point
        :param evaluateAction: Integer, the index for the action originally taken that needs to be ignored at the evaluatePoint

    OUTPUTS:
        returns a list of tuples, containing the Rt value and the index for the action that resulted in said Rt value
    """
    agentRtValues = []

    currRegion = deepcopy(currRegion)
    
    if evaluatePoint:
        iterations = range(1, len(currRegion))
    else:
        iterations = range(len(currRegion))

    for currRtIndex in iterations:
        removedAction = False
        if evaluatePoint != True or currRtIndex-1 != 0:
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
        
        #Introduce the action taken into the data
        if currRtIndex > 0:
            currRegion[currRtIndex] = currRegion[currRtIndex-1] + actionList[currBestIndex]

        agentRtValues.append((currRegion[currRtIndex], currBestIndex))
        
    return agentRtValues

def mapGreedyAgent(regionalRtAndActionList):
    """
    The method of utililsing the map function, and unpacking the information stored within the singular variable
    so that it can be used by the main greedyAgent() function.

    INPUT:
        :param regionalRtAndActionList: Tuple, containing all of the Rt values for the current region, the action list for the current region and
                                               information for the Greedy Agent function.

    OUTPUT:
        returns the result of the main Greedy agent function.
    """
    currRegion, actionList, firstValue, evaluatePoint, evaluateAction = regionalRtAndActionList

    if evaluatePoint == False:
        return greedyAgent(currRegion, actionList, firstValue, evaluatePoint, evaluateAction)
    else:
        regionEvaluationResults = []
        
        for currGroup in range(len(currRegion)):
            groupResults = []
            for currPoint in range(len(currRegion[currGroup])):
                groupResults.append(greedyAgent(currRegion[currGroup][currPoint], actionList, firstValue, evaluatePoint, evaluateAction[currGroup][currPoint]))

            regionEvaluationResults.append(groupResults)

        return regionEvaluationResults

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
        actionListResults.append((prevPoint + currAction) - currPoint)
    
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

def createGraph(origRt, agentRt, regionNo, agentType):
    """
    Create a line graph that provides a clean comparison between the original Rt,
    and the one produced from the current agent

    Code from Xiyui Fan, adjusted for the dataset and modified further for this application

    INPUT:
        :param origRt: List of floats, the Rt values for the current region from the dataset
        :param agentRt: List of floats, the Rt values produced by the agent when trying to mimic the Rt values.
        :param regionNo: Integer, the current regional number within the dataset
        :param agentType: String, the agent used to obtain these results
    
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
    plt.savefig(f"../../images/Rt/machine learning/rt/{regionNo+1} - {agentType}.png")
    plt.close()

def getCumulativeDifference(element):
    """
    Gets the cumulative difference from the Node
    This was created to be used as the key in sort()

    INPUT:
        :param element: Node, a point within the tree search with a potential action applied
    
    OUTPUT:
        returns the cumulative difference of the Rt for the current Node
    """
    return element.cumulativeDifference

def treeSearch(regionRt, potentialActions, isEvaluationAgent=False, originalAction=None, depthReduction=None):
    """
    Performs a tree search out the MAX_DEPTH to select the best (according to the heuristic) action to take

    INPUTS:
        :param regionRt: List of floats, the regional data that will be explored by the tree search
        :param potentialActions: List of floats, containing floats to be used to alter the Rt value of the previous point
        :param isEvaluationAgent: Boolean, whether the current use of the function is for evaluating a previous run
        :param originalAction: Integer, the index of the original agent action within the potential action list
        :param depthReduction: Integer, the amount to reduce the depth by in order to stop at the goal index during evaluation
    
    OUTPUT:
        returns a list of tuples, containing the Rt value and the index for the action that resulted in said Rt value
    """
    agentRtValues = []

    #Iterate over the current region's data, using the current point as the root for the search
    for currRootIndex in range(len(regionRt)):
        toBeExploredNodes = []
        childNodes = []
        
        #Set up the inital point that we are finding the action for
        if currRootIndex == 0:
            root = Node(regionRt[currRootIndex], 0, 0, regionRt[currRootIndex])
        else:
            root = Node(regionRt[currRootIndex], 0, 0, currentNode.currVal)            

        root.isExpandedPoint = True

        #Get child nodes of the root
        for actionIndex in range(len(potentialActions)):
            if isEvaluationAgent and actionIndex == originalAction:
                continue
            action = potentialActions[actionIndex]

            toBeExploredNodes.append(Node(regionRt[currRootIndex], action, actionIndex, root.currVal, root))
        
        currDepth = 1
        maxDepth = MAX_DEPTH
        if depthReduction != None:
            maxDepth = MAX_DEPTH - depthReduction

        #Explore to max depth or until end of the region
        while maxDepth > currDepth and currRootIndex + currDepth <= len(regionRt) - 1:
            
            while len(toBeExploredNodes) > 0:
                currentNode = toBeExploredNodes.pop(0)
                
                for actionIndex in range(len(potentialActions)):
                    action = potentialActions[actionIndex]
                    childNodes.append(Node(regionRt[currRootIndex + currDepth], action, actionIndex, currentNode.currVal, currentNode))
            
            #Once all explored, prepare childNodes to be explored
            toBeExploredNodes, childNodes = childNodes, toBeExploredNodes
            
            currDepth += 1

        toBeExploredNodes.sort(key=getCumulativeDifference)
        currentNode = toBeExploredNodes.pop(0)

        while currentNode.parent.isExpandedPoint is False:
            currentNode = currentNode.parent
        
        agentRtValues.append((currentNode.currVal, currentNode.actionIndex))
        
        #If evaluating, then only run for the first point, not following points
        if isEvaluationAgent:
            break

    return agentRtValues

def mapTreeSearch(regionalRtAndActionList):
    """
    The method of utililsing the map function, and unpacking the information stored within the singular variable
    so that it can be used by the main treeSearch() function.

    INPUT:
        :param regionalRtAndActionList: Tuple, containing all of the Rt values for the current region, the action list for the current region and
                                               information for the tree function.

    OUTPUT:
        returns the result of the main Greedy agent function.
    """   

    currRegion, potentialActions, isEvaluationAgent, originalAction = regionalRtAndActionList
    if isEvaluationAgent == False:
        return treeSearch(currRegion, potentialActions)
    else:
        regionEvaluationResults = []
        
        for currGroup in range(len(currRegion)):
            groupResults = []

            for currPoint in range(len(currRegion[currGroup])):
                groupResults.append(treeSearch(currRegion[currGroup][currPoint], potentialActions, isEvaluationAgent, originalAction[currGroup][currPoint], currPoint))

            regionEvaluationResults.append(groupResults)

        return regionEvaluationResults
    
def runAgent(potentialActionLists, regionRt, agentType="greedy"):
    """
    Given the actions lists that the agent can take for each region, have it try to match
    as closely as possible to the actual Rt as it can.

    INPUTS:
        :param potentialActionLists: List of lists, containing the potential alterations the agent can make for each region
        :param regionRt: List of lists, containing the Rt values for each day within each region
        :param agentType: String, which type of agent will be use to try and mimic the Rt value
    
    OUTPUT:
        returns the results of running the agent on each region within the dataset
    """
    regionalRtAndActionList = []
    
    #Prepare data for map function
    for currIndex in range(len(regionRt)):
        currActionList = potentialActionLists[currIndex]
        currRegionRt = regionRt[currIndex]
        
        agentInfo = [currRegionRt, currActionList]
        #Based upon what Agent is being used, store the relevant information for its use
        if agentType.lower() == "greedy":
            #firstValue
            agentInfo.append(True)
        
        #isEvaluationAgent
        agentInfo.append(False)
        #Action
        agentInfo.append(None)
        
        regionalRtAndActionList.append(tuple(agentInfo))        
    
    if agentType.lower() == "greedy":
        with multiprocessing.Pool(NUM_PROCESSES) as p:
            agentResults = p.map(mapGreedyAgent, regionalRtAndActionList)
    
    elif agentType.lower() == "tree":
        with multiprocessing.Pool(NUM_PROCESSES) as p:
            agentResults = p.map(mapTreeSearch, regionalRtAndActionList)

    for currIndex in range(len(agentResults)):
        agentRt = [currRt for currRt, _ in agentResults[currIndex]]
        createGraph(regionRt[currIndex], agentRt, currIndex, agentType)
    
    return agentResults

def runEvaluationAgent(potentialActionLists, regionRt, regionalAgentResults, agentType="greedy"):
    """
    Determine how effective the agent's attempt to mimic the Rt is through re-running the agent, but this time,
    removing the actions that it took from its decision making.

    INPUTS:
        :param potentialActionLists: List of lists, containing the potential alterations the agent can make for each region
        :param regionRt: List of lists, containing the Rt values for each day within each region
        :param regionalAgentResults: List of lists, containing all information relating to the actions of the original agent (see runAgent())
        :param agentType: String, which type of agent will be use to try and mimic the Rt value
    
    OUTPUTS:
        returns a list of lists containing tuples within which it contains the range of indexes that the current group covers,
                and the result for each point after denying it the action originally taken.
    """
    regionalRtAndActionList = []
    regionalIndexes = []
    #For each region, get the start and end index for each group, along with the index of the goal value
    for currRegionIndex in range(len(regionalAgentResults)):
        currRegion = []
        currRegionActions = []
        currRegionIndexes = []

        for currGroupIndex in range(int(len(regionalAgentResults[currRegionIndex]) / GROUP_SIZE)):
            currGroup = []
            currGroupActions = []
            #Split the Rt values into k sized groups
            startIndex = currGroupIndex * GROUP_SIZE
            goalIndex = currGroupIndex * GROUP_SIZE + GROUP_SIZE
            
            

            #Gather the current groups data
            currRegionSubset = regionRt[currRegionIndex][startIndex:goalIndex + 1]

            noIterations = range(len(currRegionSubset)-1)
            
            #Don't run on the last elements of the region if they don't fit neatly
            if agentType == "greedy" and goalIndex > len(regionalAgentResults[currRegionIndex])-1:
                break
            elif agentType == "tree" and goalIndex + GROUP_SIZE > len(regionalAgentResults[currRegionIndex]):
                break

            for currIndex in noIterations:
                agentRt, agentAction = regionalAgentResults[currRegionIndex][startIndex + currIndex]
                
                #Due to this being the start val, replace with the agent Rt
                currRegionSubset[0] = agentRt
                currGroup.append(deepcopy(currRegionSubset))
                currGroupActions.append(agentAction)
                
                #Remove the value since it is no longer needed
                currRegionSubset.pop(0)

            currRegion.append(currGroup)
            currRegionActions.append(currGroupActions)
            currRegionIndexes.append((startIndex, goalIndex))
        
        regionalIndexes.append(currRegionIndexes)


        agentInfo = [currRegion, potentialActionLists[currRegionIndex]]
        #Based upon what Agent is being used, store the relevant information for its use
        agentInfo.append(True)

        if agentType.lower() == "greedy":
            agentInfo.append(True)
            
        agentInfo.append(currRegionActions)
        regionalRtAndActionList.append(tuple(agentInfo))

    if agentType.lower() == "greedy":
        with multiprocessing.Pool(NUM_PROCESSES) as p:
            agentResults = p.map(mapGreedyAgent, regionalRtAndActionList)
    
    elif agentType.lower() == "tree":
        with multiprocessing.Pool(NUM_PROCESSES) as p:
            agentResults = p.map(mapTreeSearch, regionalRtAndActionList)

    #After parrallel processing, make some final additions to the results
    for currRegion in range(len(agentResults)):
        for currGroup in range(len(agentResults[currRegion])):
            for currPoint in range(len(agentResults[currRegion][currGroup])):

                goalRt, _ = agentResults[currRegion][currGroup][currPoint][-1]

                _, goalIndex = regionalIndexes[currRegion][currGroup]
                
                distanceFromGoal = regionRt[currRegion][goalIndex] - goalRt
                agentResults[currRegion][currGroup][currPoint] = (agentResults[currRegion][currGroup][currPoint], distanceFromGoal)

            agentResults[currRegion][currGroup] = (regionalIndexes[currRegion][currGroup], agentResults[currRegion][currGroup])

    return agentResults

def evalutateAgentPerformance(regionRt, regionalAgentResults, regionalEvaluationGroups):
    """
    Using the results of the evaluation agent and the original agent, find which point within the group
    provided the most impactful change on the evaluation agents attempt to reach the goal index.

    Where most impactful refers to the idea that if the denial of an action or actions from a specific point
    results in a adversely (larger) effected difference from the goal value (original Rt value).

    INPUTS:
        :param regionRt: List of lists, containing the Rt values for each day within each region
        :param regionalAgentResults: List of lists, containing all information relating to the original agent's actions (see runAgent())
        :param regionalEvaluationGroups: List of lists, containing all the information relating to the evaluation agent's actions (see runEvaluationAgent())

    OUTPUT:
        returns a list of tuples, where each tuple contains (which point(s) in the group had a large impact on Rt, the difference in value between original and evaluation
                agent results, when it reached the goal value)
    """
    regionalGroupResults = []
    
    #Compare the original agent action, with the newer attempts
    for currRegionIndex in range(len(regionalAgentResults)):
        currRegionGroupResults = []
        
        for currGroupIndex in range(len(regionalEvaluationGroups[currRegionIndex])):
            #Get the information about the current group
            indexs, group = regionalEvaluationGroups[currRegionIndex][currGroupIndex]
            startIndex, goalIndex = indexs
            
            #Get the distance of the original attempt
            agentRt, _ = regionalAgentResults[currRegionIndex][goalIndex]

            agentRtDifference = regionRt[currRegionIndex][goalIndex] - agentRt

            pointImpact = []
            differenceTotal = 0
            firstRun = True

            for currIndex in range(len(group)):
                currEvalAgentsDifference = group[currIndex][1]
                
                #Reduce the evaluation agents Rt difference with the original's
                agentDifference = abs(currEvalAgentsDifference - agentRtDifference)
                
                differenceTotal += agentDifference
                
                currPoint = 0
                if firstRun:
                    pointImpact = [[agentDifference, [currIndex]]]
                    firstRun = False
                    continue

                #Rank the impact of the points
                while currPoint < len(pointImpact) and pointImpact[currPoint][0] > agentDifference:
                    currPoint += 1
                
                if currPoint == len(pointImpact):
                    pointImpact.append([agentDifference, [currIndex]])
                
                elif pointImpact[currPoint][0] < agentDifference:
                    pointImpact.insert(currPoint, [agentDifference, [currIndex]])
                
                elif pointImpact[currPoint][0] == agentDifference:
                    pointImpact[currPoint][1].append(currIndex)

            #Enumerate the impact of each point in relation to their difference
            for currPoint in range(len(pointImpact)):
                if pointImpact[currPoint][0] > 0:
                    currPercent = round((pointImpact[currPoint][0] / float(differenceTotal))  * 100)
                    pointImpact[currPoint].append(int(currPercent))
                else:
                    pointImpact[currPoint].append(0)
            currRegionGroupResults.append(pointImpact)
        regionalGroupResults.append(currRegionGroupResults)
    
    return regionalGroupResults

def saveResults(regionRt, regionalAgentResults, regionalEvaluationResults, regionalGroupResults, regionalActionLists, agentType):
    """
    Take in all of the results from the two runs of the Agent and write the desired parts to a 
    .csv file for further use elsewhere.

    INPUTS:
        :param regionRt: List of lists, containing the Rt values for each day for each region
        :param regionalAgentResults: List of lists, containing the results of the agents attempts to mimic regionRt
        :param regionalEvaluationResults: List of lists, groups results of re-running the agent and restricting actions
        :param regionalGroupResults: List of tuples, the result of further comparison of the previous variables in order to determine the most impactful points
        :param regionalActionLists: List of lists, the potential actions that could be taken by the agent for each of the regions
        :param agentType: String, the name of the agent currently being used

    OUTPUT:
        returns the filepath for the newly created .csv containing the formatted data
    """
    filePath = "../../data/core/" + "uk/predictor/" + agentType + ".csv"
    labels = ["Region No", "Group No"]
    """
    What will be needed for the new data set up:

    Region No | Group No | Point 1 Rt Agent | Diff | Point 1 Action to Next | ... | Goal Point Rt Orig | Goal Point Rt Agent Goal Point Diff | Next Point Rt Orig | Next Point Rt Agent | Agent Action to Next Point |

    Where the idea is that the data needs to offer as much information about the actions taken  
    """
    outputList = []

    for currPoint in range(GROUP_SIZE):
        labels.append(f"Point {currPoint +1} Orig Rt")
        labels.append(f"Point {currPoint +1} Agent Difference")
        labels.append(f"Point {currPoint +1} Action to Next Point")

    

    labels += ["Goal Point Orig Rt", "Goal Point Agent Difference", "Next Point Rt Orig", "Next Point Agent Difference", "Agent Action to Next Point"]

    for currRegionIndex in range(len(regionalGroupResults)):
        for currGroupIndex in range(len(regionalGroupResults[currRegionIndex])):
            currRow = {}
            currRow["Region No"] = currRegionIndex + 1
            currRow["Group No"] = currGroupIndex + 1

            goalIndex = 0
            for currIndex in range(GROUP_SIZE):

                indexs, _ = regionalEvaluationResults[currRegionIndex][currGroupIndex]
                startIndex, goalIndex = indexs

                #Calculate the difference in Rt value for the original data, and the agent
                pointOrig = regionRt[currRegionIndex][startIndex + currIndex]
                

                agentRt, action = regionalAgentResults[currRegionIndex][startIndex + currIndex]
                
                currRow[f"Point {currIndex +1} Orig Rt"] = pointOrig
                currRow[f"Point {currIndex +1} Agent Difference"] = pointOrig - agentRt
                
                currRow[f"Point {currIndex +1} Action to Next Point"] = action
            
            goalOrig = regionRt[currRegionIndex][goalIndex]
            goalAgentRt, action = regionalAgentResults[currRegionIndex][goalIndex]
            
            currRow["Goal Point Orig Rt"] = goalOrig
            currRow["Goal Point Agent Difference"] = goalOrig - goalAgentRt
            try:
                nextPoint = regionRt[currRegionIndex][goalIndex+1]
            except:
                #For groups at the end of the regional data, simply ignore them
                #Since for the desired purpose that are not needed
                continue
            currRow["Next Point Rt Orig"] = nextPoint
            currRow["Next Point Agent Difference"] = goalAgentRt + regionalActionLists[currRegionIndex][action]
            currRow["Agent Action to Next Point"] = action

            #print(list(currRow.keys()))
            outputList.append(currRow)

    with open(filePath, "w") as optFile:
            
        outputLabels = {}
        for currFieldName in labels:
            outputLabels[currFieldName] = currFieldName
        
        myWriter = csv.DictWriter(optFile, outputLabels)
        
        myWriter.writerow(outputLabels)
        
        for row in outputList:
            myWriter.writerow(row)
    
    return filePath

def prepareData(filePath, regionNo=None):
    """
    Read in a .csv file, then prepare the data for use with Long-Short-Term Memory

    INPUT:
        :param filePath: String, the location of the .csv file to be used 
    
    OUTPUT:
        returns four separate numpy arrays, these are the training set, test set, and their corresponding most impactful.
    """
    compiledData = read_csv(filePath)

    if regionNo == None:
        maxGroupNo = findGroupNoCutoff(compiledData)

        #Remove Group Nos that don't exist in all regions
        rowsToDrop = []
        for index, row in compiledData.iterrows():
            if regionNo == None and int(row["Group No"]) > maxGroupNo:
                rowsToDrop.append(index)

    elif regionNo != None and int(regionNo) in REGIONS:
        currRegion = compiledData["Region No"] == regionNo

        compiledData = compiledData[currRegion]
    
    coreData = compiledData.drop("Agent Action to Next Point", axis=1)
    goalData = compiledData["Agent Action to Next Point"]
    #Split the grouped data
    trainingCore, testCore, trainingGoal, testGoal = train_test_split(coreData, goalData, test_size=0.3, random_state=randint(1, os.getpid()))

    testCore, validCore, testGoal, validGoal = train_test_split(testCore, testGoal, test_size=0.4, random_state=randint(1, os.getpid()))


    #Get ready to perform feature scaling
    scaler = MinMaxScaler()

    trainingCore = scaler.fit_transform(trainingCore)
    testCore = scaler.transform(testCore)
    validCore = scaler.transform(validCore)

    #Convert back into 3D
    trainingCore = np.reshape(trainingCore, (trainingCore.shape[0], 1, trainingCore.shape[1]))
    
    testCore = np.reshape(testCore, (testCore.shape[0], 1, testCore.shape[1]))

    validCore = np.reshape(validCore, (validCore.shape[0], 1, validCore.shape[1]))

    return trainingCore, trainingGoal.to_numpy(), testCore, testGoal.to_numpy(), validCore, validGoal

def findGroupNoCutoff(dataframe):
    """
    From the dataframe:
        1. Determine where the border points between each region in the data
        2. Using the end points, get the last Group No in each region
        3. Find the largest Group No that exists in all regions
    INPUT:
        :param dataframe: Pandas Dataframe, containing the data after having run the evaluation step on the predictor

    OUTPUT:
        returns an integer of the largest Group No
    """
    #Find where one region ends, and the next begins
    currStartIndexNo = None
    currRegion = None
    regionalRangeList = []
    for index, row in dataframe.iterrows():
        
        if currRegion != row["Region No"]:
            if currRegion != None:
                regionalRangeList.append((currStartIndexNo, index-1))
            
            currRegion = row["Region No"]
            currStartIndexNo = index

    #It misses (due to the if) one region at the end, therefore, add it on
    currIndex = regionalRangeList[-1][1]
    regionalRangeList.append((currIndex + 1, len(dataframe.index)-1))

    #Based upon the regional start and end points, get the last Group No of each region
    groupLengths = []
    dataframeGroupCol = dataframe["Group No"]
    for _, end in regionalRangeList:
        groupLengths.append(int(dataframeGroupCol[end]))

    #Find the smallest-largest point, that exists in all regions
    return min(groupLengths)

def runLSTM(trainingData, trainingAction, validData, validAction):
    """
    Runs the LSTM Recurrent Neural Network on the split dataset

    INPUTS:
        :param trainingData: 3D Numpy Array, contains the feature scaled data
        :param trainingAction: 3D Numpy Array, contains the corresponding actions taken for the training data
        :param validData: 3D Numpy Array, contains the feature scaled data
        :param validAction: 3D Numpy Array, contains the corresponding actions taken for the training data

    OUTPUT:
        returns a trained Sequential model, that includes LSTM layers
    """
    model = Sequential()
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(LSTM(256, activation="swish", return_sequences=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    """
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(LSTM(128, activation="swish", return_sequences=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())"""

    model.add(LSTM(64, activation="swish"))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())


    model.add(Dense(64, activation="swish"))
    model.add(Dropout(0.2))

    model.add(Dense((QUANTILE_NO*2)+1, activation="softmax"))

    opt = Adam(lr=0.001, decay=1e-6)

    model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    history = model.fit(trainingData, trainingAction, validation_data=(validData, validAction), epochs=100, batch_size=12)

    return model

def determineAccuracy(model, testData, testImpact):
    """
    Using a trained LSTM model, determine its accuracy in predicting the goal data, 
    but also get the accuracy for each group of the "Impactful Point?" field

    INPUTS:
        :param model: Sequential trained model, with LSTM layers
        :param testData: Numpy Array, the test data to be used to evalute the model
        :param testImpact: Numpy Array, the corresponding impactful point values for the test data

    OUTPUT:
        returns the percentage for each category and prints the results of running this function to the command line
    """
    #Prepare method of gathering results
    results = {}
    for currResult in range(QUANTILE_NO * 2 + 1):
                             #Correct  #Total
        results[currResult] = [0,       0]
        
    for currIndex in range(len(testData)):
        currentTestData = testData[currIndex]
        currentTestData = np.reshape(currentTestData, (1, currentTestData.shape[0], currentTestData.shape[1]))
        
        prediction = model.predict(currentTestData)[0]
        
        prediction = np.argmax(prediction)
        actualValue = testImpact[currIndex]

        if actualValue == prediction:
            results[prediction][0] += 1
            results[prediction][1] += 1
        else:
            results[actualValue][1] += 1
    
    resultsPercentage = [[] for _ in range(QUANTILE_NO*2+1)]
    overallCorrect = 0
    overallTotal = 0

    for current in range(len(resultsPercentage)):
        currentResults = results[current] 

        percentage = prepareForPrint(str(currentResults[0] / currentResults[1] * 100), 8)

        resultsPercentage[current] = percentage
        
        overallCorrect += currentResults[0]
        overallTotal += currentResults[1]

    overrallPercentage = prepareForPrint(str(round(overallCorrect / overallTotal * 100, 5)), 8)

    print()
    title = "-----------------Result------------------"
    headers = "|Action Index|  "

    #Adjust padding dependending upon size of data

    headers += "Results  "
    
    headers += "|   Percentage   |"
    print(title)
    print(headers)
    for current in range(len(resultsPercentage)):
        print(f"| {current}        |  {results[current]} |   {resultsPercentage[current]}     |")

    print(f"|  Overall |  [{overallCorrect}, {overallTotal}] |   {overrallPercentage}     |")

    return resultsPercentage, overrallPercentage

def prepareForPrint(value, desiredLength):
    """
    Check and adjust the length of a String in preparation for printing

    INPUTS:
        :param value: String, the value to be printed
        :param desiredLength: Integer, the amount of space needing to be occupied by the value

    OUTPUT:
        returns the adjusted String, ready to be printed.
    """

    if len(value) < desiredLength:
        for _ in range(desiredLength - len(value)):
            value += " "
    elif len(value) > desiredLength:
        try:
            value = float(value)

            if value < 10:
                value = round(value, desiredLength - 2)
            else:
                value = round(value, desiredLength - 3)
        except:
            #In this case, the value is not a number, therefore don't alter it
            print("---------------------------ATTENTION-------------------------------")
            print("Can't be shrunk due to String not being a number, therefore, don't")
            print("know how to proceed without causing potential unwanted data loss.")
            print("---------------------------ATTENTION-------------------------------")
    return value

def main():
    """filePath = "../../data/core/uk/2. Rt/uk_Rt.csv"

    regionRt = readFile("uk", filePath)

    rtValueChange = getRtValueChange(regionRt)

    rtValueChange = convertRtValues(rtValueChange)

    regionalPotentialActionLists = getPotentialActionLists(rtValueChange)
    
    regionalAgentResultsTree = runAgent(regionalPotentialActionLists, regionRt, "tree")
    regionalAgentResultsGreed = runAgent(regionalPotentialActionLists, regionRt)

    print("----") 

    regionalEvaluationGroupsGreed = runEvaluationAgent(regionalPotentialActionLists, regionRt, regionalAgentResultsGreed)
    regionalGroupResultsGreed = evalutateAgentPerformance(regionRt, regionalAgentResultsGreed, regionalEvaluationGroupsGreed)
    
    regionalEvaluationGroupsTree = runEvaluationAgent(regionalPotentialActionLists, regionRt, regionalAgentResultsTree, "tree")
    regionalGroupResultsTree = evalutateAgentPerformance(regionRt, regionalAgentResultsTree, regionalEvaluationGroupsTree)

    filePathTree = saveResults(regionRt, regionalAgentResultsTree, regionalEvaluationGroupsTree, regionalGroupResultsTree, regionalPotentialActionLists, "tree")
    filePathGreed = saveResults(regionRt, regionalAgentResultsGreed, regionalEvaluationGroupsGreed, regionalGroupResultsGreed, regionalPotentialActionLists, "greedy")"""

    noIterations = 1
    
    regionStartNo = 1
    regionEndNo = 2

    currRegionResultsOverall = []
    for currRegion in range(regionStartNo, regionEndNo):
        groupResultsAccuracy = [0 for _ in range(QUANTILE_NO*2+1)]
        groupOverallAccuracy = 0

        for _ in range(noIterations):
            trainingData, trainingAction, testData, testAction, validData, validAction = prepareData("../../data/core/" + "uk/predictor/tree.csv")

            model = runLSTM(trainingData, trainingAction, validData, validAction)

            print("Training Data")
            determineAccuracy(model, trainingData, trainingAction)
            
            print("Test Data")
            resultsAccuracy, overallAccuracy = determineAccuracy(model, testData, testAction)


            groupOverallAccuracy += float(overallAccuracy)

            for currIndex in range(len(resultsAccuracy)):
                groupResultsAccuracy[currIndex] += float(resultsAccuracy[currIndex])

        currRegionResultsOverall.append([currRegion]) 
        
        for currIndex in range(len(groupResultsAccuracy)):
            currRegionResultsOverall[currRegion-1].append(round(groupResultsAccuracy[currIndex] / noIterations, 2))
        
        currRegionResultsOverall[currRegion-1].append(round(groupOverallAccuracy / noIterations, 2))
        
    overallSumTest = 0
    for curr in currRegionResultsOverall:
        overallSumTest += curr[-1]
        print(curr)
    print(overallSumTest / (regionEndNo - regionStartNo))

if __name__ == "__main__":
    sys.exit(main())
