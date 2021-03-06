"""
A general summary of this file, is as follows:

Step 1: Determine the actions that should be possible for the current Rt value data

Look through the Rt data, sort the changes of data into different bins, where each bin is a range of values (e.g. 0.5).
Once all values have been sorted, the top 8(?) bins (in terms of total values within them) are averaged, and this is added as an action to the potential action list.

Worthwhile note, for the sake of simplification, all negative Rt value changes will be made positive when being sorted into bins.

An example potential action list may be:
[0, 0.5, 0.25, 1, 0.1, 0.75, 1.25, 2, 2.1, -0.5, -0.25, -1, -0.1, -0.75, -1.25, -2, -2.1]
/\ |____________________________________|  |____________________________________________|
|               a)                                              b)
0 will always be in every action list
a) These are the positve bin values
b) Negative verions of the positive bin values are added to provide more options easily

Step 2: Create an agent that using those actions tries to mimic the Rt values as closely as possible

\                      /
 x      x--x    x--x--x
  \    /    \  /
   x--x       x 
   


Step 3: Break up the actions that the agent has chosen into sections, then looking at the point to the right of a section, look
        at each of the points within that section, and evalute which one had the largest impact on its decision to make its currently
        decided action. 

\   |      |      |    |    /
 x  |     x|--x   |x--x|--x
  \ |    / |   \  |/   |
   x|--x   |    x |    |

Idea 1:
        To evalute the previous points, create a list of all potential combinations of actions that could have been performed.
        Then, look through the potential combinations and determine which one had a larger impact upon how the actual value came to be.
                Look at which previous point actions ends up leading to the current point the most?

Step 4: Put all of these window predictions into a table

|Window step|Prediction|
|           |          |
|           |          |

Step 5: Use Machine Learning techniques such as RNN and Regressors to try and predict the following action

Step 6: Use SHAP on the Machine Learning model to see what explanations it can derive
"""
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
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredLogarithmicError
from tensorflow.keras import backend

from pandas import qcut
from pandas import DataFrame
from pandas import read_csv

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.utils import resample

from statistics import mean

from matplotlib import pyplot as plt

from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SMOTEN
from imblearn.over_sampling import KMeansSMOTE
from imblearn.over_sampling import SVMSMOTE

from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import OneSidedSelection
from imblearn.under_sampling import NeighbourhoodCleaningRule
from imblearn.under_sampling import NearMiss

from copy import deepcopy
from copy import copy

from datetime import datetime

from xgboost import XGBRFRegressor

QUANTILE_NO = 3
GROUP_SIZE = 3
MAX_DEPTH = GROUP_SIZE
NUM_PROCESSES = 10
WINDOW_SIZE = 5

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
    13 : "Blaenau Gwent",
    14 : "Caerphilly",
    15 : "Monmouthshire",
    16 : "Newport",
    17 : "Torfaen",
    18 : "Conwy",
    19 : "Denbighshire",
    20 : "Flintshire",
    21 : "Gwynedd",
    22 : "Isle of Anglesey",
    23 : "Wrexham",
    24 : "Cardiff",
    25 : "Vale of Glamorgan",
    26 : "Bridgend",
    27 : "Merthyr Tydfil",
    28 : "Rhondda Cynon Taf",
    29 : "Carmarthenshire",
    30 : "Ceredigion",
    31 : "Pembrokeshire",
    32 : "Powys",
    33 : "Neath Port Talbot",
    34 : "Swansea",
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


def readFile(dataset, filePath, startYear=None, endYear=None):
    """
    Takes in the contents of the file, and compiles it into
    a usable format

    INPUT:
        :param dataset: String, the name of the dataset being read (determines the order)
        :param filePath: String, the location of the .csv file to be read from
        :param startYear: Integer, for the financial data, which year to start the data range from
        :param endYear: Integer, for the financial data, which year to end the data range at
    
    OUTPUT:
        returns a list of lists, where each sublist contains the values for each region or year
    """
    compiledData = []

    with open(filePath, "r") as dataFile:
        myReader = csv.DictReader(dataFile)
        currInsertionIndex = 0
        firstIteration = True
        prevRegionNo = None
        dateColName = ""
        prevYear = None
        
        if dataset == "eu":
            regionalColName = "Country"
        elif dataset == "fn":
            regionalColName = None
            compiledData.append([])
        else:
            regionalColName = "Regions"

        for row in myReader:
            #If a new region, start a new sublist
            if regionalColName != None and prevRegionNo != row[regionalColName]:
                
                if firstIteration:
                    firstIteration = False
                else:
                    currInsertionIndex +=1
                prevRegionNo = row[regionalColName]
                compiledData.append([])

            if dataset == "eu":
                compiledData[currInsertionIndex].append(float(row["Rt"]))
            
            elif dataset == "uk":
                compiledData[currInsertionIndex].insert(0, float(row["Rt"]))

            elif dataset == "fn":
                if firstIteration:
                    dateColName = list(row.keys())[0]
                    firstIteration == False
                
                currYear = int(row[dateColName][-4:])
                
                #Only add dates within the desired years
                if currYear >= startYear and currYear <= endYear:
                    if currYear != prevYear:
                        if type(prevYear) == int:
                            compiledData.append([])
                            currInsertionIndex += 1
                        
                        prevYear = currYear
                    
                    compiledData[currInsertionIndex].append(float(row["US $ TO UK £ (WMR) - EXCHANGE RATE"]))
                
                elif currYear > endYear:
                    #Rather pointless to continue looking through the data if it is exceeding
                    #what is desired
                    break


            else:
                print("The dataset that you are trying to use has not been implemented, or, there is an error in how it has been typed.")
                print("Please check the code to determine the course of correction to be taken, thank you =)")
                sys.exit()

    return compiledData

def getValueChange(regionValues):
    """
    From the dataset, go through each regional sublist and obtain the value change from
    one day to the next.

    INPUT:
        :param regionValues: List of Lists, containing only the Rt values for each day in each region
    
    OUTPUT:
        returns a new list of lists, in the same structure and lengths as the input, but with the Rt values
                now being the difference in value between one day to the next.
    """
    valueChange = []
    currIndex = 0

    #Get the Rt value changes
    for currRegion in regionValues:
        valueChange.append([])
        prevVal = float(0)
        
        for currVal in currRegion:
            valueChange[currIndex].append(float(float(currVal) - prevVal))

            prevVal = float(currVal)
        
        currIndex += 1
    return valueChange

def convertValues(valueChange):
    """
    Converts all negative values to positive and remove 0 from the dataset.

    INPUT:
        :param valueChange: List of lists, the day to day value change of value
    
    OUTPUT:
        returns the altered valueChange, which may have sublists now shorter.
    """
    for currRegionIndex in range(len(valueChange)):
        newRegionValueChange = []
    
        for currIndex in range(len(valueChange[currRegionIndex])):
            currVal = valueChange[currRegionIndex][currIndex]
            if currVal < 0:
                newRegionValueChange.append(-currVal)
            elif currVal == 0:
                continue
            else:
                newRegionValueChange.append(currVal)

        valueChange[currRegionIndex] = newRegionValueChange
    
    return valueChange

def getPotentialActionLists(rtValueChange):
    """
    For each region, it performs Quantile Bucketing as a method,
    of splitting a region's values in an equal manner.
    Then, derives the action list using this split data.

    INPUT:
        :param rtValueChange: List of lists, the day to day value change
    
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

        #Where the val changes will be sorted based on quantileBuckets
        quantileBucketsValues = [[] for _ in range(QUANTILE_NO)]
        
        #Sort the values into their corresponding bucket
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
    movements of the values.

    INPUTS:
        :param currRegion: List of floats, of the current region's values
        :param actionList: List of floats, the changes in value that the agent will be able to use.
        :param evaluatePoint: Boolean, whether the agent is being used to evaluate the impact of a point
        :param evaluateAction: Integer, the index for the action originally taken that needs to be ignored at the evaluatePoint

    OUTPUTS:
        returns a list of tuples, containing the Rt value and the index for the action that resulted in said value
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
        :param regionalRtAndActionList: Tuple, containing all of the values for the current region, the action list for the current region and
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
        :param prevPoint: Float, the value of the previous point in the data
        :param currPoint: Float, the value of the current point in the data
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
    From the action list, provide how close to the original value the actions are able
    to achieve.

    INPUT:
        :param currRegion: List of floats, a list of the values from the dataset
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

def createGraph(origVal, agentVal, regionNo, agentType, dataset, startYear=None, currRegionNo=None):
    """
    Create a line graph that provides a clean comparison between the original,
    and the one produced from the current agent

    Code from Xiyui Fan, adjusted for the dataset and modified further for this application

    INPUT:
        :param origVal: List of floats, the values for the current region from the dataset
        :param agentVal: List of floats, the values produced by the agent when trying to mimic the dataset values.
        :param regionNo: Integer, the current regional number within the dataset
        :param agentType: String, the agent used to obtain these results
        :param dataset: String, which dataset is currently being used
        :param startYear: Integer, for financial dataset, what is the starting year for the range of data
    
    OUTPUT:
        returns nothing, but produces a line graph of the two Rt values plotted together for comparison
    """
    width = .4

    m1_t = DataFrame({'Original' : origVal, 'Agent' : agentVal})
    ax1 = m1_t[['Original']].plot(figsize=(12,3))
    ax2 = m1_t['Agent'].plot(secondary_y=False, color='red', label='Agent Rt')

    ax2.legend(loc='upper right')

    plt.xlim([-width, len(m1_t['Agent'])-width])
    xticks = [int(len(origVal)/10)*k for k in range(10)]

    ax1.set_xticks(xticks)
    ax1.set_xticklabels([str(x) for x in xticks])
    ax1.set_xlabel('Days')

    if dataset == "uk":
        ax1.set_ylabel('Rt')
        ax1.set_title(f"{REGIONS[regionNo + 1]} - Comparison of Original vs Agent Rt")
    else:
        ax1.set_ylabel('$ to £ Exchange rate')
        ax1.set_title(f"{startYear + regionNo} - Comparison of Original vs Agent")

    plt.tight_layout()

    filename = f"../../images/machine learning/predictor/{dataset}/"

    if dataset == "uk":
        filename += f"{regionNo + 1}"
    elif dataset == "fn":
        filename += f"{startYear + regionNo}"
    else:
        filename += f"{regionNo + 1}"
    
    filename +=f"- {agentType}.png"

    plt.savefig(filename, dpi =600)
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

def treeSearch(regionVal, potentialActions, isEvaluationAgent=False, originalAction=None, depthReduction=None):
    """
    Performs a tree search out the MAX_DEPTH to select the best (according to the heuristic) action to take

    INPUTS:
        :param regionVal: List of floats, the regional data that will be explored by the tree search
        :param potentialActions: List of floats, containing floats to be used to alter the value of the previous point
        :param isEvaluationAgent: Boolean, whether the current use of the function is for evaluating a previous run
        :param originalAction: Integer, the index of the original agent action within the potential action list
        :param depthReduction: Integer, the amount to reduce the depth by in order to stop at the goal index during evaluation
    
    OUTPUT:
        returns a list of tuples, containing the value and the index for the action that resulted in said value
    """
    agentRtValues = []

    #Iterate over the current region's data, using the current point as the root for the search
    for currRootIndex in range(len(regionVal)):
        toBeExploredNodes = []
        childNodes = []
        
        #Set up the inital point that we are finding the action for
        if currRootIndex == 0:
            root = Node(regionVal[currRootIndex], 0, 0, regionVal[currRootIndex])
        else:
            root = Node(regionVal[currRootIndex], 0, 0, currentNode.currVal)            

        root.isExpandedPoint = True

        #Get child nodes of the root
        for actionIndex in range(len(potentialActions)):
            if isEvaluationAgent and actionIndex == originalAction:
                continue
            action = potentialActions[actionIndex]

            toBeExploredNodes.append(Node(regionVal[currRootIndex], action, actionIndex, root.currVal, root))
        
        currDepth = 1
        maxDepth = MAX_DEPTH
        if depthReduction != None:
            maxDepth = MAX_DEPTH - depthReduction

        #Explore to max depth or until end of the region
        while maxDepth > currDepth and currRootIndex + currDepth <= len(regionVal) - 1:
            
            while len(toBeExploredNodes) > 0:
                currentNode = toBeExploredNodes.pop(0)
                
                for actionIndex in range(len(potentialActions)):
                    action = potentialActions[actionIndex]
                    childNodes.append(Node(regionVal[currRootIndex + currDepth], action, actionIndex, currentNode.currVal, currentNode))
            
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
        :param regionalRtAndActionList: Tuple, containing all of the values for the current region, the action list for the current region and
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
    
def runAgent(potentialActionLists, regionVal, dataset, agentType="greedy", startYear=None):
    """
    Given the actions lists that the agent can take for each region, have it try to match
    as closely as possible to the actual Rt as it can.

    INPUTS:
        :param potentialActionLists: List of lists, containing the potential alterations the agent can make for each region
        :param regionVal: List of lists, containing the values for each day within each region
        :param dataset: String, which dataset is currently being used
        :param agentType: String, which type of agent will be use to try and mimic the Rt value
        
    OUTPUT:
        returns the results of running the agent on each region within the dataset
    """
    regionalRtAndActionList = []
    
    #Prepare data for map function
    for currIndex in range(len(regionVal)):
        currActionList = potentialActionLists[currIndex]
        currRegionRt = regionVal[currIndex]
        
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
        createGraph(regionVal[currIndex], agentRt, currIndex, agentType, dataset, startYear)
    
    return agentResults

def runEvaluationAgent(potentialActionLists, regionVal, regionalAgentResults, agentType="greedy"):
    """
    Determine how effective the agent's attempt to mimic the Rt is through re-running the agent, but this time,
    removing the actions that it took from its decision making.

    INPUTS:
        :param potentialActionLists: List of lists, containing the potential alterations the agent can make for each region
        :param regionVal: List of lists, containing the values for each day within each region
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
            currRegionSubset = regionVal[currRegionIndex][startIndex:goalIndex + 1]

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
                
                distanceFromGoal = regionVal[currRegion][goalIndex] - goalRt
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

def saveResults(regionRt, regionalAgentResults, regionalEvaluationResults, regionalGroupResults, regionalActionLists, agentType, dataset, useDifferentWindowSize=None):
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
        :param dataset: String, which dataset is currently being used 
        :param useDifferentWindowSize: Integer, if given to the function, it will use that value over the value of WINDOW_SIZE

    OUTPUT:
        returns the filepath for the newly created .csv containing the formatted data
    """
    windowSize = WINDOW_SIZE
    filePath = f"../../data/core/{dataset}/predictor/{agentType}.csv"

    if type(useDifferentWindowSize) == int:
        windowSize = useDifferentWindowSize
        filePath = f"../../data/core/{dataset}/predictor/{agentType}_{useDifferentWindowSize}.csv"

    labels = []

    labels.append("Region No")    

    labels.append("Group No")

    outputList = []

    for currPoint in range(1, windowSize+1):
        labels.append(f"Action {currPoint}")

    labels.append(f"Action to Next Point")

    for currRegionIndex in range(len(regionalAgentResults)):
        for currPoint in range(len(regionalAgentResults[currRegionIndex])):
            currRow = {}
            currRow["Region No"] = currRegionIndex + 1
            currRow["Group No"] = currPoint +1

            try:
                currPointNo = 1
                for currIndex in range(windowSize):
                    _, action = regionalAgentResults[currRegionIndex][currPoint + currIndex]
                    if action > QUANTILE_NO:
                        action = QUANTILE_NO - action
                    currRow[f"Action {currPointNo}"] = action
                    currPointNo+= 1

                _, action = regionalAgentResults[currRegionIndex][currPoint+currIndex+1]

                
                if action > QUANTILE_NO:
                    action = QUANTILE_NO - action
                
                currRow[f"Action to Next Point"] = action
            except:
                #For groups at the end of the regional data, simply ignore them
                #Since for the desired purpose that are not needed
                continue
        
            #print(list(currRow.keys()))
            outputList.append(currRow)

    writeFile(filePath, outputList, labels)
    
    return filePath

def writeFile(filePath, outputList, labels=None):
    """
    Write the current Dataset to .csv File

    INPUTS:
        :param filePath: String, the directory path to where the new .csv will be saved
        :param outputList: List of Dictionaries, where each dictionary is a row for the .csv
        :param labels: List of Strings, the desired order for the .csv to be saved in

    OUTPUT:
        returns nothing, but creates a new .csv at the specified location
    """
    with open(filePath, "w") as optFile:
            
        outputLabels = {}
        if labels != None and type(labels[0]) is str:
            for currFieldName in labels:
                outputLabels[currFieldName] = currFieldName
        else:
            for currFieldName in list(outputList[0].keys()):
                outputLabels[currFieldName] = currFieldName

        myWriter = csv.DictWriter(optFile, outputLabels)
        
        myWriter.writerow(outputLabels)
        
        for row in outputList:
            myWriter.writerow(row)

def prepareDataLSTM(filePath, dataset, regionNo=None):
    """
    Read in a .csv file, then prepare the data for use with Long-Short-Term Memory

    INPUT:
        :param filePath: String, the location of the .csv file to be used 
        :param dataset: String, what dataset is currently being used
        :param regionNo: Integer, the specific regional data to be used

    OUTPUT:
        returns six separate numpy arrays, these are the training set, test set, and the validation set.
    """
    compiledData = read_csv(filePath)

    if regionNo != None and int(regionNo) in REGIONS:
        currRegion = compiledData["Region No"] == regionNo

        compiledData = compiledData[currRegion]

    columnsToDrop = ["Action to Next Point"]

    if dataset == "uk":
        columnsToDrop += ["Region No", "Group No"]
    coreData = compiledData.drop(columnsToDrop, axis=1)
    goalData = compiledData["Action to Next Point"]

    smote = SMOTE(k_neighbors=7)
    rus = RandomUnderSampler()
    #oss = OneSidedSelection()

    #Split the grouped data
    trainingCore, testCore, trainingGoal, testGoal = train_test_split(coreData, goalData, test_size=0.2, random_state=10, stratify=goalData)
    
    #trainingCore, trainingGoal = smote.fit_resample(trainingCore, trainingGoal)
    #trainingCore, trainingGoal = rus.fit_resample(trainingCore, trainingGoal)
    testCore, validCore, testGoal, validGoal = train_test_split(testCore, testGoal, test_size=0.5, random_state=10, stratify=testGoal)


    #Get ready to perform feature scaling
    scaler = MinMaxScaler(feature_range=(-QUANTILE_NO, QUANTILE_NO))

    trainingCore = scaler.fit_transform(trainingCore)
    testCore = scaler.transform(testCore)
    validCore = scaler.transform(validCore)

    #Convert back into 3D
    trainingCore = np.reshape(trainingCore, (trainingCore.shape[0], 1, trainingCore.shape[1]))
    
    testCore = np.reshape(testCore, (testCore.shape[0], 1, testCore.shape[1]))

    validCore = np.reshape(validCore, (validCore.shape[0], 1, validCore.shape[1]))

    return trainingCore, trainingGoal.to_numpy(), testCore, testGoal.to_numpy(), validCore, validGoal

def prepareDataUnalteredLSTM(regionValues, windowSize):
    """
    Read in a .csv file of the original values, then prepare the data for use with Long-Short-Term Memory

    Not currently really setup to work yet, my plan was:

    Round the data to 1 decimal place
    Format the goal data so that there are no negatives, or see if it will work with RMSE/MSE
    See what the results are and tinker from there

    INPUT:
        :param regionValues: List of lists, containing numerical values for each day 
        :param windowSize: Integer, how many days to put together 

    OUTPUT:
        returns six separate numpy arrays, these are the training set, test set, and the validation set.
    """
    windowedData = []

    for currGroup in regionValues:
        for currStart in range(len(currGroup)-(windowSize)):
            
            currRow = {}
            for currIndex in range(windowSize):
                currRow[f"Action {currIndex +1}"] = currGroup[currStart + currIndex]

            currRow["Action to Next Point"] = currGroup[currStart + currIndex +1]
            windowedData.append(currRow)

    windowedDataframe = DataFrame(windowedData)

    #windowedDataframe.round(1)
    
    coreData = windowedDataframe.drop(["Action to Next Point"], axis=1)
    goalData = windowedDataframe["Action to Next Point"]

    print(goalData)
    #Split the grouped data
    trainingCore, testCore, trainingGoal, testGoal = train_test_split(coreData, goalData, test_size=0.2, random_state=10)

    testCore, validCore, testGoal, validGoal = train_test_split(testCore, testGoal, test_size=0.5, random_state=10)


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

def createNewGroupRow(currGroup, nextGroup, pointsToGather, endPoint):
    """
    Using the points present within two groups, create a third new group using the points specified
    by pointsToGather and work out what else needs to be taken through endPoint.

    INPUTS:
        :param currGroup: Dictionary, containing a row of the dataset
        :param nextGroup: Dictionary, containing the next row of the dataset
        :param pointsToGather: 2D list of integers, containing what points need to be taken from either dictionary
        :param endPoint: Boolean, whether it is the last point within the group that is being taken from currGroup

    OUTPUT:
        returns a Dictionary, containing a mixture of contents from currGroup and nextGroup
    """
    newRow = {}
    newRow["Region No"] = currGroup["Region No"]
    currPoint = 1
    #Sort out the Group Points
    for currPointIndex in range(len(pointsToGather[0])):
        currGroupPoint = pointsToGather[0][currPointIndex]
        newRow[f"Point {currPoint + currPointIndex} Orig Rt"] = currGroup[f"Point {currGroupPoint} Orig Rt"]
        newRow[f"Point {currPoint + currPointIndex} Agent Difference"] = currGroup[f"Point {currGroupPoint} Agent Difference"]
        newRow[f"Point {currPoint + currPointIndex} Action to Next Point"] = currGroup[f"Point {currGroupPoint} Action to Next Point"]
    
    currPoint += len(pointsToGather[0])

    for currPointIndex in range(len(pointsToGather[1])):
        currGroupPoint = pointsToGather[1][currPointIndex]
        newRow[f"Point {currPoint + currPointIndex} Orig Rt"] = nextGroup[f"Point {currGroupPoint} Orig Rt"]
        newRow[f"Point {currPoint + currPointIndex} Agent Difference"] = nextGroup[f"Point {currGroupPoint} Agent Difference"]
        newRow[f"Point {currPoint + currPointIndex} Action to Next Point"] = nextGroup[f"Point {currGroupPoint} Action to Next Point"]
    
    #Sort out the Goal Point and Next Point
    goalPoint = pointsToGather[1][currPointIndex]+1

    newRow[f"Goal Point Orig Rt"] = nextGroup[f"Point {goalPoint} Orig Rt"]
    newRow[f"Goal Point Agent Difference"] = nextGroup[f"Point {goalPoint} Agent Difference"]

    if endPoint:
        newRow[f"Next Point Rt Orig"] = nextGroup["Goal Point Orig Rt"]
        newRow[f"Next Point Agent Difference"] = nextGroup["Goal Point Agent Difference"]
        newRow[f"Agent Action to Next Point"] = nextGroup[f"Point {goalPoint} Action to Next Point"]
    else:
        goalPoint += 1
        newRow[f"Next Point Rt Orig"] = nextGroup[f"Point {goalPoint} Orig Rt"]
        newRow[f"Next Point Agent Difference"] = nextGroup[f"Point {goalPoint} Agent Difference"]
        newRow[f"Agent Action to Next Point"] = nextGroup[f"Point {goalPoint} Action to Next Point"]

    return newRow

def runLSTM(trainingData, trainingAction, validData, validAction):
    """
    Runs the LSTM Recurrent Neural Network on the dataset

    INPUTS:
        :param trainingData: 3D Numpy Array, contains the feature scaled data
        :param trainingAction: 3D Numpy Array, contains the corresponding actions taken for the training data
        :param validData: 3D Numpy Array, contains the feature scaled data
        :param validAction: 3D Numpy Array, contains the corresponding actions taken for the training data

    OUTPUT:
        returns a trained Sequential model, that includes LSTM layers
    """

    model = Sequential()
    model.add(LSTM(64, activation="swish", return_sequences=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    
    model.add(LSTM(128, activation="swish", return_sequences=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(LSTM(256, activation="swish"))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Dense(64, activation="swish"))

    model.add(Dense(128, activation="softmax"))
     
    opt = Adam(lr=0.0005, decay=0.001)

    model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    history = model.fit(trainingData, trainingAction, validation_data=(validData, validAction), epochs=256, batch_size=300)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model train vs validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig(f"../../images/machine learning/predictor/{datetime.now().strftime('%d_%m_%Y_%H_%M_%S_%f')}.png")
    plt.close()

    return model

def testLSTM(trainingData, trainingAction, validData, validAction, dataset, lstmConfiguration):
    """
    Runs a LSTM of the specified configuration. Requires altering of the code in the desired parts for
    effects to take place.

    INPUTS:
        :param trainingData: 3D Numpy Array, contains the feature scaled data
        :param trainingAction: 3D Numpy Array, contains the corresponding actions taken for the training data
        :param validData: 3D Numpy Array, contains the feature scaled data
        :param validAction: 3D Numpy Array, contains the corresponding actions taken for the training data
        :param dataset: String, which dataset is currently being used
        :param lstmConfiguration: Iterable (list or tuple) of numerical values, where each position contains the desired value for a component within the LSTM.

    OUTPUT:
        returns a trained Sequential model, that includes LSTM layers
    """
    learningRate = 0.0005
    epochs = 40
    decay = 1e-6
    model = Sequential()

    model.add(LSTM(32, activation="swish", return_sequences=True, dropout=0.2))

    model.add(LSTM(64, activation="swish", return_sequences=True, dropout=0.2))

    model.add(LSTM(128, activation="swish", return_sequences=True, dropout=0.2))

    model.add(LSTM(256, activation="swish", return_sequences=True, dropout=0.2))

    model.add(LSTM(128, activation="swish", return_sequences=True, dropout=0.2))

    model.add(LSTM(64, activation="swish", dropout=0.4))    

    model.add(Dense(32, activation="swish"))

    #output layer
    model.add(Dense((QUANTILE_NO*2)+1))

    opt = Adam(lr=learningRate, decay=decay, beta_1=0.5)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    history = model.fit(trainingData, trainingAction, epochs=epochs, validation_data=(validData, validAction), batch_size=128)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model train vs validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig(f"../../images/machine learning/predictor/{dataset}/{lstmConfiguration}---{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.png")
    plt.close()

    return model, history

def determineAccuracy(model, testData, testAction, printTable=False):
    """
    Using a trained LSTM model, derive some statistics on how well it has performed.

    INPUTS:
        :param model: Sequential trained model, with LSTM layers
        :param testData: Numpy Array, the test data to be used to evalute the model
        :param testAction: Numpy Array, the resultant action after the previous actions
        :param printTable: Boolean, whether or not to print the results for the test data

    OUTPUT:
        returns the percentage for each category and prints the results of running this function to the command line
    """
    #Prepare method of gathering results
    results = {}
    for currResult in range(QUANTILE_NO * 2 + 1):
                             #Correct  #Total #FP #FN
        results[currResult] = [0,       0,     0,  0]
    predictionResults = []
    for currIndex in range(len(testData)):
        currentTestData = testData[currIndex]
        currentTestData = np.reshape(currentTestData, (1, currentTestData.shape[0], currentTestData.shape[1]))
        
        prediction = model.predict(currentTestData)
        
        prediction = int(np.argmax(prediction))
        predictionResults.append(prediction)
        actualValue = testAction[currIndex]

        if prediction < 0:
            prediction = QUANTILE_NO - prediction
        if actualValue < 0:
            actualValue = QUANTILE_NO - actualValue

        if actualValue == prediction:
            results[prediction][0] += 1
            results[prediction][1] += 1
        else:
            results[actualValue][1] += 1
            results[prediction][2] += 1
            results[actualValue][3] += 1
    
    resultsPercentage = [[] for _ in range(QUANTILE_NO*2+1)]
    overallCorrect = 0
    overallTotal = 0

    for current in range(len(resultsPercentage)):
        currentResults = results[current] 
        try:
            percentage = prepareForPrint(str(currentResults[0] / currentResults[1] * 100), 8)
        except:
            #In the event that the current action doesn't exist within the test set, then return 0 
            percentage = 100.0
        resultsPercentage[current] = percentage
        
        overallCorrect += currentResults[0]
        overallTotal += currentResults[1]

    overrallPercentage = prepareForPrint(str(round(overallCorrect / overallTotal * 100, 5)), 8)
    if printTable:
        print()
        title = "-----------------Result------------------"
        headers = "|Action Index|  "
        #Adjust padding dependending upon size of data
        headers += "Results  "
        headers += "|   Percentage   | FP | FN |"
        print(title)
        print(headers)

        for current in range(len(resultsPercentage)):
            print(f"| {current}        |  {results[current][0:2]} |   {resultsPercentage[current]}     | {results[current][2]}| {results[current][3]}")

        print(f"|  Overall |  [{overallCorrect}, {overallTotal}] |   {overrallPercentage}     |")

        print("Confusion Matrix")
        confuse = confusion_matrix(testAction, np.array(predictionResults))

        print(confuse)

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

def generateSineData(noRegions, length):
    """
    Generate data that produces a Sine wave for the sake of testing

    INPUTS:
        :param noRegions: Integer, used to determine how to split the data into equal portions
        :param length: Integer, the amount of data to be produced for each region

    OUTPUT:
        returns a 2D list, containing the split Sine data.
    """

    totalData = noRegions * length

    #Based upon the example from https://pythontic.com/visualization/charts/sinewave

    time = np.arange(0, totalData, 1)

    amplitude = np.sin(time * np.pi / 10) #this alteration is based upon https://numpy.org/doc/stable/reference/generated/numpy.sin.html

    plt.plot(amplitude)
    plt.title('Sine Wave')
    plt.savefig(f"../../images/machine learning/predictor/sn/{datetime.now().strftime('%d_%m_%Y_%H_%M_%S_%f')}.png")
    plt.close()
    
    #Break up the Sine data into equal chunks, to mimic the regional nature of the original data
    currRegionLength = 0
    regionalData = []
    currRegionData = []
    for currIndex in range(len(amplitude)):
        if currRegionLength == length:
            currRegionLength = 0
            regionalData.append(currRegionData)
            currRegionData = []
        
        currRegionData.append(amplitude[currIndex])
        currRegionLength += 1
    
    #Add on the final region
    regionalData.append(currRegionData)

    return regionalData

def rmse(actual, predicted):
    """
    Returns a percentage of how similar the numbers in actual are to those in prediction.

    INPUT:
        :param actual: List of what is the real values for the data
        :param predicted: List of what the current model guesses the values are

    OUTPUT:
        returns a float to give an indication of, on average, how similar the  
    """
    msle = MeanSquaredLogarithmicError()
    return backend.sqrt(msle(actual, predicted)) 

def normalise(regionalVal, amount):
    """
    Iterate over the regional values and normalise their values to reduce potentially extreme trends

    INPUTS:
        :param regionalVal: 3D list of floats, contains each regions Rt values for each day
        :param amount: Integer, the number of points to gather when averaging

    OUTPUT:
        returns a normalised version of the regional Rt data
    """

    halfOfAmountFloored = int((amount -1) / 2)
    normalisedRt = []
    for currRegion in regionalVal:
        currRegionNormalisedRt = []

        for currIndex in range(halfOfAmountFloored, len(currRegion) - halfOfAmountFloored):
            currPointRtValues = []
            #Give more weight to the surrounding values
            currPointRtValues.append(currRegion[currIndex])
            #Get points before and after current
            for indexAdjustment in range(1, halfOfAmountFloored + 1):
                currPointRtValues.append(currRegion[currIndex - indexAdjustment])
                currPointRtValues.append(currRegion[currIndex + indexAdjustment])
            
            currRegionNormalisedRt.append(mean(currPointRtValues))
        
        normalisedRt.append(currRegionNormalisedRt)

    return normalisedRt

def prepareDataRegressor(filepath, dataset):
    """
    Prepare data for use with Regressor Models

    INPUTS:
        :param filepath: String, the location of the actions .csv
        :param dataset: String, which dataset is currently being used
    
    OUTPUTS:
        returns four 2D Lists and two 2D Dataframes: 
                Lists, two each for the training and test set containing data 
                (series of actions) and the label that needs to be predicted (next action).
                Dataframe, a portion of the original dataset that has been split off to be used for testing,
                ensuring that the model has not seen this portion.
    """
    compiledData = read_csv(filepath)

    #compiledData = compiledData.sample(frac=0.4)

    coreData = compiledData.drop(["Region No", "Group No", "Action to Next Point"], axis=1)
    goalData = compiledData["Action to Next Point"]

    kFold = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)
    
    coreData, testCoreUnaltered, goalData, testGoalUnaltered = train_test_split(coreData, goalData, test_size=0.2, random_state=10, stratify=goalData)
    
    ros = RandomOverSampler()
    #rus = RandomUnderSampler()
    #kms = KMeansSMOTE(k_neighbors=7)
    #svm = SVMSMOTE()
    #tl = TomekLinks(sampling_strategy="majority")
    #ed = EditedNearestNeighbours()
    #oss = OneSidedSelection(n_neighbors=4, n_seeds_S=300)
    #ncr = NeighbourhoodCleaningRule(n_neighbors=7, kind_sel="mode", threshold_cleaning=0.5)
    #nm = NearMiss(version=3, n_neighbors=7, n_neighbors_ver3=7)
    #smote = SMOTE()

    #coreData, goalData = ros.fit_resample(coreData, goalData)
    #coreData, goalData = rus.fit_resample(coreData, goalData)

    combined = deepcopy(coreData)
    combined["Action to Next Point"] = goalData
    combined.to_csv(f"../../data/core/{dataset}/predictor/tree_balanced.csv")
    
    testingData = deepcopy(testCoreUnaltered)
    testingData["Action to Next Point"] = testGoalUnaltered
    testingData.to_csv(f"../../data/core/{dataset}/predictor/tree_test_orig.csv")

    coreData = coreData.to_numpy()
    goalData = goalData.to_numpy()

    #coreData, goalData = resample(coreData, goalData, stratify=goalData, )
    
    split = kFold.split(coreData, goalData)
    trainingCore, trainingGoal = [], []
    testingCore, testingGoal = [], []

    for currTraining, currTest in split:
        trainingCore.append(coreData[currTraining])
        trainingGoal.append(goalData[currTraining])
        testingCore.append(coreData[currTest])
        testingGoal.append(goalData[currTest])

    return trainingCore, trainingGoal, testingCore, testingGoal, testCoreUnaltered,  testGoalUnaltered

def runRegressors(trainingData, trainingActions, testData, testActions):
    """
    Run various Regressors on the training data, then determine its accuracy with test set.

    INPUTS:
        :param trainingData: 2D Numpy of Integers, containing the sequence of actions that was taken by the agent
        :param trainingActions: 1D Numpy of Integers, containing the action that occurs after the sequence to be predicted
        :param testData: 2D Numpy of Integers, containing the sequence of actions that was taken by the agent
        :param testActions: 1D Numpy of Integers, containing the action that occurs after the sequence to be predicted

    OUTPUT:
        returns a list of the models that has been trained on the training data, score and accuracy for each model.
    """

    regressorRF = RandomForestRegressor(bootstrap=True, verbose=0, criterion="mse", max_features="auto", oob_score=True, max_samples=0.1)
    regressorBR = BaggingRegressor(bootstrap=True, n_estimators=20)
    regressorGB = GradientBoostingRegressor(max_depth=4, loss="lad", learning_rate=0.1)
    regressorGB2 = GradientBoostingRegressor(max_depth=5, loss="quantile", learning_rate=0.1, alpha=0.6)
    regressorXG = XGBRFRegressor()        

    regressorRF.fit(trainingData, trainingActions)
    regressorBR.fit(trainingData, trainingActions)
    regressorGB.fit(trainingData, trainingActions)
    regressorGB2.fit(trainingData, trainingActions)
    regressorXG.fit(trainingData, trainingActions)

    score = []
    score.append(round(regressorRF.score(testData, testActions), 2))
    score.append(round(regressorBR.score(testData, testActions), 2))
    score.append(round(regressorGB.score(testData, testActions), 2))
    score.append(round(regressorGB2.score(testData, testActions), 2))
    score.append(round(regressorXG.score(testData, testActions), 2))

    predictionsRF = regressorRF.predict(testData)
    predictionsBR = regressorBR.predict(testData)
    predictionsGB = regressorGB.predict(testData)
    predictionsGB2 = regressorGB2.predict(testData)
    predicitionsXG = regressorXG.predict(testData)

    print("-")

    predictionsRF = np.rint(predictionsRF)
    predictionsBR = np.rint(predictionsBR)
    predictionsGB = np.rint(predictionsGB)
    predictionsGB2 = np.rint(predictionsGB2)
    predicitionsXG = np.rint(predicitionsXG)

    accuracy = []
    accuracy.append(round(accuracy_score(testActions, predictionsRF), 2))
    accuracy.append(round(accuracy_score(testActions, predictionsBR), 2))
    accuracy.append(round(accuracy_score(testActions, predictionsGB), 2))
    accuracy.append(round(accuracy_score(testActions, predictionsGB2), 2))
    accuracy.append(round(accuracy_score(testActions, predicitionsXG), 2))

    confuse = confusion_matrix(testActions, np.array(predictionsBR).astype(int))

    print(confuse)
    models = [regressorRF, regressorBR, regressorGB, regressorGB2, regressorXG]
    print("")
    return models, score, accuracy

def runBaggingRegressor(trainingData, trainingActions, testData, testActions):
    """
    Train a Bagging Regressor, then evaluate using the Test data.

    INPUTS:
        :param trainingData: 2D Numpy of Integers, containing the sequence of actions that was taken by the agent
        :param trainingActions: 1D Numpy of Integers, containing the action that occurs after the sequence to be predicted
        :param testData: 2D Numpy of Integers, containing the sequence of actions that was taken by the agent
        :param testActions: 1D Numpy of Integers, containing the action that occurs after the sequence to be predicted

    OUTPUT:
        returns the trained model, the score for the model's predictions, the accuracy of the model's predictions and the confusion matrix
    """
    regressorBR = BaggingRegressor(bootstrap=True, n_estimators=15)
    regressorBR.fit(trainingData, trainingActions)

    score = round(regressorBR.score(testData, testActions), 2)

    predictionsBR = regressorBR.predict(testData)
    predictionsBR = np.rint(predictionsBR)

    accuracy = round(accuracy_score(testActions, predictionsBR), 2)

    confuse = confusion_matrix(testActions, np.array(predictionsBR).astype(int))

    return regressorBR, score, accuracy, confuse 

def main():
    windowSizes = [5, 7, 10]
    dataset = "fn"
    
    startYear = 1984
    endYear = 2021

    #filepath = "../../data/core/uk/2. Rt/uk_Rt.csv"
    filepath = "../../data/core/fn/raw/financial.csv"
    regionValues = readFile(dataset, filepath, startYear, endYear)

    #regionValues = generateSineData(12, 300)

    regionValues = normalise(regionValues, 3)

    valueChange = getValueChange(regionValues)

    valueChange = convertValues(valueChange)

    regionalPotentialActionLists = getPotentialActionLists(valueChange)
    
    regionalAgentResultsTree = runAgent(regionalPotentialActionLists, regionValues, dataset, "tree", startYear)
    regionalAgentResultsGreed = runAgent(regionalPotentialActionLists, regionValues, dataset, "greedy", startYear)

    agentCumulativeDifferenceTree = 0
    agentCumulativeDifferenceGreedy = 0

    for currRegionIndex in range(len(regionValues)):
        for currPoint in range(len(regionValues[currRegionIndex])):
            origVal = regionValues[currRegionIndex][currPoint]
            agentValTree, _ = regionalAgentResultsTree[currRegionIndex][currPoint]
            agentValGreedy, _ = regionalAgentResultsGreed[currRegionIndex][currPoint]

            agentCumulativeDifferenceTree += abs(origVal - agentValTree)
            agentCumulativeDifferenceGreedy += abs(origVal - agentValGreedy)

    print("Tree: ", agentCumulativeDifferenceTree / len(regionValues))
    print("Greedy: ", agentCumulativeDifferenceGreedy / len(regionValues)) 

    print("----") 

    regionalEvaluationGroupsGreed = runEvaluationAgent(regionalPotentialActionLists, regionValues, regionalAgentResultsGreed)
    regionalGroupResultsGreed = evalutateAgentPerformance(regionValues, regionalAgentResultsGreed, regionalEvaluationGroupsGreed)
    
    regionalEvaluationGroupsTree = runEvaluationAgent(regionalPotentialActionLists, regionValues, regionalAgentResultsTree, "tree")
    regionalGroupResultsTree = evalutateAgentPerformance(regionValues, regionalAgentResultsTree, regionalEvaluationGroupsTree)

    
    for curr in windowSizes:
        filePathTree = saveResults(regionValues, regionalAgentResultsTree, regionalEvaluationGroupsTree, regionalGroupResultsTree, regionalPotentialActionLists, "tree", dataset, curr)
        filePathGreed = saveResults(regionValues, regionalAgentResultsGreed, regionalEvaluationGroupsGreed, regionalGroupResultsGreed, regionalPotentialActionLists, "greedy", dataset, curr)
   
    """
    LSTM
    """
    """
    history = []
    groupOverallAccuracy = 0
    noIterations = 10
    for _ in range(noIterations):
        #trainingData, trainingAction, testData, testAction, validData, validAction = prepareDataUnalteredLSTM(regionValues, 7)
        trainingData, trainingAction, testData, testAction, validData, validAction = prepareDataLSTM(f"../../data/core/{dataset}/predictor/tree_7.csv", dataset)
        model, currHistory = testLSTM(trainingData, trainingAction, validData, validAction, dataset, "boop")
        
        history.append(currHistory)
        resultsAccuracy, overallAccuracy = determineAccuracy(model, testData, testAction, True)

        groupOverallAccuracy += round(float(overallAccuracy), 2)

    print(resultsAccuracy, " | ", groupOverallAccuracy/noIterations)
    
    for currHistory in history:
        plt.plot(currHistory.history['loss'])
    plt.title('Model Training')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    #plt.legend(['validation'], loc='upper right')
    plt.savefig(f"../../images/machine learning/predictor/{dataset}/{curr}-Training-{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.png")
    plt.close()

    for currHistory in history:
        plt.plot(currHistory.history['val_loss'])
    plt.title('Model Validation')
    plt.ylabel('val_loss')
    plt.xlabel('epoch')
    #plt.legend(['validation'], loc='upper right')
    plt.savefig(f"../../images/machine learning/predictor/{dataset}/{curr}-Validation-{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.png")
    plt.close()
    existingResults = []

    if firstRun != True:
        with open(filePath, "r") as dataFile:
            myReader = csv.DictReader(dataFile)

            for row in myReader:
                existingResults.append(row)
    """
    """
    Regressors
    """

    for curr in windowSizes:
        print(f"-----{curr}-----")
        totalScore = [[],[], [], [], []]
        totalAccuracy = [[], [], [], [], []]
        totalAccuracyOrig = [[], [], [], [], []] 

        trainingData, trainingActions, testData, testActions, origCore, origGoal = prepareDataRegressor(f"../../data/core/{dataset}/predictor/tree_{curr}.csv", dataset)
                
        for curr in range(len(trainingData)):
            
            model, score, accuracy = runRegressors(trainingData[curr], trainingActions[curr], testData[curr], testActions[curr])
            
            for currIndex in range(len(accuracy)):
                totalScore[currIndex].append(score[currIndex])
                totalAccuracy[currIndex].append(accuracy[currIndex])

            for curr in range(len(model)):
                currModel = model[curr]

                predictions = currModel.predict(origCore.to_numpy())
                predictions = np.rint(predictions)

                currAccuracy = round(accuracy_score(origGoal, predictions), 2)
                print("Current Orig Data Accuracy: ", currAccuracy)
                totalAccuracyOrig[curr].append(currAccuracy)
                confuse = confusion_matrix(origGoal, np.array(predictions).astype(int))
                
                #Which model to print the confuse for
                if curr == 2:
                    print(confuse)
                
            
        print("Unaltered Test Data accuracy:")
        for curr in totalAccuracyOrig:
            print(round(mean(curr), 2))

        print("===========")
        for currIndex in range(len(totalAccuracy)):
            print(round(mean(totalScore[currIndex]), 2), " | ", round(mean(totalAccuracy[currIndex]), 2))
        print("===========")

if __name__ == "__main__":
    sys.exit(main())
