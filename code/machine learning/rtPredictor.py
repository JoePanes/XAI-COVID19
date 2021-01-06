import sys
import multiprocessing

from csv import DictReader
from pandas import qcut
from pandas import DataFrame
from statistics import mean
from matplotlib import pyplot as plt
from copy import deepcopy


QUANTILE_NO = 8
GROUP_SIZE = 3
MAX_DEPTH = GROUP_SIZE - 1

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

filePath = "../../data/core/uk/Rt/uk_Rt.csv"

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
        self.cost = None
        #Just something to make returning to the currently expanded node easier to find
        #when backtracking up the tree
        self.isExpandedPoint = False
        
        currDifference = self.currVal - origRt

        if currDifference < 0:
            currDifference = -currDifference

        if parent != None:
            self.cumulativeDifference = parent.cumulativeDifference + currDifference
            self.cost = parent.cost + 1
        else:
            self.cumulativeDifference = currDifference
            self.cost = 0


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

def treeSearch(regionRt, potentialActions):
    """
    Performs a tree search out the MAX_DEPTH to select the best (according to the heuristic) action to take

    INPUTS:
        :param regionRt: List of floats, the regional data that will be explored by the tree search
        :param potentialActions: List of floats, containing floats to be used to alter the Rt value of the previous point
        
    
    OUTPUT:
        returns a list of tuples, containing the Rt value and the index for the action that resulted in said Rt value
    """
    agentRtValues = []
    expandedStates = []
    leafState = []

    currPoint = 1
    currDepth = 0
    for currRootIndex in range(len(regionRt)):
        #Set up the inital point that we are finding the action for
        root = Node(regionRt[currRootIndex], 0, 0, regionRt[currRootIndex])
        root.isExpandedPoint = True

        #Get child nodes of the root
        for actionIndex in range(len(potentialActions)):
            action = potentialActions[actionIndex]

            expandedStates.append(Node(regionRt[root.cost], action, actionIndex, root.currVal, root))
        
        #Explore the child nodes to the max depth
        while MAX_DEPTH > currDepth or currPoint + currDepth >= len(regionRt):
            currDepth += 1
            print("--boop---")

            while len(expandedStates) > 0:
                currentNode = expandedStates.pop(0)
                print(len(expandedStates)," | ", len(leafState))
                
                for actionIndex in range(len(potentialActions)):
                    action = potentialActions[actionIndex]
                    leafState.append(Node(regionRt[currPoint + currentNode.cost + 1], action, actionIndex, currentNode.currVal, currentNode))
            #Once all child nodes of the current level have been expanded, swap    
            expandedStates, leafState = leafState, expandedStates
            leafState = []

        expandedStates.sort(key=getCumulativeDifference)
        currentNode = expandedStates.pop(0)

        while currentNode.parent.isExpandedPoint is False:
            currentNode = currentNode.parent
        
        agentRtValues.append((currentNode.currVal, currentNode.actionIndex))
    
    print(agentRtValues)
    print(currentNode.cumulativeDifference)
    print(currentNode.currVal)

    return agentRtValues

    """
    1. It needs to iterate over the entire region
    
    2. While doing so, it needs to use the point prior to the current as the root

    3. Stop expanding nodes when it has fully expanded the max depth from the initial node
        3.1 Once it reaches this state it needs to only make its decision based upon the nodes at the max depth
            all other nodes should be removed through the process of expansion, rather than needing to be removed
            afterwards.


    Where each O is a point within the dataset:

    O - O - O - O - O
    
    The first point in the dataset is the root, since there is no proceeding point,
    nothing is done to it.
    [] - O - O - O - O

    We're looking at finding which move it make from the root to the next point
    []-> O - O - O - O
                /\
        With a depth of 3, we would be expanding ahead to here
    
    Expand the node, such that we have a queue of potential action that can be performed on the current state,
    leading to various levels of how close each will lead to the desired Rt value (the original).
        0
      /
    [] - 0 - O - O - O
      \  
        0

    Expand these point and so on, until the max depth is reached
           0 ...
          /
         0 - 0 ...
        /  \
       /     0 ...
      /      0 ...
     |      /
    [] -- 0 - 0 ... - O
     |      \
      \      0 ...
       \    0 ...
        \ /
         0 - 0 ...   
          \
            0 ...

    Once expanded to the max depth, retain only the nodes at the max-depth, all intermediate nodes should be not present, this is due to the end nodes then being sorted,
    based upon the cumulative difference in the Rt value over the course of the chosen actions to the current node. 
    
    Then, the end node that has the least cumulative difference is selected from all the end nodes and proceed to backtrack to the root from the chosen end node to conclude on which action
    should be taken.
    
            
       -<-0--<--0-
      /            \
    []               <-0--<-0
      
      /\
    This action would be selected

    Store the result of this decision making, then proceed to the next point
        1
    [] ->- [X] - O - O - O
    
    Now, using the result of previous steps, we move the root node one point forward
        1
    [] ->- [X] - O - O - O

    Repeat the previous steps until all points within the region have been run through with the agent
        1       0       2       1
    [] ->- [0] ->- [0] ->- [0] ->- [0]

    """


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
    regionalAgentResults = []
    for currIndex in range(len(regionRt)):
        currActionList = potentialActionLists[currIndex]
        currRegionRt = regionRt[currIndex]

        print(len(currActionList))
        if agentType.lower() == "greedy":
            agentResults = greedyAgent(currRegionRt, currActionList)
        
        elif agentType.lower() == "tree":
            agentResults = treeSearch(currRegionRt, currActionList)
        print(agentResults)
        agentRt = [currRt for currRt, _ in agentResults]

        createGraph(currRegionRt, agentRt, currIndex, agentType)
        regionalAgentResults.append(agentResults)
        
    return regionalAgentResults

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

                if agentType.lower() == "greedy":
                    agentResults = greedyAgent(currRegionSubset, potentialActionLists[currRegionIndex], firstValue=False, 
                                                evaluatePoint=True, evaluateIndex=0, evaluateAction=agentAction)
                #Remove the value since it is no longer needed
                currRegionSubset.pop(0)
                
                agentRt, _ = agentResults[-1]

                distanceFromGoal = regionRt[currRegionIndex][goalIndex] - agentRt
                #Add the result to current grouping
                currGroup.append((agentResults, distanceFromGoal))
            
            if ignoreCurrentValues == False:
                regionEvaluationGroups[currRegionIndex].append(((startIndex, goalIndex), currGroup))

    return regionEvaluationGroups

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

            if agentRtDifference < 0:
                agentRtDifference = -agentRtDifference
            currentlyImpactfulPoints = []
            prevDifference = 0

            for currIndex in range(goalIndex - startIndex):
                currEvalAgentsDifference = group[currIndex][1]
                if currEvalAgentsDifference < 0:
                    currEvalAgentsDifference = -currEvalAgentsDifference

                #Reduce the evaluation agents Rt difference with the original's
                agentDifference = currEvalAgentsDifference - agentRtDifference

                if agentDifference < 0:
                    agentDifference = -agentDifference
                
                if prevDifference < agentDifference:
                    prevDifference = agentDifference
                    currentlyImpactfulPoints = [currIndex]
                
                elif prevDifference == agentDifference:
                    currentlyImpactfulPoints.append(currIndex)

            currRegionGroupResults.append((currentlyImpactfulPoints, prevDifference))
        
        regionalGroupResults.append(currRegionGroupResults)
    
    return regionalGroupResults

def main():

    regionRt = readFile("uk",filePath)

    rtValueChange = getRtValueChange(regionRt)

    rtValueChange = convertRtValues(rtValueChange)

    regionalPotentialActionLists = getPotentialActionLists(rtValueChange)
    
    regionalAgentResults = runAgent(regionalPotentialActionLists, regionRt, "tree")

    print("----") 
    """
    regionalEvaluationGroups = runEvaluationAgent(regionalPotentialActionLists, regionRt, regionalAgentResults)

    regionalGroupResults = evalutateAgentPerformance(regionRt, regionalAgentResults, regionalEvaluationGroups)
                
    for currRegion in regionalGroupResults[11]:
        #for currGroup in currRegion:
        print(currRegion)
    """

    """
    For tree search algorithm:
    
    Have it so that the metric used for deciding what is the best move is based upon the cumulative result of adding the difference in Rt value
    between the dataset and the potential actions.
    Where this can be seen as the cost, whereby the aim is to find a series of actions that result in the minimal total cumululative difference.
    Other ways focusing achieving the best goal value or max depth value, without regard for the intermediate will be useless.

    Handle evaluation in the same way as the Greedy agent, where only one action is prevented in the decision making process, with the same method of steping forward
    to progress through the points and use the previous agent Rt value as a starting point.

    Due to the need to look deeper, insure that there are an extra Depth - 1 elements at the end or to the end of the list

    Insure that the algorithm is either flexible enough to handle reaching the end of the file during earlier levels of its search, or that there is a method of adjusting the depth as needed
    to accomodate this event.

    Use the group size as the depth for the trees (or atleast be dependent on it)?

    Like in the A* search use a sort with a key to find the best series of actions
    """
    
if __name__ == "__main__":
    sys.exit(main())
