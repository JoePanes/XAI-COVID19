from sklearn.ensemble import RandomForestClassifier
from csv import reader
from random import shuffle
from random import randint
from copy import deepcopy
from itertools import chain


filePath = "../../data/core/uk/final/uk_final.csv"


def chunks(list, noChunks):
    """
    Taken from
    https://stackoverflow.com/a/54802737
    Yield n number of striped chunks from l.
    
    INPUTS:
        :param list: List of anything, splits the list into equal (or as equal as possible) chunks between noChunks
        :param noChunks: Integer, the number of chunks (sublists) to be created from list
    
    OUTPUT:
        returns a list of lists, where each sublist is an equal sized chunk of the original list
    """
    for i in range(0, noChunks):
        yield list[i::noChunks]


#Read in the file
compiledData = []
rtData = []
readLabel = True
with open(filePath, "r") as dataFile:
    myReader = reader(dataFile)

    for row in myReader:
        if readLabel:
            readLabel = False
            continue
        compiledData.append(row)

shuffle(compiledData)
shuffle(compiledData)

#Split compiledData into equally sized chunks        
splitCompiledData = list(chunks(compiledData,5))

for _ in range(2):
    shuffle(splitCompiledData)

    for currIndex in range(len(splitCompiledData)):
        shuffle(splitCompiledData[currIndex])

classifer = RandomForestClassifier(bootstrap=False, criterion="gini")


splitData = deepcopy(splitCompiledData)

testData = splitData.pop(randint(1, len(splitData))-1)

testRt = [row[-1] for row in testData]

trainingData = []
for currChunk in splitData:
    for currRow in currChunk:
        trainingData += [currRow]
trainingRt = [row[-1] for row in trainingData]

classifer.fit(trainingData, trainingRt)

print(classifer.score(testData, testRt))