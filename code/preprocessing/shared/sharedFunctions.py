"""
A place to keep functions that are used by more than one file
"""
from datetime import datetime
from sys import stdout
from csv import DictReader

#Taken from https://stackoverflow.com/a/14598135
def removeBrackets(fieldName):
    """
    Removes any bracketed text from a string

    INPUT:
        :param fieldName: String, the field name being checked
    OUPUT:
        returns a string that contains only text not within a bracket
    """
    ret = ''
    skip1c = 0
    skip2c = 0
    for i in fieldName:
        if i == '[':
            skip1c += 1
        elif i == '(':
            skip2c += 1
        elif i == ']' and skip1c > 0:
            skip1c -= 1
        elif i == ')'and skip2c > 0:
            skip2c -= 1
        elif skip1c == 0 and skip2c == 0:
            ret += i
    return ret

def createErrorFile(filePath, chosenDataset, dataErrorCount, errorList):
    """
    Create a nice way of outputting where errors exist in the datasets without
    clogging up the command line, and will persist in its own file.

    INPUTS:
        :param filePath: String, the area within the file structure to be stored at
        :param chosenDataset: String, what dataset has produced these errors
        :param dataErrorCount: Integer, the total number of errors
        :param errorList: List of Strings, information on where and what caused the error
    
    OUTPUT:
        returns nothing, but creates a text file to keep record of the errors
    """
    fileName = chosenDataset+ "_" + datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

    print(f"There were {dataErrorCount} errors when reading the data.")
    print(f"Go to '{filePath}{chosenDataset}/errors/{fileName}' for the dates where these occured.")

    fileName = chosenDataset+ "_" + datetime.now().strftime("%d_%m_%Y_%H_%M_%S_.%f")

    errorOutput = open(f"{filePath}{chosenDataset}/errors/{fileName}.txt", "x")

    errorText = f"Total errors {dataErrorCount}\n\n"
    for currError in errorList:
        errorText += currError + "\n"
        
    errorOutput.write(errorText)

def printProgressBar(currNo, totalNo):
    """
    Calculate the current progress through the files and print a visual representation

    INPUT:
        :param currNo: Integer, the number of files that have been looked at already
        :param totalNo: Integer, the total number of files in the list of files
    
    OUTPUT:
        returns nothing, but, prints the current progress out to the command line
    """
    #Calculate the percentage (reduced by x10 to map to progress bar)
    progress = int((currNo / totalNo) * 10)

    progressBar = ["-"]*10
    for i in range(0, progress):
        progressBar[i] = "#"


    progressBar = "".join(progressBar)

    output = "Progress: ["+ progressBar + f"] {currNo} / {totalNo}"

    #Copied from https://stackoverflow.com/a/6361028
    stdout.write("\r\x1b[K"+output)
    stdout.flush()

def readFile(dataset, filePath):
    """
    Takes in the contents of the file, and compiles it into
    a usable format

    INPUT:
        :param dataset: String, the name of the dataset being read (determines the order)
        :param filePath: String, the location of the .csv file to be read from
    
    OUTPUT:
        returns a list of dictionaries, containing the contents of the dataset
    """
    compiledData = []

    with open(filePath, "r") as dataFile:
        myReader = DictReader(dataFile)

        for row in myReader:
            if dataset == "eu":
                compiledData.append(row)
            else:
                compiledData.insert(0, row)

    return compiledData