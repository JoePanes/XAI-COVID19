from datetime import datetime

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

    fileName = chosenDataset+ "_" + datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

    errorOutput = open(f"{filePath}{chosenDataset}/errors/{fileName}.txt", "x")

    errorText = f"Total errors {dataErrorCount}\n\n"
    for currError in errorList:
        errorText += currError + "\n"
        
    errorOutput.write(errorText)