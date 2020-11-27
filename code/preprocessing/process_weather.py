"""
Processes all of the .csv files present within the raw weather folder
and extracts the temperature and humidity, averages all of their values for a
given date then writes them into the processed folder.

Uses weather data from this website: https://rp5.ru/Weather_archive_in_Chisinau
Click "See on map" to access other weather forecasts and archives
"""

import csv
import sys
import statistics
import os
from shared.sharedFunctions import createErrorFile
from datetime import datetime

filePath = "../../data/weather/"

def writeToFile(fileWriter, date, tempList, humidList):
    """
    Takes the input, processes it and writes the result to the output file

    INPUTS:
        :param fileWriter: csv.DictWriter, the interface to the .csv file
        :param date: String, the current date being added
        :param tempList: List of floats, temperature readings throughout the specific day
        :param humidList: List of floats, humidity readings throughout the specific day

    OUTPUTS:
        returns an optional method of clearing the original input values in one line
            (input values meaning: date, tempList and humidList, NOT fileWriter)
    """
    processedDate = date

    #Average and round to 2 d.p.
    processedTemp = round(statistics.mean(tempList), 2)
    processedHumidity = round(statistics.mean(humidList), 2)

    fileWriter.writerow({
        "Date" : processedDate, 
        "Temp" : processedTemp,
        "Humidity" : processedHumidity,
        }
    )

    return None, [], []

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
    sys.stdout.write("\r\x1b[K"+output)
    sys.stdout.flush()

def prepareRawFile(fileName, dataset, encoding):
    """
    Reads in the .csv file (straight from the website, which is all one column), and prepare it so that it can be processed

    INPUTS:
        :param fileName: String, the name of the current file to be read
        :param dataset: String, the corresponding dataset which the data is being processed for
        :param encoding: String, either UTF-8 or cp1252
    
    OUPUT:
        returns nothing, but splits the data in one column, in to a multicolumn dataset ready for processing.
    """
    rawData = []
    labels = {}
    with open(filePath + dataset + "/raw/" + fileName + ".csv", "r", encoding=encoding) as dataFile:
        #Skip over rows that are not part of the database
        for _ in range(6):
            next(dataFile)

        myReader = csv.reader(dataFile)
        readLabel = True

        for row in myReader:
            #Some rows have their data split randomly over columns
            rowContents = "".join(row)
            rowContents = rowContents.split(";")
            #Remove unnecessary speech/quote marks
            for i in range(len(rowContents)):
                rowContents[i] = rowContents[i].replace('"', '')

            if readLabel:
                for currCol in rowContents:
                    labels[currCol] = currCol

                readLabel = False
            
            rawData.append(rowContents)

    with open(filePath + dataset + "/unprocessed/" + fileName + ".csv", "w", encoding=encoding) as optFile:
        myWriter = csv.DictWriter(optFile, labels)
        
        fieldNames = list(labels.keys())
        for row in rawData:
            formattedEntry = {}
            
            i = 0
            for currCol in fieldNames:
                formattedEntry[currCol] = row[i]
                i += 1

            myWriter.writerow(formattedEntry)

                

def readFile(fileName, dataset, encoding):
    """
    Reads in the given file using the specified encoder, then orders the rows to match the desired dataset.

    INPUTS:
        :param fileName: String, the name of the current file to be read
        :param dataset: String, the corresponding dataset which the data is being processed for
        :param encoding: String, either UTF-8 or cp1252

    OUTPUT:
        returns the rows of the read file, ordered to match the desired dataset.
                Along with the column where temperature and humidity are located
    """

    rawData = []
    with open(filePath + dataset + "/unprocessed/" + fileName +".csv","r", encoding=encoding) as dataFile:
            #Skip over rows that are not part of the database
            """for _ in range(6):
                next(dataFile)"""

            myReader = csv.reader(dataFile)
            readLabel = True
            
            tempIndex = None
            humidityIndex = None

            for row in myReader:
                if readLabel:
                    #Find where the temperature and humidity are within the table
                    i = 0
                    for currElem in row:
                        if currElem == "T":
                            tempIndex = i
                        elif currElem == "U":
                            humidityIndex = i
                        elif tempIndex != None and humidityIndex != None:
                            break
                        i += 1 
                    readLabel = False
                else:
                    if dataset == "eu":
                        #Reverse the order of the data to match the EU dataset
                        rawData.insert(0, row)
                    else:
                        rawData.append(row)

    return rawData, tempIndex, humidityIndex

def main():
    """
    Goes through the weather files present within the raw folder, and outputs the
    avg Temp and Humidity for each day.

    INPUTS:
        NONE
    
    OUTPUTS:
        returns nothing, but, it converts the .csv files in raw into the expected format
            and outputs them in processed
    """
    acceptedDatasets = {
        "uk" : 1,
        "eu" : 2,
    }
    
    if len(sys.argv) != 2 or sys.argv[1].lower() not in acceptedDatasets:
        print("In order to use this program, you need to specify the dataset you are using in two characters")
        print("The options for datasets, are: uk or eu")
        print("Such as, python process_weather.py uk")

        return -1
    
    chosenDataset = sys.argv[1].lower()



    fileList = os.listdir(filePath + chosenDataset + "/raw/")

    inputFiles = []
    for currFile in fileList:
        if currFile[-4::] == ".csv":
            inputFiles.append(currFile[:-4:])
    
    totalFiles = len(inputFiles)
    
    print(f"Found {totalFiles} files")
    
    currFileNo = 1
    dataErrorCount = 0
    errorList = []

    for currFileName in inputFiles:
        printProgressBar(currFileNo, totalFiles)
        
        rawData = []
        encoding = "UTF-8"
        #Some datasets have lied about being UTF-8 encoded, hence the need for this
        try:
            prepareRawFile(currFileName, chosenDataset, encoding)
            
        except:
            encoding = "cp1252"
            prepareRawFile(currFileName, chosenDataset, encoding)

        rawData, tempIndex, humidityIndex = readFile(currFileName, chosenDataset, encoding)

        with open(filePath + chosenDataset + "/processed/" + currFileName + "_processed.csv", "w") as optFile:
            myWriter = csv.DictWriter(optFile, ["Date", "Temp", "Humidity",])

            myWriter.writerow({"Date" : "Date", "Temp" : "Temp", "Humidity" : "Humidity"})
            #Where the values for a date to calculate the average are stored
            currDateTemp = []
            currDateHumid = []
            currDate = None
            for row in rawData:
                if currDate != row[0][0:10] and currDate is not None:
                    currDate, currDateTemp, currDateHumid = writeToFile(myWriter, currDate, currDateTemp, currDateHumid)
                if currDate is None:
                    #Get just the day, month, year
                    currDate = row[0][0:10]

                #Compare just the day, month, year
                if currDate == row[0][0:10]:
                    try:
                        currDateTemp.append(float(row[tempIndex]))
                        currDateHumid.append(float(row[humidityIndex]))
                    except:
                        """
                        This tends to occur when a field (or fields) in a record has no value
                        
                        However, it seems to be that only one or two readings may have this occur on a specific date
                        and has very little impact.
                        """
                        dataErrorCount += 1
                        errorList.append(f"{row[0]} | {currFileName}")

                            
            writeToFile(myWriter, currDate, currDateTemp,currDateHumid)
            
        currFileNo +=1

    #When finished, restore order to the command line     
    print("\n")
    
    if dataErrorCount > 0:
        createErrorFile(filePath, chosenDataset, dataErrorCount, errorList)
if __name__ == '__main__':
    sys.exit(main()) 