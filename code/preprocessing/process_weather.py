"""
Processes all of the .csv files present within the raw weather folder
and extracts the temperature and humidity, averages all of their values for a
given date then writes them into the processed folder.

Uses weather data from this website: https://rp5.ru/Weather_archive_in_Chisinau
"""

import csv
import sys
import statistics
import os

filePath = "../../data/weather/raw/"
optFilePath = "../../data/weather/processed/"

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
    #Calculate the percentage
    progress = int((currNo / totalNo) * 10)

    progressBar = ["-"]*10
    for i in range(0, progress):
        progressBar[i] = "#"


    progressBar = "".join(progressBar)
    print("Progress: ["+ progressBar + f"] {currNo} / {totalNo}")
    
def readFile(fileName, encoding):
    """
    Reads in the given file using the specified encoder, then orders the rows to match the desired dataset.

    INPUTS:
        :param fileName: String, the name of the current file to be read
        :param encoding: String, either UTF-8 or cp1252

    OUTPUT:
        returns the rows of the read file, ordered to match the desired dataset.
                Along with the column where temperature and humidity are located
    """

    rawData = []
    with open(filePath + fileName +".csv","r", encoding=encoding) as dataFile:
            #Skip over rows that are not part of the database
            for _ in range(6):
                next(dataFile)

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
                    #Reverse the order of the data to match the EU dataset
                    rawData.insert(0, row)

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
    fileList = os.listdir(filePath)

    inputFiles = []
    for currFile in fileList:
        if currFile[-4::] == ".csv":
            inputFiles.append(currFile[:-4:])
    
    totalFiles = len(inputFiles)
    
    print(f"Found {totalFiles} files")
    
    currFileNo = 1

    for currFileName in inputFiles:
        printProgressBar(currFileNo, totalFiles)
        rawData = []
        
        

        rawData = []

        #Some of the datasets despite claming to be UTF-8 encoded, are not, hence this is necessary
        try:
            rawData, tempIndex, humidityIndex = readFile(currFileName, "UTF-8")
        except:
            rawData, tempIndex, humidityIndex = readFile(currFileName, "cp1252")


        with open(optFilePath + currFileName + "_processed.csv", "w") as optFile:
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
                        This tends to occur when a feature within a dataset has no value
                        
                        However, it seems to be that only one or two readings may have this occur on a specific date
                        and has very little impact.
                        """
                            
            writeToFile(myWriter, currDate, currDateTemp,currDateHumid)
            
        currFileNo +=1 

if __name__ == '__main__':
    sys.exit(main()) 