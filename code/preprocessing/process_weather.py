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

    processedTemp = statistics.mean(tempList)
    processedHumidity = statistics.mean(humidList)

    fileWriter.writerow({
        "Date" : processedDate, 
        "Temp" : processedTemp,
        "Humidity" : processedHumidity,
        }
    )

    return None, [], []

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

    print(inputFiles)
    
    for currFileName in inputFiles:
        print(f"-----{currFileName}-----")
        rawData = []
        
        tempIndex = None
        humidityIndex = None

        with open(filePath + currFileName +".csv","r") as dataFile:
            #Skip over rows that are not part of the database
            for _ in range(6):
                next(dataFile)

            myReader = csv.reader(dataFile)
            readLabel = True
            
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
                        print(f"Invalid data at: {row[0]} in Temp (T) or Humididty (U) column")
                            
            writeToFile(myWriter, currDate, currDateTemp,currDateHumid)
            

if __name__ == '__main__':
    sys.exit(main()) 