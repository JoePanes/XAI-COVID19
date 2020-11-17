import csv
import sys
import statistics

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
    rawData = []
    with open(filePath + "uk.csv","r") as dataFile:
        #Skip over rows that are not part of the database
        for _ in range(7):
            next(dataFile)

        myReader = csv.reader(dataFile)
        
        for row in myReader:
            #Reverse the order of the data to match the EU dataset
            rawData.insert(0, row)
        
    with open(optFilePath + "uk" + "_processed.csv", "w") as optFile:
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
                currDateTemp.append(float(row[1]))
                currDateHumid.append(float(row[5]))
                        
        writeToFile(myWriter, currDate, currDateTemp,currDateHumid)
            

if __name__ == '__main__':
    sys.exit(main()) 