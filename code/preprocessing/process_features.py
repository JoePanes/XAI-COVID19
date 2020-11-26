import csv
import os
import output_data_error
import sys

class processFeatures:
    FILE_PATH = "../../data/core/"
    INPUT_FILE = None
    OUTPUT_FILE = None
    OUTPUT_ERROR = None
    
    SEVERITY = {
    "low" : 1,
    "moderate" : 2,
    "high" : 3,
    }

    CLOSURE = {
    "opened" : 0,
    "closed" : 1,
    }
    
    TEMP_THRESHOLD = [-float("inf"), 0, 10, 20, float("inf")]
    HUMIDITY_THRESHOLD = [0, 40, 80, float("inf")]

    #Some shared fields need to categorised from their string values
    CATEGORISED_FIELDS = {
        "meeting friends/family" : SEVERITY,
        "domestic travel" : SEVERITY,
        "cafes and restaurants" : SEVERITY,
        "pubs and bars" : SEVERITY,
        "sports and leisure" : SEVERITY,
        "schools closure" : CLOSURE,
        "international travel" : SEVERITY,
    }

    DISCRETIZE_FIELDS = {
        "temperature" : TEMP_THRESHOLD,
        "humidity" : HUMIDITY_THRESHOLD,
     }

    errorList = []
    errorNo = 0

    def getFieldNames(self):
        """
        Read the first row of the file and retrieve its field names

        INPUT:
            NONE
        
        OUPUT:
            returns the field names for the database
        """
        databaseFields = {}
        
        with open(self.FILE_PATH + self.INPUT_FILE, "r") as dataFile:
            myReader = csv.reader(dataFile)
            readLabel = True

            for row in myReader:
                if readLabel:
                    readLabel = False
                    
                    for currFieldName in row:
                        databaseFields[currFieldName] = currFieldName
                break
        
        return databaseFields

    def readFile(self):
        """
        Retrieve the data from the database and put into a usable format

        INPUT:
            NONE
        
        OUTPUT:
            returns the database all placed into a list of dictionaries
        """
        rawData = []

        with open(self.FILE_PATH + self.INPUT_FILE, "r") as dataFile:
            myReader = csv.DictReader(dataFile)

            for row in myReader:
                rawData.append(row)

        return rawData

    def gatherData(self, row, fieldNames):
        """
        Iterate over a row, retrieve information to be written as it goes
        and converting strings to their numerical equivalent.

        INPUT:
            :param row: Dictionary, one row from the current dataset
            :param fieldNames: List, the list of all fields present within the dataset

        OUTPUT:
            returns the processed row ready for writing to the output file
        """
        newLine = {}
        for currField in fieldNames:
            try:
                data = row[currField]
                standardFieldName = self.removeBrackets(currField).rstrip()

                # Certain fields require processing according to certain rules 
                # and in different ways
                if standardFieldName.lower() in self.CATEGORISED_FIELDS:
                    categoryConversion = self.CATEGORISED_FIELDS.get(standardFieldName.lower())
                    newLine[currField] = categoryConversion[data.lower()]
                
                elif standardFieldName.lower() in self.DISCRETIZE_FIELDS:
                    discreteConversion = self.DISCRETIZE_FIELDS.get(currField.lower())
                    newLine[currField] = self.discretizeVal(data, discreteConversion)

                else:
                    newLine[currField] = data
            except:
                #If invalid data, do not include it
                #If missing data, carry on
                errorText = self.generateErrorEntry(row, currField)

                self.errorList.append(errorText)
                self.errorNo += 1

        return newLine
    
    #Taken from https://stackoverflow.com/a/14598135
    def removeBrackets(self, fieldName):
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

    def discretizeVal(self, val, THRESHOLDS):
        """
        Finds where the current value belongs amongst the corresponding list thresholds

        INPUTS:
            :param val: Float, the value from a feature that needs to be discreatized
            :param THRESHOLDS: Constant, a call to one of the constants that exist above this function (e.g. TEMP_THRESHOLD)
        
        OUTPUT:
            returns the index value in which the value meets the criteria for

        Copied from code by Xiyui Fan
        """
        for i in range(len(THRESHOLDS)-1):
            if float(val) >= THRESHOLDS[i] and float(val) <= THRESHOLDS[i+1]:
                return i
    
    def generateErrorEntry(self, row, currField):
        """
        Specifies where within the dataset the error has occured

        INPUT:
            :param row: Dictionary, current row of the dataset
            :param currField: String, specifically which column the invalid data exists
        
        OUPUT:
            returns the text entry to be added to the error file
        """
        date = row["Date"]
        if len(row["Date"]) == 9:
            date = "0" + date

        errorText = f"{date} | "
        fieldNamePadding = [" "]*40
        
        i = 0
        for letter in currField:
            fieldNamePadding[i] = letter
            i +=1
        
        currField = "".join(fieldNamePadding)
        errorText += f"{currField} | "

        location = self.getRecordsRegion(row)

        errorText = errorText + location
        
        return errorText
        
    def getRecordsRegion(self, row):
        return -1 

    def main(self):
        """
        Converts the dataset into a format that is usuable in the next stage

        INPUT:
            NONE
        
        OUTPUT:
            returns nothing, but, writes the converted UK dataset to the output .csv file
        """
       
        fieldNames = self.getFieldNames()
        rawData = self.readFile()
        
        with open(self.FILE_PATH + self.OUTPUT_FILE, "w") as optFile:
            myWriter = csv.DictWriter(optFile, list(fieldNames.keys()))
            #Put the column labels in
            myWriter.writerow(fieldNames)

            for row in rawData:
                #Copy over info from the original, and discretize where relevant
                newLine = self.gatherData(row, fieldNames)
                myWriter.writerow(newLine)

        if self.errorNo > 0:
            #Get the current dataset from the command used
            dataset = sys.argv[0][-5:-3]
            output_data_error.createErrorFile(self.FILE_PATH, dataset, 
                                              self.errorNo, self.errorList)

if __name__ == '__main__':
    #In the event that the user runs this file
    inp = input("Do you wish to process either the EU, UK, WA dataset? ")

    os.system(f"python process_features_{inp.lower()}.py")
