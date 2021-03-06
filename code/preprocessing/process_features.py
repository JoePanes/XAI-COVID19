"""
Parent class of process_features_eu and process_features_uk

Takes in the raw datasets (as in, unaltered from the more human focused presentation of data)
Then, converts the control measures from strings, into their numerical translation
Also, discretizes relevant data such as Temperature and Humidity (Case count, deaths and tests may be added at some point)
Finally, writes the altered dataset to a new .csv file.
"""

import csv
import os
import sys

from shared.sharedFunctions import createErrorFile
from shared.sharedFunctions import removeBrackets
from shared.sharedFunctions import readFile
from shared.sharedFunctions import discretizeVal
from shared.sharedVariables import FILE_PATH_CORE

class processFeatures:
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
        "school closure" : CLOSURE,
        "international travel" : SEVERITY,
    }

    DISCRETIZE_FIELDS = {
        "temperature" : TEMP_THRESHOLD,
        "humidity" : HUMIDITY_THRESHOLD,
    }

    NOT_WANTED_FIELDS = {}

    errorList = []
    errorNo = 0

    def getFieldNames(self):
        """
        Read the first row of the file and retrieve the field names that are wanted (useful for machine learning).

        INPUT:
            NONE
        
        OUPUT:
            returns the field names for the database
        """
        databaseFields = {}
        
        with open(FILE_PATH_CORE + self.INPUT_FILE, "r") as dataFile:
            myReader = csv.reader(dataFile)
            readLabel = True

            for row in myReader:
                if readLabel:
                    readLabel = False
                    
                    for currFieldName in row:
                        if currFieldName in self.NOT_WANTED_FIELDS:
                            continue
                        databaseFields[currFieldName] = currFieldName
                break
        
        return databaseFields

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
                standardFieldName = removeBrackets(currField).rstrip()

                # Certain fields require processing according to certain rules 
                # and in different ways
                if standardFieldName.lower() in self.CATEGORISED_FIELDS:
                    categoryConversion = self.CATEGORISED_FIELDS.get(standardFieldName.lower())
                    newLine[currField] = categoryConversion[data.lower()]
                
                elif standardFieldName.lower() in self.DISCRETIZE_FIELDS:
                    discreteConversion = self.DISCRETIZE_FIELDS.get(currField.lower())
                    newLine[currField] = discretizeVal(data, discreteConversion)

                else:
                    try:
                        newLine[currField] = int(data)
                    except:
                        if currField == "Tests":
                            #replace with the case number (due to these being obtained through testing)
                            newLine[currField] = int(row["Cases"])
                            row["Cumulative Tests"] = row["Cumulative Cases"]
                        else:
                            raise
                        

            except:
                #If invalid data, do not include it
                #If missing data, carry on
                errorText = self.generateErrorEntry(row, currField)

                self.errorList.append(errorText)
                self.errorNo += 1

        return newLine
    
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

    def convertDate(self, date):
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
        dataset = sys.argv[0][-5:-3]
        rawData = readFile(dataset,f"{FILE_PATH_CORE + self.INPUT_FILE}")
        
        outputData = []
        for row in rawData:
            #Retain info from the original, and discretize where relevant
            newLine = self.gatherData(row, fieldNames)

            if dataset == "eu":
                outputData.append(newLine)
            else:
                outputData.insert(0, newLine)
                
        with open(FILE_PATH_CORE + self.OUTPUT_FILE, "w") as optFile:
            myWriter = csv.DictWriter(optFile, list(fieldNames.keys()))
            #Put the column labels in
            myWriter.writerow(fieldNames)

            for newLine in outputData:
                myWriter.writerow(newLine)

        if self.errorNo > 0:
            #Get the current dataset from the command used
            dataset = sys.argv[0][-5:-3]
            createErrorFile(FILE_PATH_CORE, dataset, self.errorNo, self.errorList)

if __name__ == '__main__':
    #In the event that the user runs this file
    dataset = input("Do you wish to process either the EU or UK dataset? ")

    dataset = dataset.lower()
    os.system(f"python process_features_{dataset}.py")

    inp = input("Do you want to calculate Rt? (y/n) ")

    if inp.lower() == "y" or inp.lower() == "yes":
        os.system(f"python compute_Rt_{dataset}.py")

        print("Performing final pre-processing on dataset...")

        os.system(f"python discretize_features.py uk")

    print("Done")
    print("Thank you")
    
    exit()

