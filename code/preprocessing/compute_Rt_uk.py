from compute_Rt import computeRt
from shared.sharedVariables import FILE_PATH_CORE
from shared.sharedFunctions import removeBrackets

import csv
def setControlMeasures():
    """
    Imports the control measures from the parent and
    adds on the ones specific to dataset

    INPUT:
        NONE
    
    OUTPUT:
        returns a completed list of the control measures present within the dataset
    """
    controlMeasures = computeRt.CONTROL_MEASURES
    
    controlMeasures.update({
        "Non-Essential Shops" : ("Binary", ["Opened", "Closed"]),
        "Hospitals/ Care and Nursing Home Visits" : ("Trinary", ["Low", "Moderate", "High"]),
    })

    return controlMeasures

class computeRtUK(computeRt):
    INPUT_FILE = "/processed/uk_mlTable_0_0.csv"
    OUTPUT_FILE = "/Rt/uk_Rt.csv"
    OUTPUT_ERROR = "/uk/errors/"

    CONTROL_MEASURES = setControlMeasures()

    def getRegion(self, currRow):
        """
        Retrieves the current region name from the row

        INPUT:
            :param currRow: Dictionary, a row from the /core/processed/ dataset
        
        OUTPUT:
            returns an integer, indicating which categorised region it is
        """
        return currRow.get("Regions")

    def readFile(self):
        """
        Takes in the contents of the file, and compiles it into
        a usable format

        INPUT:
            NONE
        
        OUTPUT:
            returns a list of dictionaries, containing the contents of the dataset
        """
        processedData = []

        with open(f"{FILE_PATH_CORE}{self.DATASET}{self.INPUT_FILE}", "r") as dataFile:
            myReader = csv.DictReader(dataFile)

            for row in myReader:
                processedData.insert(0, row)

        return processedData

    def writeFile(self, dataset, fileName, containRt = False):
        """
        Writes the current state of the dataset to a .csv file, (along with reorder the dataset
        inorder to match the desired format).

        INPUTS:
            :param dataset: List of dictionaries, where each entry in the list is a row from the dataset
            :param fileName: String, what the name of the outputted file will be
            :param containRt: Boolean, OPTIONAL, whether the current state of the dataset contains an Rt row

        OUTPUT:
            returns nothing, but creates/overwrites a .csv file of the given filename
        """
        newDataset = []

        for row in dataset:
            newDataset.insert(0, row)
        with open(FILE_PATH_CORE + self.DATASET + fileName, "w") as optFile:
            
            labels = {}
            for currFieldName in newDataset[0].keys():
                labels[currFieldName] = currFieldName
            
            desiredOrder = ["Day", "Date", "Regions", "Cases", "Cumulative Cases", "Deaths", "Cumulative Deaths",  "Tests", "Cumulative Tests", "Temperature", "Humidity"]

            #Insert control measures between deaths and temp
            insertIndex = 9
            
            
            for currControlMeasure in self.CONTROL_MEASURES:
                controlType = self.CONTROL_MEASURES.get(currControlMeasure)

                if controlType[0] is "Trinary":
                    for currLevel in controlType[1]:
                        desiredOrder.insert(insertIndex, f"{currControlMeasure} ({currLevel})")
                        insertIndex += 1
                
                elif controlType[0] is "Binary":
                    desiredOrder.insert(insertIndex, currControlMeasure)
                    insertIndex += 1

            if containRt:
                desiredOrder.extend("Rt")

            #Based on https://stackoverflow.com/a/52044835        
            reorderedLabels = {k : labels[k] for k in desiredOrder}

            myWriter = csv.DictWriter(optFile, reorderedLabels)
            
            myWriter.writerow(labels)
            
            for row in newDataset:
                myWriter.writerow(row)
run = computeRtUK()
run.main()