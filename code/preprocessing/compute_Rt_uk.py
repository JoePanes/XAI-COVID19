from compute_Rt import computeRt
from shared.sharedVariables import FILE_PATH_CORE

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
        "Non-Essential Shops" : "Binary",
        "Hospitals/ Care and Nursing Home Visits" : "Trinary",
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

    def writeFile(self, dataset):
        newDataset = []

        for row in dataset:
            newDataset.insert(0, row)
        with open(FILE_PATH_CORE + self.DATASET + self.OUTPUT_FILE, "w") as optFile:
            
            labels = {}
            for currFieldName in newDataset[0].keys():
                labels[currFieldName] = currFieldName
            
            myWriter = csv.DictWriter(optFile, labels)
            
            myWriter.writerow(labels)
            
            for row in newDataset:
                myWriter.writerow(row)
run = computeRtUK()
run.main()