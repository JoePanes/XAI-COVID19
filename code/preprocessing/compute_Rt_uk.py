"""
Child class of compute_Rt
Contains the variations of functions and variables necessary to compute using the UK dataset, then output
as desired.
"""

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
    INPUT_FILE = "/1. unprocessed/uk_data.csv"
    OUTPUT_FILE = "/2. Rt/uk_Rt.csv"
    OUTPUT_ERROR = "/uk/errors/"

    CONTROL_MEASURES = setControlMeasures()

    REGIONAL_FIELD_NAME = "Regions"
    
    def getRegion(self, currRow):
        """
        Retrieves the current region name from the row

        INPUT:
            :param currRow: Dictionary, a row from the /core/processed/ dataset
        
        OUTPUT:
            returns an integer, indicating which categorised region it is
        """
        return currRow.get("Regions")

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

        #Reorder to match original
        for row in dataset:
            newDataset.insert(0, row)

        with open(FILE_PATH_CORE + self.DATASET + fileName, "w") as optFile:
            
            labels = {}
            for currFieldName in newDataset[0].keys():
                labels[currFieldName] = currFieldName
            
            reorderedLabels = self.orderFields(labels, containRt)
            myWriter = csv.DictWriter(optFile, reorderedLabels)
            
            myWriter.writerow(labels)
            
            for row in newDataset:
                myWriter.writerow(row)

if __name__ == '__main__':
    run = computeRtUK()
    run.main()