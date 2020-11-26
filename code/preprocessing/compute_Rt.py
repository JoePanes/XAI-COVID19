import sys
from info.general import FILE_PATH_CORE

class computeRt():
        
    OUTPUT_FOLDER = None

    def main(self):
        """
        Over the course of its functions, it will calculate the Rt for each day,
        then add them as a new column to the dataset, and save it to a new .csv file.

        INPUT:
            NONE
        
        OUTPUT:
            returns nothing, but creates a new .csv file in the folder /Rt/ of the given
            dataset

        """
        #Retrieve dataset being used from command line
        dataset = sys.argv[0][-5:-3]

        optDataList = []
        oldLabel = None



