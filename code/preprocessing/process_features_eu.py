"""
Used to process the EU dataset (not compatible with the UK dataset)


"""
import csv
import sys
from process_features import processFeatures

COUNTRIES = {
    "albania" : 1,
    "austria" : 2,
    "czech republic" : 3,
    "denmark" : 4,
    "france" : 5,
    "germany" : 6,
    "hungary" : 7,
    "italy" : 8,
    "moldova" : 9,
    "poland" : 10,
    "portugal" : 11,
    "serbia" : 12,
    "slovakia" : 13,
    "slovenia" : 14,
    "spain" : 15,
    "switzerland" : 16,
    "uk" : 17,
}

def setupCategoriedFields():
    """
    Extend the constant such that it contains both the original values and
    values specific to the UK database

    INPUTS:
        NONE
    
    OUTPUTS:
        returns the merged dictionary
    """
    category = processFeatures.CATEGORISED_FIELDS
    category.update({"countries" : COUNTRIES})

    return category

class processFeaturesEU(processFeatures):
    CATEGORISED_FIELDS = setupCategoriedFields()

    INPUT_FILE = "/eu/raw/eu_data.csv"
    OUTPUT_FILE = "/eu/processed/eu_mlTable_0_0.csv"       

if __name__ == '__main__':
    run = processFeaturesEU()
    sys.exit(run.main()) 