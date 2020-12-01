"""
Child class of process_features
Used to process the UK dataset (not compatible with the EU dataset)


"""
import csv
import sys
from process_features import processFeatures

REGIONS = {
    "east midlands" : 1,
    "east of england" : 2,
    "london" : 3,
    "north east" : 4,
    "north west" : 5,
    "south east" : 6,
    "south west" : 7,
    "west midlands" : 8,
    "yorkshire and the humber" : 9,
    "northern ireland" : 10,
    "scotland" : 11,
    "blaenau gwent" : 12,
    "caerphilly" : 13,
    "monmouthshire" : 14,
    "newport" : 15,
    "torfaen" : 16,
    "conwy" : 17,
    "denbighshire" : 18,
    "flintshire" : 19,
    "gwynedd" : 20,
    "isle of anglesey" : 21,
    "wrexham" : 22,
    "cardiff" : 23,
    "vale of glamorgan" : 24,
    "bridgend" : 25,
    "merthyr tydfil" : 26,
    "rhondda cynon taf" : 27,
    "carmarthenshire" : 28,
    "ceredigion" : 29,
    "pembrokeshire" : 30,
    "powys" : 31,
    "neath port talbot" : 32,
    "swansea" : 33,
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
    category.update(
        {
            "regions" : REGIONS,
            "hospitals/ care and nursing home visits" : processFeatures.SEVERITY,
            "non-essential shops" : processFeatures.CLOSURE,

        })

    return category

class processFeaturesUK(processFeatures):

    CATEGORISED_FIELDS = setupCategoriedFields()
    
    INPUT_FILE = "/uk/raw/uk_data.csv"
    OUTPUT_FILE = "/uk/processed/uk_mlTable_0_0.csv"
    OUTPUT_ERROR = "/uk/errors/"
    
    def getRecordsRegion(self, row):
        """
        Retrieve the record's Region

        INPUT:
            :param row: Dictionary, the contents of the current row of the dataset

        OUTPUT:
            returns the current locations name
        """

        return row["Regions"]

if __name__ == '__main__':
    run = processFeaturesUK()
    sys.exit(run.main()) 