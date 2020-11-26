"""
Used to process the Wales dataset (not compatible with the EU or uk dataset)


"""
import csv
import sys
from process_features import processFeatures

REGIONS = {
    "blaenau gwent" : 1,
    "bridgend" : 2,
    "caerphilly" : 3,
    "cardiff" : 4,
    "carmarthenshire" : 5,
    "ceredigion" : 6,
    "conwy" : 7,
    "denbighshire" : 8,
    "flintshire" : 9,
    "gwynedd" : 10,
    "isle of anglesey" : 11,
    "merthyr tydfil" : 12,
    "monmouthshire" : 13,
    "neath port talbot" : 14,
    "newport" : 15,
    "pembrokeshire" : 16,
    "powys" : 17,
    "rhondda cynon taf" : 18,
    "swansea" : 19,
    "torfaen" : 20,
    "vale of glamorgan" : 21,
    "wrexham" : 22,
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
            "regions in wales" : REGIONS,
            "hospitals/ care and nursing home visits" : processFeatures.SEVERITY,
            "non-essential shops" : processFeatures.CLOSURE,

        })

    return category

class processFeaturesWA(processFeatures):

    CATEGORISED_FIELDS = setupCategoriedFields()
    
    INPUT_FILE = "/wa/raw/wa_data.csv"
    OUTPUT_FILE = "/wa/processed/wa_mlTable_0_0.csv"
    OUTPUT_ERROR = "/wa/errors/"
    
    def getRecordsRegion(self, row):
        """
        Retrieve the record's Region

        INPUT:
            :param row: Dictionary, the contents of the current row of the dataset

        OUTPUT:
            returns the current locations name
        """

        return row["Regions in Wales"]

if __name__ == '__main__':
    run = processFeaturesWA()
    sys.exit(run.main()) 