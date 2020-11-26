from compute_Rt import computeRt

class computeRtUK(computeRt):
    INPUT_FILE = "/processed/uk_mlTable_0_0.csv"
    OUTPUT_FILE = "/Rt/uk_Rt.csv"
    OUTPUT_ERROR = "/uk/errors/"

    def getRegion(self, currRow):
        """
        Retrieves the current region name from the row
        """
        return currRow.get("Regions")


run = computeRtUK()
run.main()