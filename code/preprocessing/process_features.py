import csv
class processFeatures:
    FILE_PATH = "../../data/core/"
    INPUT_FILE = None
    
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

    def main(self):
        pass

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
    
    def discretizeWeather(self, temp, humidity):
        """
        Convert the float values of temp and humidity into an integer indicating which threshold
        they exist within.

        INPUTS:
            :param currLine: Dictionary, the data stored about the current row to be written
            :param temp: Float, the average temperature for a given day
            :param humidity: Float, the average humidity for a given day

        OUTPUTS:
            returns the number for the threshold which each fit into
        """
        discreteTemp = self.discretizeVal(temp, self.TEMP_THRESHOLD)

        discreteHumid = self.discretizeVal(humidity, self.HUMIDITY_THRESHOLD)

        return discreteTemp, discreteHumid

