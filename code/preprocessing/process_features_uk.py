"""
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
    "wales" : 12,
}




class processFeaturesUK(processFeatures):
    INPUT_FILE = "/uk/raw/uk_data.csv"
    OUTPUT_FILE = "/uk/processed/uk_mlTable_0_0.csv"

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


    def main(self):
        """
        Converts the UK dataset into a format that is usuable in the next stage

        INPUT:
            NONE
        
        OUTPUT:
            returns nothing, but, writes the converted UK dataset to the output .csv file
        """
        
        newLabel = {
            'Day' : "Day",
            'Date' : "Date",
            'Regions' : "Regions",
            'Cases' : "Cases",
            'Cumulative Cases' : "Cumulative Cases",
            'Deaths' : "Deaths", 
            'Cumulative Deaths' : "Cumulative Deaths", 
            'Tests' : "Tests", 
            'Cumulative Tests' : "Cumulative Tests", 
            'Meeting Friends/Family' : "Meeting Friends/Family",
            'Domestic Travel' : "Domestic Travel", 
            'Cafes and Restaurants' : "Cafes and Restaurants",
            'Pubs and Bars' : "Pubs and Bars", 
            'Sports and Leisure' : "Sports and Leisure", 
            'Schools Closure' : "Schools Closure",
            'Temperature' : 'Temperature',
            'Humidity' : 'Humidity'
        }
        #readLabel = True
        
        rawData = []


        with open(self.FILE_PATH + self.INPUT_FILE, "r") as dataFile:
            myReader = csv.DictReader(dataFile)

            for row in myReader:
                """if readLabel:
                    readLabel = False
                    
                    newLabel = list(row.keys())
                    continue"""

                rawData.append(row)

        with open(self.FILE_PATH + self.OUTPUT_FILE, "w") as optFile:
            myWriter = csv.DictWriter(optFile, list(newLabel.keys()))
            #Put the column labels in
            myWriter.writerow(newLabel)

            for row in rawData:
                #Copy over info from the original, and discretize where relevant
                newLine = {}

                newLine["Day"] = row["Day"]

                """
                TO DO:
                    Convert the date?
                """
                newLine["Date"] = row["Date"]

                newLine["Regions"] = REGIONS.get(row["Regions"].lower())

                newLine["Cases"] = row["Cases"]
                newLine["Cumulative Cases"] = row["Cumulative Cases"]
                
                newLine["Deaths"] = row["Deaths"]
                newLine["Cumulative Deaths"] = row["Cumulative Deaths"]
                
                newLine["Tests"] = row["Tests"]
                newLine["Cumulative Tests"] = row["Cumulative Tests"]

                newLine["Meeting Friends/Family"] = self.SEVERITY.get(row["Meeting Friends/Family"].lower())

                newLine["Domestic Travel"] = self.SEVERITY.get(row["Domestic Travel"].lower())

                newLine["Cafes and Restaurants"] = self.SEVERITY.get(row["Cafes and Restaurants"].lower())

                newLine["Pubs and Bars"] = self.SEVERITY.get(row["Pubs and Bars"].lower())

                newLine["Sports and Leisure"] = self.SEVERITY.get(row["Sports and Leisure"].lower())

                newLine["Schools Closure"] = self.CLOSURE.get(row["Schools Closure"].lower())

                """
                For when this is made available in the data set
                newLine["Temperature"] = discretizeVal(row["Temperature"], TEMP_THRESHOLD)

                newLine["Humidity"] = discretizeVal(row["Humidity"], HUMIDITY_THRESHOLD)
                """
                myWriter.writerow(newLine)
                

if __name__ == '__main__':
    run = processFeaturesUK()
    sys.exit(run.main()) 