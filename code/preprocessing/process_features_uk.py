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

    def main(self):
        """
        Converts the UK dataset into a format that is usuable in the next stage

        INPUT:
            NONE
        
        OUTPUT:
            returns nothing, but, writes the converted UK dataset to the output .csv file
        """
       
        fieldNames = self.getFieldNames()
        rawData = self.readFile()
        


        with open(self.FILE_PATH + self.OUTPUT_FILE, "w") as optFile:
            myWriter = csv.DictWriter(optFile, list(fieldNames.keys()))
            #Put the column labels in
            myWriter.writerow(fieldNames)

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
                try:
                    newLine["Temperature"], newLine["Humidity"] = self.discretizeWeather(row["Temperature"], row["Humidity"])
                except:
                    #In this event there exists no value within the current entry
                    pass
                """
                myWriter.writerow(newLine)
                

if __name__ == '__main__':
    run = processFeaturesUK()
    sys.exit(run.main()) 