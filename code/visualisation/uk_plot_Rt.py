"""
Takes in the UK dataset from the Rt folder, then plots each row within each region on a chart.
Using the features of the New (avg) Cases calculated previously, and the Rt number from each row.
"""
import csv
from matplotlib import pyplot as plt
import pandas as pd

filePath = "../../data/core/uk/2. Rt/uk_Rt.csv"

outputFilePath = "../../images/Rt/uk"

REGIONS = {
    0 : "East Midlands",
    1 : "East of England",
    2 : "London",
    3 : "North East",
    4 : "North West",
    5 : "South East",
    6 : "South West",
    7 : "West Midlands",
    8 : "Yorkshire and the Humber",
    9 : "Northern Ireland",
    10 : "Scotland",
    11 : "Blaenau gwent",
    12 : "Caerphilly",
    13 : "Monmouthshire",
    14 : "Newport",
    15 : "Torfaen",
    16 : "Conwy",
    17 : "Denbighshire",
    18 : "Flintshire",
    19 : "Gwynedd",
    20 : "Isle of Anglesey",
    21 : "Wrexham",
    22 : "Cardiff",
    23 : "Vale of Glamorgan",
    24 : "Bridgend",
    25 : "Merthyr Tydfil",
    26 :"Rhondda Cynon Taf",
    27 : "Carmarthenshire",
    28 :"Ceredigion",
    29 : "Pembrokeshire",
    30 : "Powys",
    31 : "Neath Port Talbot",
    32 : "Swansea",
}

def getRegionalIndexs(dataset):
        """
        From the dataset, get the start and end indexes for each region

        INPUT:
            :param dataset: List of Dictionaries, where each entry is a row from the dataset
        
        OUTPUT:
            returns a list of tuples, which specify the start and end index range of each region in the
            dataset
        """
        currStartIndexNo = None
        currRegion = None
        regionalRangeList = []
        for currIndex in range(len(dataset)):
            
            if currRegion != dataset[currIndex]["Regions"]:
                if currRegion != None:
                    regionalRangeList.append((currStartIndexNo, currIndex))
                
                currRegion = dataset[currIndex]["Regions"]
                currStartIndexNo = currIndex            
        
        #It misses (due to the if) one region at the end, therefore, add it on
        currIndex = regionalRangeList[-1][1]
        regionalRangeList.append((currIndex + 1, len(dataset)))

        return regionalRangeList

rtData = []
with open(filePath, "r") as dataFile:
    myReader = csv.DictReader(dataFile)
    
    for row in myReader:
        rtData.insert(0, row)

regionIndexs = getRegionalIndexs(rtData)

#Breaks the list into regional sublists
splitList = []
for currStart, currEnd in regionIndexs:
    splitList.append(rtData[currStart:currEnd])

#Iterate through the regional sublists, then plot the Rt for each day with the avg. new cases on that day
for currRegion in range(len(splitList)):
    regionalRts = []
    regionalNewCases = []
    for currRow in splitList[currRegion]:
        
        regionalRts.append(float(currRow["Rt"]))
        regionalNewCases.append(float(currRow["Cases"]))

    #Code from Xiyui Fan, adjusted for the dataset
    width = .4
    m1_t = pd.DataFrame({'Daily New Cases' : regionalNewCases, 'Rt' : regionalRts})
    ax1 = m1_t[['Daily New Cases']].plot(kind='bar', width = width, figsize=(12,3))
    ax2 = m1_t['Rt'].plot(secondary_y=True, color='red', label='Rt')

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.xlim([-width, len(m1_t['Daily New Cases'])-width])

    xticks = [int(len(regionalNewCases)/10)*k for k in range(10)]

    ax1.set_xticks(xticks)
    ax1.set_xticklabels([str(x) for x in xticks])

    ax1.set_xlabel('Days after Total Confirmed Case Reaches 20')
    ax1.set_ylabel('Number of New Cases')
    ax2.set_ylabel('Rt', rotation=-90)
    
    ax1.set_title(REGIONS[currRegion])

    plt.tight_layout()
    plt.savefig(f"{outputFilePath}/{currRegion}.png")



    #plt.show()