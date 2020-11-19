class processFeatures():
    FILE_PATH = "../../data/core/"
    
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

    def readFile(self):
        pass

    def main(self):
        pass