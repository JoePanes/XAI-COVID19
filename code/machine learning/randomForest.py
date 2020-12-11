import xgboost
import os

from pandas import read_csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from random import randint
def prepareData(filePath):
    """
    Read in a .csv file, then prepare the data for use with the Random Forest

    INPUT:
        :param filePath: String, the location of the final pre-processed dataset
    
    OUTPUT:
        returns four separate Pandas Dataframes, these are the training set, test set, then the corresponding Rt (labels) for both.
    """
    #Read in the file
    compiledData = read_csv(filePath)

    splitTrainData, splitTestData = train_test_split(compiledData, test_size=0.2, random_state=randint(1, os.getpid()))

    #Extract Rt from the data
    trainingData = splitTrainData.drop(["Rt"], axis=1)
    trainingRt = splitTrainData["Rt"]

    testData = splitTestData.drop(["Rt"], axis=1)
    testRt = splitTestData["Rt"]

    return trainingData, trainingRt, testData, testRt

def runRandomForest(trainingData, trainingRt, testData, testRt):
    """
    Using the training data, train a Random Forest Classifer, then test the model it hasn't
    been exposed to before.

    INPUT:
        :param trainingData: Pandas Dataframe, the main chunk of the dataset's core data containing control measures, cases, etc.
        :param trainingRt: Pandas Dataframe, the Rt values for the training data records.
        :param testData: Pandas Dataframe, the remainder of the dataset, containing the same core data fields as training data
        :param testRt: Pandas Dataframe, the Rt values for the test data records.

    OUTPUT:
        returns the classifer and the accuracy score from the model using the test data.
    """
    classifer = RandomForestClassifier(bootstrap=True, random_state=os.getpid(), criterion="gini", n_estimators=25)

    classifer.fit(trainingData, trainingRt)

    accuracyScore = classifer.score(testData, testRt)

    return classifer, accuracyScore