import joblib

modelPath = 'models/'
decisionTreeName = 'DecisionTree'
dummyClassifierName = 'DummyClassifier'
nearestNeighborsName = 'K-Nearest Neighbors'
resultsName = '_results'
pklName = '.pkl'


def print_model(modelResultName):
    modelResult = joblib.load(modelPath + modelResultName)


if __name__=='__main__':
    print(modelPath + dummyClassifierName + pklName)
    model = joblib.load(modelPath + dummyClassifierName + pklName)