import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from models import Models


pathToData = 'dataAfterProcessingCSV/'
models = Models().models


def print_result(cm_test, title):
    print('Results for \"' + str(title) + '\"')
    print('------------------------------------')
    plt.figure(figsize=(8,8))
    sns.heatmap(cm_test, annot=True, fmt="d", cmap="Blues")
    sns.set(font_scale=1.9)
    plt.title(title)
    plt.ylabel("Actual ")
    plt.xlabel("Predicted ")
    plt.tight_layout()
    plt.savefig(title + ".png")
    plt.show()


def test(X_test, y_test, classifier):
    y_pred = classifier.predict(X_test)
    # print(y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    # print(matrix)
    print("Accuracy Score:", accuracy_score(y_test, y_pred))
    print("Precision Score : ", precision_score(y_test, y_pred))
    print("Recall Score:", recall_score(y_test, y_pred, zero_division=1))
    print("f1 Score:", f1_score(y_test, y_pred))
    return matrix


def splitData(pandas_data):
    X = pandas_data.loc[:, 'statuses_count':'description_length']
    y = pandas_data['is_bot']
    return X, y


bots = pd.read_csv(pathToData + 'botsAfterProcessing.csv')
real = pd.read_csv(pathToData + 'humansAfterProcessing.csv')
test_bots = pd.read_csv(pathToData + 'politicalBotsAfterProcessing.csv')
test_real = pd.read_csv(pathToData + 'celebritiesAfterProcessing.csv')

data = pd.concat((bots, real), axis=0)
data.to_excel('bots_and_humans.xls')

X, y = splitData(data)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
numberOfAllAccounts = np.array(y_train).size
numberOfHumans = np.count_nonzero(np.array(y_train))
numberOfBots = numberOfAllAccounts - numberOfHumans
print('Number of bots in train data set: ' + str(numberOfBots))
print('Number of humans in train data set: ' + str(numberOfHumans))
print('Bots: {0}%, humans: {1}%'.format(numberOfBots/numberOfAllAccounts*100, numberOfHumans/numberOfAllAccounts*100))
print('')

# decisionTree = DecisionTreeClassifier().fit(X_train, y_train)
# dummyClassifier = DummyClassifier(strategy='stratified', random_state=432).fit(X_train, y_train)
#
# print_result(test(X_test, y_test, decisionTree), 'First test DecisionTree')
# print_result(test(X_test, y_test, dummyClassifier), 'First test Dummy')
#
# X_test, y_test = splitData(test_bots)
# print_result(test(X_test, y_test, decisionTree), 'Political bots DecisionTree')
# print_result(test(X_test, y_test, dummyClassifier), 'Political bots Dummy')
#
# X_test, y_test = splitData(test_real)
# print_result(test(X_test, y_test, decisionTree), 'Celebrities DecisionTree')
# print_result(test(X_test, y_test, dummyClassifier), 'Celebrities Dummy')

for name, (model, parameters) in models.items():
    print('Results for \"' + str(name) + '\"')
    model = model.fit(X_train, y_train)
    test(X_test, y_test, model)
    print('------------------------------------')