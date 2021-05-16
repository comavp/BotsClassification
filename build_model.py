import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from models import Models


pathToData = 'dataAfterProcessingCSV/'
pathToPictures = 'pictures/'
pathToModels = 'models/'
models = Models().models


def print_model(modelResultPath):
    modelResult = joblib.load(pathToModels + modelResultPath)
    print(modelResult)


def show_confusion_matrix(cm_test, title):
    print('Results for \"' + str(title) + '\"')
    print('------------------------------------')
    plt.figure(figsize=(8,8))
    sns.heatmap(cm_test, annot=True, fmt="d", cmap="Blues")
    sns.set(font_scale=1.9)
    plt.title(title)
    plt.ylabel("Actual ")
    plt.xlabel("Predicted ")
    plt.tight_layout()
    plt.savefig(pathToPictures + title + ".png")
    plt.show()


def get_confusion_matrix(X_test, y_test, classifier):
    y_pred = classifier.predict(X_test)
    # print(y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    # print(matrix)
    print("Accuracy Score:", accuracy_score(y_test, y_pred))
    print("Precision Score : ", precision_score(y_test, y_pred))
    print("Recall Score:", recall_score(y_test, y_pred))
    print("f1 Score:", f1_score(y_test, y_pred))
    return matrix


def split_data(pandas_data):
    X = pandas_data.loc[:, 'statuses_count':'description_length']
    y = pandas_data['is_bot']
    return X, y


def read_data():
    bots = pd.read_csv(pathToData + 'botsAfterProcessing.csv')
    real = pd.read_csv(pathToData + 'humansAfterProcessing.csv')
    test_bots = pd.read_csv(pathToData + 'politicalBotsAfterProcessing.csv')
    test_real = pd.read_csv(pathToData + 'celebritiesAfterProcessing.csv')

    data = pd.concat((bots, real), axis=0)
    data.to_excel('bots_and_humans.xls')
    data = data.sample(frac=1)
    data.to_excel('bots_and_humans_mixed.xls')
    return data


def build_and_evaluate_models(data):
    X, y = split_data(data)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    numberOfAllAccounts = np.array(y_train).size
    numberOfBots = np.count_nonzero(np.array(y_train))
    numberOfHumans = numberOfAllAccounts - numberOfBots
    print('Number of bots in train data set: ' + str(numberOfBots))
    print('Number of humans in train data set: ' + str(numberOfHumans))
    print('Bots: {0}%, humans: {1}%'.format(numberOfBots/numberOfAllAccounts*100, numberOfHumans/numberOfAllAccounts*100))
    print('')

    for name, (model, parameters) in models.items():
        print('Results for \"' + str(name) + '\"')
        gs = GridSearchCV(model, parameters, cv=10, verbose=0, n_jobs=-1, scoring='f1')
        gs.fit(X_train, y_train)
        print("Best Parameters:", gs.best_params_)
        print("")
        print("Best Score:", gs.best_score_)

        joblib.dump(gs.best_estimator_, pathToModels + f"{name}.pkl", compress=1)
        joblib.dump(gs.cv_results_, pathToModels + f"{name}_results.pkl", compress=1)

        y_pred = gs.predict(X_test)

        print("")
        model = model.fit(X_train, y_train)
        show_confusion_matrix(get_confusion_matrix(X_test, y_test, model), name)
        print('------------------------------------')


if __name__ == '__main__':
    data = read_data()
    build_and_evaluate_models(data)