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
from sklearn.feature_selection import SequentialFeatureSelector as SFS
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris


pathToData = 'dataAfterProcessingCSV/'
pathToPictures = 'pictures/'
pathToModels = 'models/'
pathToRetrainModel = 'retrainModels/'
pathToRetrainPictures = 'retrainPictures/'
models = Models().models


def print_model(modelResultPath):
    modelResult = joblib.load(pathToModels + modelResultPath)
    print(modelResult)


def show_confusion_matrix(cm_test, title):
    print('Print confusion matrix for \"' + str(title) + '\"')
    plt.figure(figsize=(8,8))
    sns.heatmap(cm_test, annot=True, fmt="d", cmap="Blues")
    sns.set(font_scale=1.9)
    plt.title(title)
    plt.ylabel("Actual ")
    plt.xlabel("Predicted ")
    plt.tight_layout()
    plt.savefig(pathToPictures + title + ".png")
    #plt.show()
    plt.close()


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

    bestScore = 0
    bestAlgName = ''

    for name, (model, parameters) in models.items():
        print('Results for \"' + str(name) + '\"')
        gs = GridSearchCV(model, parameters, cv=10, verbose=1, n_jobs=-1, scoring='roc_auc')
        gs.fit(X_train, y_train)
        print("Best Parameters:", gs.best_params_)
        print("")
        print("Best Score:", gs.best_score_)

        joblib.dump(gs.best_estimator_, pathToModels + f"{name}.pkl", compress=1)
        joblib.dump(gs.cv_results_, pathToModels + f"{name}_results.pkl", compress=1)

        y_pred = gs.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        if bestScore < f1:
            bestScore = f1
            bestAlgName = name

        print("")
        show_confusion_matrix(get_confusion_matrix(X_test, y_test, gs), name)
        print('------------------------------------')

    print('"' + bestAlgName + '" ' + 'showed best results on test dataset')
    print('f1 score: ' + str(bestScore))
    return bestAlgName


def print_bots_and_humans_count(y):
    numberOfAllAccounts = np.array(y).size
    numberOfBots = np.count_nonzero(np.array(y))
    numberOfHumans = numberOfAllAccounts - numberOfBots
    print('Number of bots in train data set: ' + str(numberOfBots))
    print('Number of humans in train data set: ' + str(numberOfHumans))
    print('------------------------------------')


def test_retrain_model_on_unseen_data(model_name, list_to_drop):
    political_bots = pd.read_csv(pathToData + 'politicalBotsAfterProcessing.csv')
    real = pd.read_csv(pathToData + 'celebritiesAfterProcessing.csv')
    pron_bots = pd.read_csv(pathToData + 'pronBotsAfterProcessing.csv')
    vendor_bots = pd.read_csv(pathToData + 'vendorBotsAfterProcessing.csv')

    real_and_political_bots = pd.concat((political_bots, real), axis=0).sample(frac=1)
    real_and_pron_bots = pd.concat((pron_bots, real), axis=0).sample(frac=1)
    real_and_vendor_bots = pd.concat((vendor_bots, real), axis=0).sample(frac=1)

    best_model = joblib.load(pathToRetrainModel + model_name + '.pkl')

    X1, y1 = split_data(real_and_political_bots)
    X1.drop(X1.columns[list_to_drop], axis=1, inplace=True)
    show_confusion_matrix(get_confusion_matrix(X1, y1, best_model), model_name + 'CelebritiesAndPoliticalBots')
    print_bots_and_humans_count(y1)

    X2, y2 = split_data(real_and_pron_bots)
    X2.drop(X2.columns[list_to_drop], axis=1, inplace=True)
    show_confusion_matrix(get_confusion_matrix(X2, y2, best_model), model_name + 'CelebritiesAndPronBots')
    print_bots_and_humans_count(y2)

    X3, y3 = split_data(real_and_vendor_bots)
    X3.drop(X3.columns[list_to_drop], axis=1, inplace=True)
    show_confusion_matrix(get_confusion_matrix(X3, y3, best_model), model_name + 'CelebritiesAndVendorBots')
    print_bots_and_humans_count(y3)


def test_model_on_unseen_data(model_name):
    political_bots = pd.read_csv(pathToData + 'politicalBotsAfterProcessing.csv')
    real = pd.read_csv(pathToData + 'celebritiesAfterProcessing.csv')
    pron_bots = pd.read_csv(pathToData + 'pronBotsAfterProcessing.csv')
    vendor_bots = pd.read_csv(pathToData + 'vendorBotsAfterProcessing.csv')

    #unseen_data = pd.concat((political_bots, real, pron_bots, vendor_bots), axis=0).sample(frac=1)
    #unseen_data.to_excel('unseen_data.xls')

    real_and_political_bots = pd.concat((political_bots, real), axis=0).sample(frac=1)
    real_and_pron_bots = pd.concat((pron_bots, real), axis=0).sample(frac=1)
    real_and_vendor_bots = pd.concat((vendor_bots, real), axis=0).sample(frac=1)

    best_model = joblib.load(pathToModels + model_name + '.pkl')

    X1, y1 = split_data(real_and_political_bots)
    show_confusion_matrix(get_confusion_matrix(X1, y1, best_model), model_name + 'CelebritiesAndPoliticalBots')
    print_bots_and_humans_count(y1)

    X2, y2 = split_data(real_and_pron_bots)
    show_confusion_matrix(get_confusion_matrix(X2, y2, best_model), model_name + 'CelebritiesAndPronBots')
    print_bots_and_humans_count(y2)

    X3, y3 = split_data(real_and_vendor_bots)
    show_confusion_matrix(get_confusion_matrix(X3, y3, best_model), model_name + 'CelebritiesAndVendorBots')
    print_bots_and_humans_count(y3)


def select_features(model_name):
    data = read_data()
    X, y = split_data(data)
    model = joblib.load(pathToModels + model_name + '.pkl')

    all_features = X.columns.values
    print('Исходный набор:' + str(all_features))
    print('--------------------------------')

    print('First')
    sfs1 = SFS(model, n_features_to_select=16, direction='backward', scoring='f1', cv=10)
    result1 = sfs1.fit(X, y)
    part_of_features1 = all_features[result1.get_support(indices=True)]
    print(result1.get_support())
    print(result1.get_support(indices=True))
    print(str(part_of_features1))
    print('--------------------------------')

    print('Second')
    sfs2 = SFS(model, n_features_to_select=16, direction='forward', scoring='f1', cv=10)
    result2 = sfs2.fit(X, y)
    part_of_features2 = all_features[result2.get_support(indices=True)]
    print(result2.get_support())
    print(result2.get_support(indices=True))
    print(str(part_of_features2))
    print('--------------------------------')

    print('Third')
    sfs3 = SFS(model, direction='forward', scoring='f1', cv=10)
    result3 = sfs3.fit(X, y)
    part_of_features3 = all_features[result3.get_support(indices=True)]
    print(result3.get_support())
    print(result3.get_support(indices=True))
    print(str(part_of_features3))
    print('--------------------------------')

    print('Fourth')
    sfs4 = SFS(model, direction='backward', scoring='f1', cv=10)
    result4 = sfs4.fit(X, y)
    part_of_features4 = all_features[result4.get_support(indices=True)]
    print(result4.get_support())
    print(result4.get_support(indices=True))
    print(str(part_of_features4))


def retrain_model(data, model_name, lists_to_drop):
    cnt = 1
    name = model_name

    for l_to_drop in lists_to_drop:
        bestScore = 0
        bestAlgName = ''

        X, y = split_data(data)
        X.drop(X.columns[l_to_drop], axis=1, inplace=True)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        (model, parameters) = models[model_name]
        name = model_name + str(cnt)
        print('Results for \"' + str(name) + '\"')
        gs = GridSearchCV(model, parameters, cv=10, verbose=1, n_jobs=-1, scoring='roc_auc')
        gs.fit(X_train, y_train)
        print("Best Parameters:", gs.best_params_)
        print("")
        print("Best Score:", gs.best_score_)

        joblib.dump(gs.best_estimator_, pathToRetrainModel + f"{name}.pkl", compress=1)
        joblib.dump(gs.cv_results_, pathToRetrainModel + f"{name}_results.pkl", compress=1)

        y_pred = gs.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        if bestScore < f1:
            bestScore = f1
            bestAlgName = name

        print("")
        show_confusion_matrix(get_confusion_matrix(X_test, y_test, gs), name)
        print('------------------------------------')

        cnt += 1

    print('"' + bestAlgName + '" ' + 'showed best results on retrain datasets')
    print('f1 score: ' + str(bestScore))


if __name__ == '__main__':
    first_list = [0, 5, 10, 12]
    second_list = [12, 14, 15, 19]
    third_list = [0, 2, 3, 4, 8, 10, 14, 15, 17, 19]
    fourth_list = [1, 2, 5, 6, 9, 11, 12, 15, 16, 18]
    parse_list = [3, 4, 12, 13]
    list = [first_list, second_list, third_list, fourth_list]

    data = read_data()
    #retrain_model(data, "RandomForest", list)
    #build_and_evaluate_models(data)
    #for model_name in models.keys():
    #    test_model_on_unseen_data(model_name)
    #select_features('RandomForest')

    # for i in range(1, 5):
    #     name = "RandomForest" + str(i)
    #     test_retrain_model_on_unseen_data(name, list[i - 1])

    retrain_model(data, "RandomForest", [parse_list])
    test_retrain_model_on_unseen_data("RandomForest1", parse_list)
