from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


class Models:
	def __init__(self):
		self.models = {
			"RandomForest": (
				RandomForestClassifier(),
				{
					"criterion": ['gini', 'entropy'],
					"n_estimators": [50, 100, 200],
					"max_features": [None, 'sqrt', 0.3, 0.4, 0.5, 0.6]
				}
			),
			"DecisionTree": (
				DecisionTreeClassifier(),
				{
					"max_depth": [None, 4, 10],
					"min_samples_leaf": [1, 10, 20],
					"min_samples_split": [1.0, 10, 20, 30, 40],
					"max_features": ['sqrt', 'log2', None, 0.2, 0.3]
				}
			),
			"SVM linear": (
				LinearSVC(),
				{
					"C": [0.001, 0.1, 1, 5, 10, 50, 100]
				}
			),
			"SVM RBF": (
				SVC(),
				{
					"C": [0.001, 0.1, 1, 5, 10, 50, 100],
					"gamma": ['scale', 'auto', 1, 5, 10]
				}
			),
			"K-Nearest Neighbors": (
				KNeighborsClassifier(algorithm='auto'),
				{
					"p": [1, 2, 3],
					"n_neighbors": [1, 2, 5, 10, 15, 20, 25, 30],
					"leaf_size": [1, 2, 5, 10, 15, 20, 30, 35, 40, 45, 50],
					"weights": ['uniform', 'distance'],
					"metric": ['minkowski', 'chebyshev']
				}
			),
			"Neural Network": (
				MLPClassifier(),
				{
					"solver": ['lbfgs', 'sgd', 'adam'],
					"hidden_layer_sizes": [(10,), (13,), (15,), (20, 15, 2)],
					"learning_rate": ['constant', 'invscaling', 'adaptive']
				}
			)
		}