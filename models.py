from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier


class Models:
	def __init__(self):
		self.models = {
			"RandomForest": (
				RandomForestClassifier(),
				{
					"criterion": ["gini", "entropy"],
					"n_estimators": [50, 100, 200],
					"max_features": [None, "sqrt", 0.3, 0.4, 0.5, 0.6]
				}
			),
			"DecisionTree": (
				DecisionTreeClassifier(),
				{
					"max_depth": [None, 4, 10],
					"min_samples_leaf": [1, 10, 20],
					"max_features": ["sqrt", "log2", None, 0.2, 0.3]
				}
			),
			"SVM linear": (
				LinearSVC(),
				{

				}
			),
			"SVM RBF": (
				SVC(),
				{

				}
			),
			"Nearest Neighbor": (
				KNeighborsClassifier(),
				{

				}
			)
		}