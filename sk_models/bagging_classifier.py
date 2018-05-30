from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from utils import score

class Bagging():

    def __init__(self):
    	base = LogisticRegression(class_weight='balanced', C=10.)
    	# base = DecisionTreeClassifier(max_depth=4, class_weight='balanced')
        self.reg = BaggingClassifier(base_estimator=base, n_estimators=30, max_samples=0.6)

    def fit(self, X, y):
        self.reg.fit(X, y)

    def predict(self, X):
        return self.reg.predict(X)

    def score(self, X, y):
        pred = self.predict(X)
        return score(pred, y)
