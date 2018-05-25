from sklearn.ensemble import BaggingClassifier
from utils import score

class Bagging():

    def __init__(self):
        self.reg = BaggingClassifier(n_estimators=100)

    def fit(self, X, y):
        self.reg.fit(X, y)

    def predict(self, X):
        return self.reg.predict(X)

    def score(self, X, y):
        pred = self.predict(X)
        return score(pred, y)
