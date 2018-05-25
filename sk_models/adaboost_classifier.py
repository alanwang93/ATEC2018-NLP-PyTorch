from sklearn.ensemble import AdaBoostClassifier
from utils import score

class AdaBoost():

    def __init__(self):
        self.reg = AdaBoostClassifier(n_estimators=100)

    def fit(self, X, y):
        self.reg.fit(X, y)

    def predict(self, X):
        return self.reg.predict(X)

    def score(self, X, y):
        pred = self.predict(X)
        return score(pred, y)
