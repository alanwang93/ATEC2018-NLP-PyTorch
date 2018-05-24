from sklearn.ensemble import RandomForestClassifier
from utils import score

class RandomForest():

    def __init__(self):
        self.reg = RandomForestClassifier(class_weight={1:10, 0:1})

    def fit(self, X, y):
        self.reg.fit(X, y)

    def predict(self, X):
        return self.reg.predict(X)

    def score(self, X, y):
        pred = self.predict(X)
        return score(pred, y)
