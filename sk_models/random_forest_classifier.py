from sklearn.ensemble import RandomForestClassifier
from utils import score

class RandomForest():

    def __init__(self):
        self.reg = RandomForestClassifier(n_estimators=500, class_weight={1:4., 0:1.}, max_features=None)

    def fit(self, X, y):
        self.reg.fit(X, y)

    def predict(self, X):
        return self.reg.predict(X)

    def score(self, X, y):
        pred = self.predict(X)
        return score(pred, y)