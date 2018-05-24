from sklearn.linear_model import LogisticRegression
from utils import score

class LogisticRegressor():

    def __init__(self):
        self.reg = LogisticRegression(class_weight='balanced', C=0.1)

    def fit(self, X, y):
        self.reg.fit(X, y)

    def predict(self, X):
        return self.reg.predict(X)

    def score(self, X, y):
        pred = self.predict(X)
        return score(pred, y)
