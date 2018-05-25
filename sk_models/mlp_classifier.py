from sklearn.neural_network import MLPClassifier
from utils import score

class MLP():

    def __init__(self):
        self.reg = MLPClassifier()

    def fit(self, X, y):
        self.reg.fit(X, y)

    def predict(self, X):
        return self.reg.predict(X)

    def score(self, X, y):
        pred = self.predict(X)
        return score(pred, y)
