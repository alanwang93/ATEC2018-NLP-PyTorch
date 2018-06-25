import os
import cPickle as pickle
import numpy as np
from utils import init_log, score
from models import SKModel
from sklearn.model_selection import train_test_split
from data.features import Features
logger = init_log('log/sk_model.log')

rebuild = True
feats = Features()
feats._load(mode='train')
feat_names = ['word_bool']
        
X, d = feats.get_feats_by_name(feat_names, return_dict=False)
print(d)

# deep_features = np.load('data/processed/features_siamese_default_best.npy')
# X = np.concatenate((X, deep_features), axis=1)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=666)

train_idx = np.load('data/processed/train_idx.npy')
valid_idx = np.load('data/processed/valid_idx.npy')

X_train = X[train_idx]
X_valid = X[valid_idx]
y_train = y[train_idx]
y_valid = y[valid_idx]

print("Number of features", X_train.shape[1])

configs = [
    {
        'module': 'linear_model',
        'clf': 'LogisticRegression',
        'kwargs': {
            'class_weight': 'balanced', 
            'C':1.
        }
    }
]
preds = []
for c in configs:
    clf = SKModel(c)
    # print(clf.clf)
    clf.fit(X_train, y_train)
    # score = clf.score(valid_X, valid_y)
    pred = clf.predict(X_test)
    preds.append(pred)
    print(c['clf'])
    print("score", clf.score(X_test, y_test))
    print()
