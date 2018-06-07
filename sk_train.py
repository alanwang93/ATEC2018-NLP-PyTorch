import os
import cPickle as pickle
import numpy as np
from utils import init_log, score
from sk_models.logistic_regressor import LogisticRegressor
from sk_models.random_forest_classifier import RandomForest
from sk_models.bagging_classifier import Bagging
from sk_models.adaboost_classifier import AdaBoost
from sk_models.mlp_classifier import MLP
from sk_models.sk_model import SKModel
from sklearn.model_selection import train_test_split

logger = init_log('log/sk_model.log')

rebuild = True#False
train_data_path = 'data/processed/train.pkl'
sk_train_path = 'data/processed/sk_train.pkl'

# Pair level or sentence level features
features = ['s1_wlen', 's1_clen', 'jaccard_char_unigram', 'jaccard_char_bigram',\
 			'jaccard_char_trigram', 'jaccard_word_unigram',# 'LevenshteinDistance_char',\
 			'LevenshteinDistance_word', 'word_bool', 's1_word_tfidf']

logger.info("Loading training data")
if rebuild or not os.path.exists(sk_train_path):
    data = pickle.load(open(train_data_path, 'r'))
    logger.info("Processing training data")
    X_features = []
    for feat in features:
        if data[feat][0] == 's':
            if feat == 's1_word_tfidf':
                X_features.append(abs(data[feat][1] - data[feat.replace('1', '2')][1]).toarray().reshape(-1,data[feat][2]))
            else:
                X_features.append(abs(data[feat][1] - data[feat.replace('1', '2')][1]).reshape(-1,data[feat][2]))
        elif data[feat][0] == 'p':
            if feat == 'word_bool':
                X_features.append(np.squeeze(data[feat][1]))
            elif len(data[feat][1].shape) == 1:
                X_features.append(data[feat][1].reshape(-1,1))
            else:
                X_features.append(np.squeeze(data[feat][1]))

    # word, char level features
    print("s1_word_tfidf shape",  data['s1_word_tfidf'][1].shape)
    print("s1_char_tfidf shape",  data['s1_char_tfidf'][1].shape)
    for f in X_features:
        print(f.shape)
    X = np.concatenate(X_features, axis=1)
    y = data['label'][1]
    del data
    pickle.dump({"X":X, "y":y}, open(sk_train_path, 'w'))
else:
    data = pickle.load(open(sk_train_path, 'r'))
    X = data['X']
    y = data['y']
    del data

print("Number of features", train_X.shape[1])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=666)

configs = [
    {
        'module': 'linear_model',
        'clf': 'LogisticRegression',
        'kwargs': {
            'class_weight': 'balanced', 
            'C':100.
        }
    },
    {
        'module': 'ensemble',
        'clf': 'AdaBoostClassifier',
        'kwargs': {
            'n_estimators': 100
        }
    },
    {
        'module': 'ensemble',
        'clf': 'RandomForestClassifier',
        'kwargs': {
            'n_estimators': 100, 
            'class_weight': 'balanced', 
            'max_features': 1.0
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
print(preds)
proba = np.mean(preds, axis=0)
print("score", score(proba, y_valid))
