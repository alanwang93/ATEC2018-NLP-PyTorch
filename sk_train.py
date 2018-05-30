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

logger = init_log('log/sk_model.log')

rebuild = False
train_data_path = 'data/processed/train.pkl'
valid_data_path = 'data/processed/valid.pkl'
sk_train_path = 'data/processed/sk_train.pkl'
sk_valid_path = 'data/processed/sk_valid.pkl'

# features = ['s1_wlen', 's1_clen', 'jaccard_char_unigram', 'jaccard_char_bigram',\
#  			'jaccard_char_trigram', 'jaccard_word_unigram', 'LevenshteinDistance_char',\
#  			'LevenshteinDistance_word', 'word_bool']
features = ['s1_wlen', 's1_clen', 'jaccard_char_unigram', 'jaccard_char_bigram',\
 			'jaccard_char_trigram', 'jaccard_word_unigram',# 'LevenshteinDistance_char',\
 			'LevenshteinDistance_word', 'word_bool']

logger.info("Loading training data")
if rebuild or not os.path.exists(sk_train_path):
    train_data = pickle.load(open(train_data_path, 'r'))
    # print(train_data['word_bool'][1].shape)
    logger.info("Processing training data")
    train_X_features = []
    for feat in features:
        if train_data[feat][0] == 's':
            train_X_features.append(abs(train_data[feat][1] - train_data[feat.replace('1', '2')][1]).reshape(-1,1))
        elif train_data[feat][0] == 'p':
            if feat == 'word_bool':
                train_X_features.append(np.squeeze(train_data[feat][1]))
            elif len(train_data[feat][1].shape) == 1:
                train_X_features.append(train_data[feat][1].reshape(-1,1))
            else:
                train_X_features.append(np.squeeze(train_data[feat][1]))
    train_X = np.concatenate(train_X_features, axis=1)
    train_y = train_data['label'][1]
    del train_data
    pickle.dump({"X":train_X, "y":train_y}, open(sk_train_path, 'w'))
else:
    train = pickle.load(open(sk_train_path, 'r'))
    train_X = train['X']
    train_y = train['y']
    del train

print("Number of features", train_X.shape[1])

logger.info("Loading valid data")
if rebuild or not os.path.exists(sk_valid_path):
    valid_data = pickle.load(open(valid_data_path, 'r'))
    logger.info("Processing valid data")
    valid_X_features = []
    for feat in features:
        if valid_data[feat][0] == 's':
            valid_X_features.append(abs(valid_data[feat][1] - valid_data[feat.replace('1', '2')][1]).reshape(-1,1))
        elif valid_data[feat][0] == 'p':
            if feat == 'word_bool':
                valid_X_features.append(np.squeeze(valid_data[feat][1]))
            elif len(valid_data[feat][1].shape) == 1:
                valid_X_features.append(valid_data[feat][1].reshape(-1,1))
            else:
                valid_X_features.append(np.squeeze(valid_data[feat][1]))
    valid_X = np.concatenate(valid_X_features, axis=1)
    valid_y = valid_data['label'][1]
    del valid_data
    pickle.dump({"X":valid_X, "y":valid_y}, open(sk_valid_path, 'w'))
else:
    valid = pickle.load(open(sk_valid_path, 'r'))
    valid_X = valid['X']
    valid_y = valid['y']
    del valid

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
    clf.fit(train_X, train_y)
    # score = clf.score(valid_X, valid_y)
    print(valid_X.shape)
    print(valid_y.shape)
    pred = clf.predict(valid_X)
    print(pred.shape)
    preds.append(pred)
    print(c['clf'])
    print("score", clf.score(valid_X, valid_y))
    print()
print(preds)
proba = np.mean(preds, axis=0)
print(proba)
print("score", score(proba, valid_y))
