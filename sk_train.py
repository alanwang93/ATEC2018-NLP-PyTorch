import os, pickle
import numpy as np
from sk_models.logistic_regressor import LogisticRegressor
from sk_models.random_forest_classifier import RandomForest
from sk_models.bagging_classifier import Bagging
from sk_models.adaboost_classifier import AdaBoost
from sk_models.mlp_classifier import MLP

train_data_path = 'data/processed/train.pkl'
valid_data_path = 'data/processed/valid.pkl'

# features = ['s1_wlen', 's1_clen', 'jaccard_char_unigram', 'jaccard_char_bigram',\
#  			'jaccard_char_trigram', 'jaccard_word_unigram', 'LevenshteinDistance_char',\
#  			'LevenshteinDistance_word', 'word_bool']
features = ['s1_wlen', 's1_clen', 'jaccard_char_unigram', 'jaccard_char_bigram',\
 			'jaccard_char_trigram', 'jaccard_word_unigram',# 'LevenshteinDistance_char',\
 			'LevenshteinDistance_word', 'word_bool']

# train data preprocessing
train_data = pickle.load(open(train_data_path, 'r'))
# print(train_data['word_bool'][1].shape)
train_X_features = []
for feat in features:
    if train_data[feat][0] == 's':
        train_X_features.append(abs(train_data[feat][1] - train_data[feat.replace('1', '2')][1]).reshape(-1,1))
    elif train_data[feat][0] == 'p':
    	if len(train_data[feat][1].shape) == 1:
        	train_X_features.append(train_data[feat][1].reshape(-1,1))
        else:
        	train_X_features.append(np.squeeze(train_data[feat][1]))
train_X = np.concatenate(train_X_features, axis=1)
train_y = train_data['label'][1]

# valid data preprocessing
valid_data = pickle.load(open(valid_data_path, 'r'))
valid_X_features = []
for feat in features:
    if valid_data[feat][0] == 's':
        valid_X_features.append(abs(valid_data[feat][1] - valid_data[feat.replace('1', '2')][1]).reshape(-1,1))
    elif valid_data[feat][0] == 'p':
    	if len(valid_data[feat][1].shape) == 1:
        	valid_X_features.append(valid_data[feat][1].reshape(-1,1))
        else:
        	valid_X_features.append(np.squeeze(valid_data[feat][1]))

valid_X = np.concatenate(valid_X_features, axis=1)
valid_y = valid_data['label'][1]

clf = Bagging()
clf.fit(train_X, train_y)
score = clf.score(valid_X, valid_y)

print("score", score)
