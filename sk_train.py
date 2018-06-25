import os
import cPickle as pickle
import numpy as np
from utils import init_log, score
from models import SKModel
from sklearn.model_selection import train_test_split
from data.features import Features
import xgboost as xgb

logger = init_log('log/sk_model.log')

rebuild = True
feats = Features()
feats._load(mode='train')
feat_names = ['label', 'power_words']
pf_names = ['s1_char_lsa100', 's2_char_lsa100', 's1_word_lsa100', 's2_word_lsa100']
        
F, d = feats.get_feats_by_name(feat_names, return_dict=False)
pF, pd = feats.get_feats_by_name(pf_names, return_dict=True)
y = F[:,0]
X = np.concatenate((F[:,1:], pF['s1_char_lsa100'] * pF['s2_char_lsa100'], pF['s1_word_lsa100'] * pF['s2_word_lsa100']), 1)
print(d)

# Pair level or sentence level features
# features = ['s1_wlen', 's1_clen', 'jaccard_char_unigram', 'jaccard_char_bigram',\
#  			'jaccard_char_trigram', 'jaccard_word_unigram',# 'LevenshteinDistance_char',\
#  			'LevenshteinDistance_word', 'word_bool', 's1_word_lsa']
# features = ['s1_word_lsa']

logger.info("Loading training data")
# if rebuild or not os.path.exists(sk_train_path):
#     data = pickle.load(open(train_data_path, 'r'))
#     logger.info("Processing training data")
#     X_features = []
#     for feat in features:
#         #if 'lsa' in feat:
#         #    X_features.append(data[feat][1])
#         if data[feat][0] == 's':
#             X_features.append(np.power(data[feat][1] - data[feat.replace('1', '2')][1], 2).reshape(-1, data[feat][2]))
#         elif data[feat][0] == 'p':
#             if feat == 'word_bool':
#                 X_features.append(np.squeeze(data[feat][1]))
#             elif len(data[feat][1].shape) == 1:
#                 X_features.append(data[feat][1].reshape(-1,1))
#             else:
#                 X_features.append(np.squeeze(data[feat][1]))

#     for f in X_features:
#         print(f.shape)
#     X = np.concatenate(X_features, axis=1)
#     y = data['label'][1]
#     del data
#     pickle.dump({"X":X, "y":y}, open(sk_train_path, 'w'))
# else:
#     data = pickle.load(open(sk_train_path, 'r'))
#     X = data['X']
#     y = data['y']
#     del data

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
    },
    {
        'module': 'discriminant_analysis',
        'clf': 'LinearDiscriminantAnalysis',
        'kwargs': {
        }
    }
]
from sklearn.linear_model import LogisticRegression as LR
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import SGDClassifier as SGD
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import ExtraTreesClassifier as ET
clfs = [
        #LR(class_weight='balanced', n_jobs=1),
        #LDA(),
        #SGD(loss='log', penalty='l2', alpha=0.0001, class_weight='balanced'),
        #DT(max_depth=5, max_features=0.8, class_weight='balanced'),
        xgb.XGBClassifier(n_estimators=400, max_depth=6, scale_pos_weight=4., n_jobs=5, subsample=0.8)
        #RF(20, max_features='auto', class_weight='balanced', max_depth=6, n_jobs=5),
        #RF(20, criterion='entropy', max_features='auto', class_weight='balanced', max_depth=6, n_jobs=5),
        #ET(100, max_features='auto', class_weight='balanced', max_depth=4, min_samples_split=5, n_jobs=5, bootstrap=True, max_leaf_nodes=50),



    ]
preds = []
for c in clfs:
    clf = SKModel(c)
    # print(clf.clf)
    print(c)
    clf.fit(X_train, y_train)
    # score = clf.score(valid_X, valid_y)
    pred = clf.predict(X_valid)
    preds.append(pred)
    s = clf.score(X_valid, y_valid)
    print('F1:{0}, P:{1}, R:{2}'.format(s[0], s[2], s[3]))
    print()
