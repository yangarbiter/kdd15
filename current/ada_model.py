
from generate_feature import *
from model import *
import cPickle, csv
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import dump_svmlight_file
from sklearn.metrics import roc_auc_score
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

def loadans():
    trainy = []
    with open('/tmp2/kdd/truth_train.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        for row in reader:
            enrole_id = int(row[0])
            trainy.append(int(row[1]))
    return trainy

def outputans(ans, id_file_path, path):
    idxs = []
    with open(id_file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[0] == 'enrollment_id': continue
            idxs.append(int(row[0]))

    with open(path, 'w') as f:
        for i, enrole_id in enumerate(idxs):
            f.write(str(enrole_id) + ',' + str(ans[i]) + '\n')

with open('./data/0610/train_feature.npy', 'r') as f:
    trainX = np.load(f)

with open('./data/0610/blend_feature.npy', 'r') as f:
    blendX = np.load(f)

with open('./data/0610/val_feature.npy', 'r') as f:
    valX = np.load(f)

with open('./data/0610/test_feature.npy', 'r') as f:
    testX = np.load(f)

#with open('fet.pkl', 'r') as f:
#    train_X, test_X = cPickle.load(f)

with open('./data/0610/label_train+blend+valid.npy') as f:
    y = np.load(f)

#trainy = loadans()

#clf = CalibratedClassifierCV(
#        base_estimator=AdaBoostClassifier(
#            base_estimator = DecisionTreeClassifier(
#                    criterion='entropy',
#                    max_depth=3,
#                ),
#            learning_rate = 0.05,
#            n_estimators = 600,
#        ),
#        method='sigmoid', 
#        cv=5
#    )
clf=AdaBoostClassifier(
    base_estimator = DecisionTreeClassifier(
            criterion='entropy',
            max_depth=3,
        ),
    learning_rate = 0.05,
    n_estimators = 600,
)
trainy = y[:len(trainX)]
clf.fit(trainX, trainy)
outputans(clf.predict_proba(blendX)[:, 1],
    '/tmp2/kdd/enrollment_blend.csv',
    '/tmp2/b01902066/KDD/kdd15/blending/preds/610_nocali/adaboost_blend.csv')

clf=AdaBoostClassifier(
    base_estimator = DecisionTreeClassifier(
            criterion='entropy',
            max_depth=3,
        ),
    learning_rate = 0.05,
    n_estimators = 600,
)
trainX = np.vstack((trainX, blendX))
trainy = y[:len(trainX)]
clf.fit(trainX, trainy)
outputans(clf.predict_proba(valX)[:, 1],
    '/tmp2/b01902066/KDD/data/internal1/enrollment_test.csv',
    '/tmp2/b01902066/KDD/kdd15/blending/valpreds/610_nocali/adaboost_val.csv')

clf=AdaBoostClassifier(
    base_estimator = DecisionTreeClassifier(
            criterion='entropy',
            max_depth=3,
        ),
    learning_rate = 0.05,
    n_estimators = 600,
)
trainX = np.vstack((trainX, valX))
trainy = y
clf.fit(trainX, trainy)
outputans(clf.predict_proba(testX)[:, 1],
    '/tmp2/b01902066/KDD/data/enrollment_test.csv',
    '/tmp2/b01902066/KDD/kdd15/blending/testpreds/610_nocali/adaboost_test.csv')
