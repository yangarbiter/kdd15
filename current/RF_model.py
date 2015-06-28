
from generate_feature import *
from model import *
import cPickle, csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.datasets import dump_svmlight_file
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
import pandas as pd

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

with open('./data/0610/label_train+blend+valid.npy') as f:
    y = np.load(f)

trainy = loadans()


def train_pred(trainX, y, predX, eidfile, dest):
#clf = CalibratedClassifierCV(
#    base_estimator=RandomForestClassifier(
#            criterion='entropy',
#            n_estimators = 3000,
#            #class_weight = "auto",
#            min_samples_split = 30,
#            max_depth = 15,
#            n_jobs=-1
#        )
#    )
    clf = RandomForestClassifier(
            criterion='gini',
            n_estimators = 1000,
            #class_weight = "auto",
            min_samples_split = 30,
            max_depth = 15,
            n_jobs=-1
        )
    #clf = ExtraTreesClassifier(
    #        criterion='entropy',
    #        n_estimators=1000,
    #        max_depth = 15,
    #        n_jobs=-1
    #    )
    clf.fit(trainX, y[:len(trainX)])
    print roc_auc_score(y[len(trainX):len(trainX)+len(predX)], clf.predict_proba(predX)[:, 1])
    outputans(clf.predict_proba(predX)[:, 1], eidfile, dest)


train_pred(
    trainX,
    y,
    blendX,
    '/tmp2/kdd/enrollment_blend.csv',
    '/tmp2/b01902066/KDD/kdd15/blending/preds/610_nocali/rf_blend.csv'
)
exit(0

train_pred(
    np.vstack((trainX, blendX)),
    y,
    valX,
    '/tmp2/b01902066/KDD/data/internal1/enrollment_test.csv',
    '/tmp2/b01902066/KDD/kdd15/blending/valpreds/610_nocali/rf_val.csv'
)

train_pred(
    np.vstack((trainX, blendX, valX)),
    y,
    testX,
    '/tmp2/b01902066/KDD/data/enrollment_test.csv',
    '/tmp2/b01902066/KDD/kdd15/blending/testpreds/610_nocali/rf_test.csv'
)
